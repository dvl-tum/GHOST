from numpy.lib import arraysetops
import torch
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
import copy
import torch.nn as nn
from torch.autograd import Variable
from src.kalman import KalmanFilter
from cython_bbox import bbox_overlaps as bbox_ious
import torch.nn.functional as F


mot_fps = {
    'MOT17-13-SDP': 25,
    'MOT17-11-SDP': 30,
    'MOT17-10-SDP': 30,
    'MOT17-09-SDP': 30,
    'MOT17-05-SDP': 14,
    'MOT17-02-SDP': 30,
    'MOT17-04-SDP': 30,
    'MOT17-13-DMP': 25,
    'MOT17-11-DMP': 30,
    'MOT17-10-DMP': 30,
    'MOT17-09-DMP': 30,
    'MOT17-05-DMP': 14,
    'MOT17-02-DMP': 30,
    'MOT17-04-DMP': 30,
    'MOT17-13-FRCNN': 25,
    'MOT17-11-FRCNN': 30,
    'MOT17-10-FRCNN': 30,
    'MOT17-09-FRCNN': 30,
    'MOT17-05-FRCNN': 14,
    'MOT17-02-FRCNN': 30,
    'MOT17-04-FRCNN': 30,
}


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def bisoftmax(x, y):
    feats = torch.mm(x, y.t())/0.1
    d2t_scores = feats.softmax(dim=1)
    t2d_scores = feats.softmax(dim=0)
    scores = (d2t_scores + t2d_scores) / 2
    return scores.numpy()


def get_proxy(curr_it, mode='inact', tracker_cfg=None, mv_avg=None):
    feats = list()
    avg = tracker_cfg['avg_' + mode]['num']
    proxy = tracker_cfg['avg_' + mode]['proxy']

    if avg != 'all':
        if proxy != 'mv_avg':
            avg = int(avg)
        else:
            avg = float(avg)

    for i, it in curr_it.items():
        # take last bb only
        if proxy == 'last':
            f = it.past_feats[-1]

        # take the first only
        elif avg == 'first':
            f = it.past_feats[0]

        # moving average of features
        elif proxy == 'mv_avg':
            if i not in mv_avg.keys():
                f = it.past_feats[-1]
            else:
                f = mv_avg[i] * avg + it.past_feats[-1] * (1-avg)
            mv_avg[i] = f

        # take all if all or number of features < avg
        elif avg == 'all' or len(it.past_feats) < avg:
            if proxy == 'mean':
                f = torch.mean(torch.stack(it.past_feats), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack(it.past_feats), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack(it.past_feats), dim=0)[0]
            elif proxy == 'meannorm':
                f = F.normalize(torch.mean(torch.stack(
                    it.past_feats), dim=0), p=2, dim=0)

        # get proxy of last avg number of frames
        else:
            if proxy == 'mean':
                f = torch.mean(torch.stack(it.past_feats[-avg:]), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack(it.past_feats[-avg:]), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack(it.past_feats[-avg:]), dim=0)[0]
            elif proxy == 'meannorm':
                f = F.normalize(torch.mean(torch.stack(
                    it.past_feats[-avg:]), dim=0), p=2, dim=1)

        feats.append(f)

    if len(feats[0].shape) == 1:
        feats = torch.stack(feats)
    elif len(feats[0].shape) == 3:
        feats = torch.cat([f.unsqueeze(0) for f in feats], dim=0)
    elif len(feats) == 1:
        feats = feats
    else:
        feats = torch.cat(feats, dim=0)

    return feats


def get_center(pos):
    # adapted from tracktor
    if len(pos.shape) <= 1:
        x1 = pos[0]
        y1 = pos[1]
        x2 = pos[2]
        y2 = pos[3]
    else:
        x1 = pos[:, 0]
        y1 = pos[:, 1]
        x2 = pos[:, 2]
        y2 = pos[:, 3]
    if type(pos) == torch.Tensor:
        return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()
    else:
        return np.array([(x2 + x1) / 2, (y2 + y1) / 2])


def get_width(pos):
    # adapted from tracktor
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    # adapted from tracktor
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    # adapted from tracktor
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    # adapted from tracktor
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).numpy()


def bbox_overlaps(boxes, query_boxes):
    # adapted from tracktor
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        # If input is ndarray, turn the overlaps back to ndarray when return
        def out_fn(x): return x.numpy()
    else:
        def out_fn(x): return x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
        (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
        (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t(
    )) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t(
    )) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def is_moving(seq, log=False):

    if "MOT" not in seq and 'dance' not in seq:
        return True
    elif 'dance' in seq:
        return False
    elif seq.split('-')[1] in ['13', '11', '10', '05', '14', '12', '07', '06']:
        return True
    elif seq.split('-')[1] in ['09', '04', '02', '08', '03', '01']:
        return False
    else:
        assert False, 'Seqence not valid {}'.format(seq)


def frame_rate(seq):
    if 'dance' in seq:
        return 20
    elif seq.split('-')[1] in ['11', '10', '12', '07',  '09', '04', '02', '08', '03', '01']:
        return 30
    elif seq.split('-')[1] in ['05', '06']:
        return 14
    else:
        return 25


def tlrb_to_xyah(tlrb):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlrb).copy()
    ret[2] = ret[2] - ret[0]
    ret[3] = ret[3] - ret[1]
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


class Track():
    def __init__(
            self,
            track_id,
            bbox, feats,
            im_index,
            gt_id,
            vis,
            conf,
            frame,
            label,
            kalman=False,
            kalman_filter=None):
        self.kalman = kalman
        self.xyah = tlrb_to_xyah(copy.deepcopy(bbox))
        if self.kalman:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(
                measurement=tlrb_to_xyah(bbox))
        self.track_id = track_id
        self.pos = bbox
        self.bbox = list()
        self.bbox.append(bbox)

        # init variables for motion model
        self.last_pos = list()
        self.last_pos.append(self.pos)
        self.last_v = np.array([0, 0, 0, 0])
        self.last_vc = np.array([0, 0])

        # embedding feature list of detections
        self.past_feats = list()
        self.feats = feats
        self.past_feats.append(feats)

        # image index list of detections
        self.past_im_indices = list()
        self.past_im_indices.append(im_index)
        self.im_index = im_index

        # corresponding gt ids of detections
        self.gt_id = gt_id
        self.past_gt_ids = list()
        self.past_gt_ids.append(gt_id)

        # corresponding gt visibilities of detections
        self.gt_vis = vis
        self.past_gt_vis = list()
        self.past_gt_vis.append(vis)

        # initialize inactive count
        self.inactive_count = 0
        self.past_frames = list()
        self.past_frames.append(frame)

        # conf of current det
        self.conf = conf

        self.past_vs = list()

        # labels of detections
        self.label = list()
        self.label.append(label)

    def __len__(self):
        return len(self.last_pos)

    def add_detection(
            self, bbox, feats, im_index, gt_id, vis, conf, frame, label):
        # update all lists / states
        self.pos = bbox
        self.last_pos.append(bbox)
        self.bbox.append(bbox)

        self.feats = feats
        self.past_feats.append(feats)

        self.past_im_indices.append(im_index)
        self.im_index = im_index

        self.gt_id = gt_id
        self.past_gt_ids.append(gt_id)
        self.gt_vis = vis
        self.past_gt_vis.append(vis)

        self.conf = conf
        self.past_frames.append(frame)
        self.label.append(label)

        if self.kalman:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, tlrb_to_xyah(bbox))

    def update_v(self, v):
        self.past_vs.append(v)
        self.last_v = v

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self.bbox.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


def multi_predict(active, inactive, shared_kalman):
    state = list()
    act = [v for v in active.values()]
    state.extend([1]*len(act))
    inact = [v for v in inactive.values()]
    state.extend([0]*len(inact))
    stracks = act + inact
    if len(stracks) > 0:
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, (st, act) in enumerate(zip(stracks, state)):
            if act != 1:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    return stracks


def get_iou_kalman(tracklets, detections):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """

    atlbrs = [track.tlbr for track in tracklets]
    btlbrs = [track['bbox'] for track in detections]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix.T


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious
