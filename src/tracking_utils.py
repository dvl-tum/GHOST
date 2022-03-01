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


def add_ioa(tracks, seq, interaction=None, occlusion=None, frame_size=None):
    curr_interaction, curr_occlusion = 0, 0
    if len(tracks) > 0:
        bbs = torch.from_numpy(np.vstack([tr['bbox'] for tr in tracks]))
        # add dummy bbs for outside of frame bounding boxes
        bbs, foot_pos = add_dummy_bb(bbs, frame_size)

        inter, area, _, inter_bbs = _box_inter_area(bbs, bbs)
        ioa = inter / np.atleast_2d(area).T
        # ioa = ioa - np.eye(ioa.shape[0])
        np.fill_diagonal(ioa, 0)

        # not taking foot position into account --> ioa_nofoot
        curr_interaction = copy.deepcopy(ioa)
        curr_interaction = remove_intersection(curr_interaction, inter_bbs, foot_pos, bbs, area)

        # get number of interactors
        num_inter = copy.deepcopy(curr_interaction)
        num_inter = np.where(num_inter > 0, 1, 0)
        num_inter = np.sum(num_inter, axis=1)

        # remove dummy bbs as not important for interaction
        curr_interaction = curr_interaction[:-4, :-4]
        ioa_nf = np.sum(curr_interaction, axis=1)
        ioa_nf = np.round(ioa_nf*100)/100
        ioa_nf = ioa_nf.tolist()
        if interaction is not None:
            interaction[seq] += ioa_nf

        # taking foot position into account
        bot = np.atleast_2d(foot_pos).T < np.atleast_2d(foot_pos)
        ioa[~bot] = 0
        ioa = remove_intersection(ioa, inter_bbs, foot_pos, bbs, area)

        # remove rows of dummy bounding boxes
        ioa = ioa[:-4, :]

        # get number of occluders
        num_occ = copy.deepcopy(ioa)
        num_occ = np.where(num_occ > 0, 1, 0)
        num_occ = np.sum(num_occ, axis=1)

        curr_occlusion = ioa
        ioa = np.sum(ioa, axis=1)
        ioa = np.round(ioa*100)/100
        ioa = ioa.tolist()
        if occlusion is not None:
            occlusion[seq] += ioa
        
        for i, s, ni, ns, t in zip(ioa, num_occ, ioa_nf, num_inter, tracks):
            assert i <= 1, "ioa cannot be larger than 1 {}".format(ioa)
            assert ni <= 1, "interaction cannot be larger than 1 {}".format(ioa_nf)
            t['ioa'] = min([1, i])
            t['num_occ'] = s
            t['ioa_nf'] = ni
            t['num_inter'] = ns

    return curr_interaction, curr_occlusion


def add_dummy_bb(bbs, frame_size):
    # frame_size = [H, B]
    left = torch.clip(torch.min(bbs[:, 0]), min=None, max=-1)
    top = torch.clip(torch.min(bbs[:, 1]), min=None, max=-1)
    right = torch.clip(torch.min(bbs[:, 2]), min=frame_size[1]+1, max=None)
    bottom = torch.clip(torch.min(bbs[:, 3]), min=frame_size[0]+1, max=None)

    # make dummy foot position to ensure dummy bbs are in foreground
    foot_position = bbs[:, 3]
    dummy_bottom = torch.tensor([bottom, bottom, bottom, bottom])
    foot_position = torch.cat([foot_position, dummy_bottom])

    dummies = torch.tensor([
        [left, 0, 0, frame_size[0]],
        [0, top, frame_size[1], 0],
        [frame_size[1], 0, right, frame_size[0]],
        [0, frame_size[0], frame_size[1], bottom]
    ])
    bbs = torch.cat([bbs, dummies])

    return bbs, foot_position


def remove_intersection(ioa, inter_bbs, bottom_coord, bbs, areas):
    # sorted indices of foot position, biggest first (foreground first)
    sorted_bot = np.argsort(bottom_coord.numpy())
    sorted_bot = np.flip(sorted_bot)
    # from foreground to background
    for i, index in enumerate(sorted_bot):
        # foreground not occluded, second only occluded by foreground
        if i == 0:
            continue

        # levels up to now are in front of current level
        idx = sorted_bot[:i]
        for j, (ioa_j, inter_bbs_j) in enumerate(zip(ioa, inter_bbs)):
            
            if ioa_j[index] == 0:
                continue

            idx_j = idx[ioa_j[idx] != 0]
            inter, area, _, _ = _box_inter_area(
                torch.from_numpy(inter_bbs_j[idx_j, :]),
                torch.from_numpy(np.atleast_2d(inter_bbs_j[index, :])))

            # changed area to areas[j]
            ioa_inter = inter / np.atleast_2d(areas[j]).T
            ioa[j, index] = np.clip(
                ioa_j[index] - np.sum(np.squeeze(ioa_inter)),
                a_min=0,
                a_max=None)

    return ioa


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


def _box_inter_area(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # left, top, right, bot
    inter_bbs = [
        torch.max(boxes1[:, None, 0], boxes2[:, 0]),
        torch.max(boxes1[:, None, 1], boxes2[:, 1]),
        torch.min(boxes1[:, None, 2], boxes2[:, 2]),
        torch.min(boxes1[:, None, 3], boxes2[:, 3])]
    inter_bbs = torch.stack(inter_bbs, dim=0)
    inter_bbs = inter_bbs.permute(1, 2, 0)

    return inter.numpy(), area1.numpy(), area2.numpy(), inter_bbs.numpy()


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
        
        # take last bb with min intersection over area
        elif proxy == 'min_ioa':
            ioa = [t['ioa'] for t in it]
            ioa.reverse()
            f = it.past_feats[-(ioa.index(min(ioa))+1)]

        # take all if all or number of features < avg
        elif avg == 'all' or len(it.past_feats) < avg:
            if proxy == 'mean':
                f = torch.mean(torch.stack(it.past_feats), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack(it.past_feats), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack(it.past_feats), dim=0)[0]

        # get proxy of last avg number of frames
        else:
            if proxy == 'mean':
                f = torch.mean(torch.stack(it.past_feats[-avg:]), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack(it.past_feats[-avg:]), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack(it.past_feats[-avg:]), dim=0)[0]
        
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


def get_reid_performance(topk=8, first_match_break=True, tracks=None):
    feats = torch.cat([t.past_feats.cpu().unsqueeze(0) for k, v in tracks.items() for t in v if t['id'] != -1], 0)
    lab = np.array([t['id'] for k, v in tracks.items() for t in v if t['id'] != -1])
    dist = sklearn.metrics.pairwise_distances(feats.numpy(), metric='cosine')
    m, n = dist.shape

    # Sort and find correct matches
    indices = np.argsort(dist, axis=1)
    indices = indices[:, 1:]
    matches = (lab[indices] == lab[:, np.newaxis])

    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0

    for i in range(m):
        if not np.any(matches[i, :]): continue

        index = np.nonzero(matches[i, :])[0]
        delta = 1. / (len(index))
        for j, k in enumerate(index):
            if k - j >= topk: break
            if first_match_break:
                ret[k - j] += 1
                break
            ret[k - j] += delta
        num_valid_queries += 1

    cmc = ret.cumsum() / num_valid_queries

    aps = []
    for k in range(dist.shape[0]):
        # Filter out the same id and same camera
        y_true = matches[k, :]

        y_score = -dist[k][indices[k]]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    
    return cmc, aps


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
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def is_moving(seq, log=False):
    if seq.split('-')[1] in ['13', '11', '10', '05', '14', '12', '07', '06']:
        if log:
            print('Seqence is moving {}'.format(seq))
        return True
    elif seq.split('-')[1] in ['09', '04', '02', '08', '03', '01']:
        if log:
            print('Seqence is not moving {}'.format(seq))
        return False
    else:
        assert False, 'Seqence not valid {}'.format(seq)

def frame_rate(seq):
    if seq.split('-')[1] in ['11', '10', '12', '07',  '09', '04', '02', '08', '03', '01']:
        return 30
    elif seq.split('-')[1] in ['05', '06']:
        return 14
    else:
        return 25


def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


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
    def __init__(self, track_id, bbox, feats, im_index, gt_id, vis, ioa, ioa_nf, area_out, num_occ, num_inter, conf, frame, area, kalman=False, kalman_filter=None):
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
        self.area = area

        # init variables for motion model
        self.last_pos = list()
        self.last_pos.append(self.pos)
        self.last_v = np.array([0, 0, 0, 0])
        self.last_vc = np.array([0, 0])

        # initialize ioa variables
        self.ioa = ioa
        self.past_ioa = list()
        self.past_ioa.append(ioa)
        self.interaction = ioa_nf
        self.past_interaction = list()
        self.past_interaction.append(ioa_nf)
        self.area_out = area_out
        self.past_areas_out = list()
        self.past_areas_out.append(area_out)
        self.num_occ = num_occ
        self.past_num_occ = list()
        self.past_num_occ.append(num_occ)
        self.num_inter = num_inter
        self.past_num_inter = list()
        self.past_num_inter.append(num_inter)

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
    
    def __len__(self):
        return len(self.last_pos)

    def add_detection(self, bbox, feats, im_index, gt_id, vis, ioa, ioa_nf, area_out, num_occ, num_inter, conf, frame, area):
        # update all lists / states
        self.pos = bbox
        self.last_pos.append(bbox)
        self.bbox.append(bbox)
        self.area = area

        self.ioa = 0.1 * ioa + 0.9 * self.ioa
        self.past_ioa.append(ioa)
        self.interaction = ioa_nf
        self.past_interaction.append(ioa_nf)
        self.area_out = area_out
        self.past_areas_out.append(area_out)
        self.num_occ = num_occ
        self.past_num_occ.append(num_occ)
        self.num_inter = num_inter
        self.past_num_inter.append(num_inter)

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


class WeightPredictor(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(WeightPredictor, self).__init__()
        self.lin1 = nn.Linear(input_dim, 15)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(15, output_dim)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.lin2(self.relu(self.lin1(x)))
        x = self.sm(x)
        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=0.6):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, dist, targets):
        mask = targets.type(torch.ByteTensor).cuda()

        # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = [], []
        for i in range(dist.shape[0]):
            mask = targets[i]
            if torch.nonzero(mask).shape[0] == 0:
                continue
            dist_ap.append(dist[i][mask].max().unsqueeze(dim=0)) #hp
            dist_an.append(dist[i][mask == 0].min().unsqueeze(dim=0)) #hn
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()

        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


def get_precision(dist, targets):
    dist_ap, dist_an = [], []
    for i in range(dist.shape[0]):
        mask = targets[i]
        if torch.nonzero(mask).shape[0] == 0:
            continue
        dist_ap.append(dist[i][mask].max().unsqueeze(dim=0)) #hp
        dist_an.append(dist[i][mask == 0].min().unsqueeze(dim=0)) #hn
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    prec = (dist_an.data > dist_ap.data).sum() * 1. / dist_an.size(0)
    return prec


def collate(batch):
    w1 = [item[0] for item in batch]
    w2 = [item[1] for item in batch]
    w3 = [item[2] for item in batch]
    same_class = [item[3] for item in batch]
    ioa = [item[4] for item in batch]
    emb = [item[4] for item in batch]

    return [w1, w2, w3, same_class, ioa, emb]


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
        multi_mean, multi_covariance = shared_kalman.multi_predict(multi_mean, multi_covariance)
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