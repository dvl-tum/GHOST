from numpy.lib import arraysetops
import torch
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
import copy


def add_ioa(tracks, seq, interaction, occlusion, frame_size):
    bbs = torch.from_numpy(np.vstack([tr['bbox'] for tr in tracks]))
    # add dummy bbs for outside of frame bounding boxes
    bbs, foot_pos = add_dummy_bb(bbs, frame_size)

    inter, area, _, inter_bbs = _box_inter_area(bbs, bbs)
    ioa = inter / np.atleast_2d(area).T
    ioa = ioa - np.eye(ioa.shape[0])

    # not taking foot position into account --> ioa_nofoot
    curr_interaction = copy.deepcopy(ioa)
    # remove dummy bbs as not important for interaction
    curr_interaction = curr_interaction[:-4, :-4]
    ioa_nf = np.sum(curr_interaction, axis=1).tolist()
    interaction[seq] += ioa_nf

    for i, t in zip(ioa_nf, tracks):
        t['ioa_nf'] = i

    # taking foot position into account
    bot = np.atleast_2d(foot_pos).T < np.atleast_2d(foot_pos)
    ioa[~bot] = 0
    ioa = remove_intersection(ioa, inter_bbs, foot_pos, bbs)

    # remove rows of dummy bounding boxes
    ioa = ioa[:-4, :]
    curr_occlusion = ioa
    ioa = np.sum(ioa, axis=1).tolist()
    occlusion[seq] += ioa
    for i, t in zip(ioa, tracks):
        if i > 1:
            print(ioa)
            quit()
        t['ioa'] = min([1, i])

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


def remove_intersection(ioa, inter_bbs, bottom_coord, bbs):
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
            ioa_inter = inter / np.atleast_2d(area).T
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
    feats = torch.mm(x, y.t())
    d2t_scores = feats.softmax(dim=1)
    t2d_scores = feats.softmax(dim=0)
    scores = (d2t_scores + t2d_scores) / 2
    return scores.numpy()


def get_proxy(curr_it, mode='inact', tracker_cfg=None, mv_avg=None):
    feats = list()
    avg = tracker_cfg['avg_' + mode]['num'] 
    proxy = tracker_cfg['avg_' + mode]['proxy']

    if proxy != 'mv_avg':
        avg = int(avg)
    else:
        avg = float(avg)

    for i, it in curr_it.items():
        # take last bb only
        if proxy == 'last':
            f = it[-1]['feats']
        
        # take the first only
        elif avg == 'first':
            f = it[0]['feats']

        # moving average of features        
        elif proxy == 'mv_avg':
            if i not in mv_avg.keys():
                f = it[-1]['feats']
            else:
                f = mv_avg[i] * avg + it[-1]['feats'] * (1-avg) 
            mv_avg[i] = f
        
        # take last bb with min intersection over area
        elif proxy == 'min_ioa':
            ioa = [t['ioa'] for t in it]
            ioa.reverse()
            f = [t['feats'] for t in it][-(ioa.index(min(ioa))+1)]

        # take all if all or number of features < avg
        elif avg == 'all' or len(it) < avg:
            if proxy == 'mean':
                f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]

        # get proxy of last avg number of frames
        else:
            if proxy == 'mean':
                f = torch.mean(torch.stack([t['feats'] for t in it[-avg:]]), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack([t['feats'] for t in it[-avg:]]), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack([t['feats'] for t in it[-avg:]]), dim=0)[0]
        
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
    feats = torch.cat([t['feats'].cpu().unsqueeze(0) for k, v in tracks.items() for t in v if t['id'] != -1], 0)
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
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


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
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()


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


class Track():
    def __init__():
        pass