import torch
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
import copy


def add_ioa(tracks, seq, interaction, occlusion):
    bbs = torch.from_numpy(np.vstack([tr['bbox'] for tr in tracks]))
    inter, area, _ = _box_inter_area(bbs, bbs)
    ioa = inter / np.atleast_2d(area).T
    ioa = ioa - np.eye(ioa.shape[0])

    # not taking foot position into account --> ioa_nofoot
    curr_interaction = copy.deepcopy(ioa)
    ioa_nf = np.sum(ioa, axis=1).tolist()
    interaction[seq] += ioa_nf

    for i, t in zip(ioa_nf, tracks):
        t['ioa_nf'] = i

    # taking foot position into account
    bot = np.atleast_2d(bbs[:, 3]).T < np.atleast_2d(bbs[:, 3])
    ioa[~bot] = 0
    curr_occlusion = ioa
    ioa = np.sum(ioa, axis=1).tolist()
    occlusion[seq] += ioa

    for i, t in zip(ioa, tracks):
        t['ioa'] = i

    return curr_interaction, curr_occlusion


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

    return inter.numpy(), area1.numpy(), area2.numpy()

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

