from __future__ import print_function, absolute_import
import torch
import numpy as np
from sklearn.metrics import average_precision_score
import os
from collections import defaultdict


def pairwise_distance(features, query=None, gallery=None, root=None):
    img_dir = os.path.join(root, 'images')
    query_paths = [i for id in os.listdir(img_dir) for i in
                   os.listdir(os.path.join(img_dir, id)) if int(id) in query]
    gallery_paths = [i for id in os.listdir(img_dir) for i in
                     os.listdir(os.path.join(img_dir, id)) if
                     int(id) in gallery]

    x = torch.cat([features[f].unsqueeze(0) for f in query_paths], 0)
    y = torch.cat([features[f].unsqueeze(0) for f in gallery_paths], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist, query_paths, gallery_paths


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=20,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0

    for i in range(m):
        # Filter out the same id and same camera
        pos = ((gallery_ids[indices[i]] != query_ids[i]) |
               (gallery_cams[indices[i]] != query_cams[i]))

        if separate_camera_set:
            # Filter out samples from same camera
            pos &= (gallery_cams[indices[i]] != query_cams[i])

        if not np.any(matches[i, pos]): continue

        if single_gallery_shot:
            # TODO: is repeat 20?!
            repeat = 10
            gids = gallery_ids[indices[i][pos]]
            inds = np.where(pos)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1

        for _ in range(repeat):
            if single_gallery_shot:  # cuhk03 old testing protocol
                # Randomly choose one instance for each id
                sampled = (pos & _unique_sample(ids_dict, len(pos)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, pos])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1

    if num_valid_queries == 0:
        raise RuntimeError("No valid query")

    return ret.cumsum() / num_valid_queries


def mean_ap_classic(dist, ql, qc, gl, gc):
    junk0 = gl == -1
    indices = np.argsort(dist, axis=1)
    ap_list = list()
    # cmc = np.zeros(indices.shape[1])
    for k in range(dist.shape[0]):
        # generate masks
        pos = np.argwhere((gl == ql[k]) & (gc != qc[k]))
        junk1 = gc == qc[k]
        junk = np.argwhere(junk0 | junk1)

        # if no pos samples, continue
        if pos.size == 0:
            continue

        # filter out junk and get good indices
        mask = np.in1d(indices[k], junk, invert=True)
        index = indices[k][mask]
        mask = np.in1d(index, pos, invert=False)
        row_good = np.argwhere(mask == True).flatten()

        # cmc_tmp = np.zeros(indices.shape[1])
        # cmc_tmp[row_good[0]:] = 1
        old_precision, ap = 1, 0
        for i in range(len(pos)):
            d_recall = 1 / (len(pos))
            precision = (i + 1) / (row_good[i] + 1)
            ap = ap + d_recall * (old_precision - precision)
            old_precision = precision

        ap_list.append(ap)
        # cmc = cmc + cmc_tmp
        # cmc/indices.shape[0]
    return sum(ap_list) / len(ap_list)


def mean_ap_sklearn(dist, ql, qc, gl, gc):
    # TODO: same camera out? junk -1 out?

    indices = np.argsort(dist, axis=1)
    matches = (gl[indices] == ql[:, np.newaxis])
    aps = []
    for k in range(dist.shape[0]):
        # Filter out the same id and same camera
        pos = (gl[indices[k]] != ql[k]) | (gc[indices[k]] != qc[k])
        y_true = matches[k, pos]
        y_score = -dist[k][indices[k]][pos]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def evaluate_all(distmat, query=None, gallery=None):
    query_ids = np.asarray(
        [int(os.path.basename(path).split('_')[0]) for path in query])
    gallery_ids = np.asarray(
        [int(os.path.basename(path).split('_')[0]) for path in
         gallery])
    query_cams = np.asarray(
        [int(os.path.basename(path).split('_')[1]) for path in query])
    gallery_cams = np.asarray(
        [int(os.path.basename(path).split('_')[1]) for path in
         gallery])

    # Compute mean AP
    mAP = mean_ap_sklearn(distmat, query_ids, query_cams, gallery_ids,
                          gallery_cams)

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'Market': dict(separate_camera_set=False,
                       single_gallery_shot=False,
                       first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores, mAP


def calc_mean_average_precision(features, query, gallery, rootdir):
    distmat, query_paths, gallery_paths = pairwise_distance(features, query,
                                                            gallery, rootdir)
    return evaluate_all(distmat, query=query_paths, gallery=gallery_paths)


if __name__ == '__main__':
    features = {'00000084_00_0000.jpg': torch.tensor([1, 2, 3, 4, 5]),
                '00000129_01_0000.jpg': torch.tensor([2, 3, 4, 5, 6]),
                '00000129_03_0000.jpg': torch.tensor([3, 4, 5, 6, 7]),
                '00000084_04_0000.jpg': torch.tensor([1, 2, 3, 4, 5]),
                '00001451_01_0000.jpg': torch.tensor([6, 6, 6, 6, 6]),
                '00001451_02_0000.jpg': torch.tensor([1, 2, 3, 4, 5]),
                }
    query = [84]
    gallery = [84, 129, 1451]

    np.random.seed(50)
    features = {'00000169_00_0000.jpg': torch.rand(5),
                '00000169_01_0001.jpg': torch.rand(5),
                '00000169_01_0000.jpg': torch.rand(5),
                '00000864_02_0000.jpg': torch.rand(5),
                '00000864_01_0000.jpg': torch.rand(5),
                '00000864_02_0001.jpg': torch.rand(5),
                '00000862_00_0000.jpg': torch.rand(5),
                '00000862_02_0001.jpg': torch.rand(5),
                '00000862_02_0000.jpg': torch.rand(5),
                '00000151_00_0000.jpg': torch.rand(5),
                '00000151_03_0000.jpg': torch.rand(5),
                '00000151_00_0001.jpg': torch.rand(5),
                '00001237_01_0000.jpg': torch.rand(5),
                '00001237_02_0000.jpg': torch.rand(5),
                '00001237_00_0000.jpg': torch.rand(5),
                '00000098_03_0000.jpg': torch.rand(5),
                '00000098_00_0000.jpg': torch.rand(5),
                '00000098_03_0001.jpg': torch.rand(5),
                '00000095_03_0000.jpg': torch.rand(5),
                '00000095_03_0001.jpg': torch.rand(5),
                '00000095_01_0000.jpg': torch.rand(5),
                '00000389_02_0000.jpg': torch.rand(5),
                '00000389_02_0001.jpg': torch.rand(5),
                '00000389_01_0000.jpg': torch.rand(5),
                '00000323_00_0000.jpg': torch.rand(5),
                '00000323_04_0000.jpg': torch.rand(5),
                '00000323_00_0001.jpg': torch.rand(5),
                '00000143_01_0000.jpg': torch.rand(5),
                '00000143_03_0000.jpg': torch.rand(5),
                '00000143_03_0001.jpg': torch.rand(5),
                '00000059_00_0000.jpg': torch.rand(5),
                '00000059_03_0000.jpg': torch.rand(5),
                '00000059_01_0000.jpg': torch.rand(5),
                '00000857_01_0000.jpg': torch.rand(5),
                '00000857_02_0001.jpg': torch.rand(5),
                '00000857_02_0000.jpg': torch.rand(5)}

    query = [857, 59, 862]
    gallery = [857, 59, 143, 323, 389, 98, 95, 1237, 151, 862, 864, 169]

    rootdir = '../../../datasets/Market'

    """for dir in os.listdir(os.path.join(rootdir, 'images')):
        images = list()
        person = os.path.join(rootdir, 'images', dir)
        for img in os.listdir(person):
            images.append(os.path.join(dir, img))
        if len(images) == 3:
            print(images)
    quit()"""

    calc_mean_average_precision(features, query, gallery, rootdir)
