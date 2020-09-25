from __future__ import print_function, absolute_import
import torch
import numpy as np
from sklearn.metrics import average_precision_score
import os
from collections import defaultdict


def dist_traintest(features, query=None, gallery=None):
    dist = torch.zeros(len(query), len(gallery))
    for i, qu in enumerate(query):
        for j, gal in enumerate(gallery):
            #dist[i, j] = features[qu][gal][qu] @ features[qu][gal][qu] + \
            #                features[qu][gal][gal] @ features[qu][gal][gal] - \
            #                2 * (features[qu][gal][gal] @ features[qu][gal][qu])
            if gal in features[qu].keys():
                dist[i, j] = features[qu][gal]
            else:
                dist[i, j] = -1000

    return dist


def pairwise_distance(features, query=None, gallery=None):

    x = torch.cat([features[f].unsqueeze(0) for f in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    dist.addmm_(1, -2, x, y.t())
    return dist


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
    #junk = gallery_ids != -1
    if type(distmat) != np.ndarray:
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
        pos2 = (distmat[i][indices[i]] != -1000) # because of train test
        pos &= pos2
        # filter out samples of class -1 (distractors)
        junk = gallery_ids[indices[i]] != -1
        pos &= junk

        if separate_camera_set:
            # Filter out samples from same camera
            pos &= (gallery_cams[indices[i]] != query_cams[i])

        if not np.any(matches[i, pos]): continue

        if single_gallery_shot:
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


def mean_ap(dist, ql, qc, gl, gc):
    # TODO: same camera out?
    # junk = gl != -1
    if type(dist) != np.ndarray:
        dist = dist.cpu().numpy()

    indices = np.argsort(dist, axis=1)
    matches = (gl[indices] == ql[:, np.newaxis])

    aps = []
    for k in range(dist.shape[0]):
        # Filter out the same id and same camera
        pos = (gl[indices[k]] != ql[k]) | (gc[indices[k]] != qc[k])
        pos2 = (dist[k][indices[k]] != -1000) # because of train test
        pos &= pos2
        # filter out samples of class -1 (distractors)
        junk = gl[indices[k]] != -1
        pos &= junk

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
    mAP = mean_ap(distmat, query_ids, query_cams, gallery_ids, gallery_cams)

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
    return mAP, cmc_scores

"""
Created on Fri, 25 May 2018 20:29:09
@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""


def re_ranking(features, query, gallery, k1=20, k2=6, lambda_value=0.3,
               local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    probFea = torch.cat([features[f].unsqueeze(0) for f in query], 0)
    galFea = torch.cat([features[f].unsqueeze(0) for f in gallery], 0)
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)

    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        # normalize each row
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        # jaccard_in = size_in / (size_s1 + size_s2 - size_in)
        # --> normalized size_1=1 and size_2=1 --> 1+1=2
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def calc_mean_average_precision(features, query, gallery, re_rank=False, lamb=0.3, k1=20, k2=6):
    if type(features[list(features.keys())[0]]) == dict:
        query = list(features.keys())
        gallery = set([k2 for k1 in features.keys() for k2 in features[k1].keys()])
        distmat = dist_traintest(features, query, gallery)
    elif re_rank:
        distmat = re_ranking(features, query, gallery, k1=k1, k2=k2, lambda_value=lamb)
    else:
        distmat = pairwise_distance(features, query, gallery)
    return evaluate_all(distmat, query=query, gallery=gallery)


if __name__ == '__main__':
    features = {'00000084_00_0000.jpg': torch.tensor([1, 2, 3, 4, 5]),
                '00000129_01_0000.jpg': torch.tensor([2, 3, 4, 5, 6]),
                '00000129_03_0000.jpg': torch.tensor([3, 4, 5, 6, 7]),
                '00000084_04_0000.jpg': torch.tensor([1, 2, 3, 4, 5]),
                '00001451_01_0000.jpg': torch.tensor([6, 6, 6, 6, 6]),
                '00001451_02_0000.jpg': torch.tensor([1, 2, 3, 4, 5]),
                }
    query = ['00000084_00_0000.jpg', '00000129_01_0000.jpg', '00001451_01_0000.jpg', '00000084_04_0000.jpg', '00000129_03_0000.jpg', '00001451_02_0000.jpg']
    gallery = ['00000084_04_0000.jpg', '00000129_03_0000.jpg', '00001451_02_0000.jpg', '00000084_00_0000.jpg', '00000129_01_0000.jpg', '00001451_01_0000.jpg']
    '''
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
    '''
    rootdir = '../../../datasets/Market'

    """for dir in os.listdir(os.path.join(rootdir, 'images')):
        images = list()
        person = os.path.join(rootdir, 'images', dir)
        for img in os.listdir(person):
            images.append(os.path.join(dir, img))
        if len(images) == 3:
            print(images)
    quit()"""

    calc_mean_average_precision(features, query, gallery, re_rank=True)
