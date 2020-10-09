from torch.utils.data.sampler import Sampler
import random
import copy
import torch
import sklearn.metrics.pairwise
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger('GNNReID.CombineSampler')

class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl):
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.feature_dict = None
        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        # add elements till every class has the same num of obs
        # np.random.choice(idxs, size=self.num_instances, replace=True)
        #for inds in l_inds:
        #    choose = copy.deepcopy(inds)
        #    while len(inds) < self.max:
        #        inds += [random.choice(choose)]
        #print("NEW MAX")

        #for inds in l_inds:
        #    n_els = self.max - len(inds) + 1  # take out 1?
        #    inds.extend(inds[:n_els])  # max + 1
        #print("OLD MAX")

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:] 
            assert len(inds) == 0
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class TrainTestCombi(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl, l_inds_train=None, l_inds_gallery=None, backbone=False):
        self.l_inds = l_inds
        self.l_inds_train = l_inds_train
        self.l_inds_gallery = l_inds_gallery if l_inds_gallery is not None else l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.backbone = backbone

        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)
        self.get_train_inds()

    def get_train_inds(self):
        random.seed(0)
        np.random.seed(0)
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds_train))
        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(
                inds) // self.n_cl + 1) * self.n_cl - len(inds),
                                           replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]
            assert len(inds) == 0

        random.shuffle(split_list_of_indices)
        self.train_samples = [item for sublist in split_list_of_indices[:self.cl_b] for item in
                          sublist]
        [self.train_samples.pop(ind) for ind in random.sample(list(range(len(self.train_samples))), 2)]
        
    def __iter__(self):
        if self.backbone:
            self.flat_list = [samp for cl in self.l_inds for samp in cl]
            gallery_list = [samp for cl in self.l_inds_gallery for samp in cl]
            self.flat_list = self.flat_list + gallery_list + self.train_samples
        else:
            self.flat_list = list()
            for i in range(60):
                #for query_class in self.l_inds:
                query_class = random.choice(self.l_inds)
                query_idx = random.choice(query_class) 
                for gallery_class in self.l_inds_gallery:
                    for gallery_idx in random.choices(gallery_class, k=3): #gallery_class
                        batch = self.train_samples + [gallery_idx, query_idx]
                        self.flat_list.append(batch)

            logger.info("{} batches".format(len(self.flat_list)))
            self.flat_list = [item for batch in self.flat_list for item in
                              batch]

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class PseudoSampler(Sampler):
    def __init__(self, num_classes, num_samples):
        self.feature_dict = None
        self.bs = num_classes * num_samples

    def __iter__(self):
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        x = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
        y = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
        indices = list(self.feature_dict.keys())
        batches = list()
        for samp in sorted_dist:
            batch = [indices[i] for i in samp[:self.bs]]
            batches.append(batch)
        logger.info("Number batches {}".format(len(batches)))
        self.flat_list = [s for batch in batches for s in batch]
        logger.info(len(self.flat_list))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerII(Sampler):
    def __init__(self, num_classes, num_samples):
        logger.info("Pseudo sampler II")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples

    def __iter__(self):
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        x = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
        y = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
        indices = list(self.feature_dict.keys())
        indices_orig = copy.deepcopy(indices)
        batches = list()
        while indices:
            batch = list()
            i = 1
            while len(batch) < self.bs:
                if indices:
                    c = random.choice(indices)
                else:
                    c = random.choice(indices_orig)
                ind = indices_orig.index(c)
                #logger.info("class")
                #logger.info('{}, {}'.format(c, ind))
                j = 0
                while len(batch) < i * self.num_samples:
                    if not indices:
                        batch.append(indices_orig[sorted_dist[ind][j]])
                    elif indices_orig[sorted_dist[ind][j]] in indices:
                        #logger.info('{}, {}'.format(sorted_dist[ind][j], indices_orig[sorted_dist[ind][j]]))
                        batch.append(indices_orig[sorted_dist[ind][j]])
                        indices.remove(indices_orig[sorted_dist[ind][j]])
                    j += 1
                #indices.remove(c)
                i += 1
            batches.append(batch)
        logger.info("Number batches {}".format(len(batches)))
        self.flat_list = [s for batch in batches for s in batch]
        logger.info(len(self.flat_list))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerIII(Sampler):
    def __init__(self, num_classes, num_samples):
        logger.info("Pseudo sampler II")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.cl_b = num_classes
        self.n_cl = num_samples
        logger.info("Using number of classes as number of clusters")
        self.nb_clusters = num_classes
        logger.info("{} Clusters for Pseudo sampling".format(self.nb_clusters))
    def __iter__(self):
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
        cluster = sklearn.cluster.KMeans(self.nb_clusters).fit(x).labels_        
        indices = [k.item() for k in self.feature_dict.keys()]
        
        ddict = defaultdict(list)
        for idx, label in zip(indices, cluster):
            ddict[label].append(idx)

        l_inds = []
        for key in ddict:
            l_inds.append(ddict[key]) 
        
        l_inds = list(map(lambda a: random.sample(a, len(a)), l_inds))
        
        # add elements till every class has the same num of obs
        # np.random.choice(idxs, size=self.num_instances, replace=True)
        #for inds in l_inds:
        #    choose = copy.deepcopy(inds)
        #    while len(inds) < self.max:
        #        inds += [random.choice(choose)]
        #print("NEW MAX")

        #for inds in l_inds:
        #    n_els = self.max - len(inds) + 1  # take out 1?
        #    inds.extend(inds[:n_els])  # max + 1
        #print("OLD MAX")

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:] 
            assert len(inds) == 0
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        #print(len(split_list_of_indices), self.cl_b) 
        #print(split_list_of_indices)
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        print(len(set(self.flat_list))) 
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class DistanceSamplerOrig(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.strategy = strategy
        self.max = -1
        self.feature_dict = dict()
        self.epoch = 0
        for inds, samp in samples.items():
            if len(samp) > self.max:
                self.max = len(samp)
        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        self.feature_dict = {k: self.feature_dict[k] for k in sorted(self.feature_dict.keys())}
        dist_mat = np.zeros([len(self.feature_dict), len(self.feature_dict)])
        i = 0
        for ind1, feat_vect1 in self.feature_dict.items():
            j = 0
            for ind2, feat_vect2 in self.feature_dict.items():
                if i > j:
                    dist_mat[i, j] = dist_mat[j, i]
                    j += 1
                    continue
                x = torch.stack(feat_vect1, 0)
                y = torch.stack(feat_vect2, 0)
                m, n = x.size(0), y.size(0)
                x = x.view(m, -1)
                y = y.view(n, -1)
                # use x^TX + y^Ty - 2x^Ty
                dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,
                                                                       m).t()
                dist.addmm_(1, -2, x, y.t())
                dist_mat[i, j] = torch.sum(dist).data.item() / (m*n)
                j += 1
            i += 1
        self.inter_class_dist = dist_mat

    def __iter__(self):
        if (self.strategy == 'alternating' and self.epoch % 5 == 2) or (
                self.strategy == 'only' and self.epoch % 5 == 2) or (
                self.strategy == 'pre' and self.epoch % 5 == 2) or (
                self.strategy == 'pre_soft' and self.epoch % 5 == 2):
            print('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()
        # shuffle elements inside each class
        l_inds = {ind: random.sample(sam, len(sam)) for ind, sam in self.samples.items()}
        if self.strategy == 'pre_soft':
            self.epoch += 30

        for c, inds in l_inds.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.max:
                l_inds[c] += [random.choice(choose)]
        # get clostest classes for each class
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()
        for cl in range(indices.shape[0]):
            possible_classes = indices[cl, :].tolist()
            possible_classes.remove(possible_classes.index(cl))
            if self.strategy != 'pre':
                sample_margin = int(len(possible_classes) * (1-(self.epoch/100)))
                classes = np.random.randint(sample_margin, size=self.num_classes-1).tolist()
                cls = [possible_classes[i] for i in classes]
            else:
                cls = possible_classes[:self.num_classes -1]

            cls.append(cl)
            batch = [s for c in cls for s in random.sample(l_inds[c], self.num_samples)]
            batches.append(batch)

        random.shuffle(batches)
        if len(batches) % self.num_classes != 0:
            b = np.random.choice(np.arange(len(batches)), size=len(batches) % self.num_classes, replace=False).tolist()
            [batches.append(batches[m]) for m in b]


        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSampler(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy, m):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.strategy = strategy
        self.m = m
        self.max = -1
        self.feature_dict = dict()
        self.index_dict = dict()
        self.epoch = 0

        # add until num_samples
        for c, inds in self.samples.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.num_samples:
                self.samples[c] += [random.choice(choose)]

        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # sort dicts and generate indicator tensor
        self.feats_sorted = defaultdict(list)
        self.index_sorted = defaultdict(list)
        self.indicator = list()
        feats = list()
        for k in sorted(self.feature_dict.keys()):
            # print(len(self.feature_dict[k]), self.feature_dict[k].keys())
            for i, f in self.feature_dict[k].items():
                self.feats_sorted[k].append(f)
                self.index_sorted[k].append(i)
            while len(self.feats_sorted[k]) < self.num_samples:
                ind = random.sample(range(len(self.feats_sorted[k])), 1)
                self.feats_sorted[k].append(self.feats_sorted[k][ind[0]])
                self.index_sorted[k].append(self.index_sorted[k][ind[0]])

            assert len(self.index_sorted[k]) == len(self.feats_sorted[k])
            self.indicator.append([k] * len(self.index_sorted[k]))
            feats.append(self.feats_sorted[k])

        self.indicator = [i for l in self.indicator for i in l]
        feats = [f for c in feats for f in c]
        # self.feature_dict = {k: self.feature_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.index_dict = {k: self.index_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.indicator = torch.tensor([k for k, v in self.feature_dict.items() for i in range(len(v))])

        self.indicator = torch.tensor(self.indicator)

        # stack features and generate pairwise sample dist
        # for f in feats:
        #    print(f, type(f))
        feats = torch.stack(feats)
        m, n = feats.size(0), feats.size(0)
        x = feats.view(m, -1)
        y = feats.view(n, -1)
        # use x^TX + y^Ty - 2x^Ty
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        self.sample_dist = dist

        # generate pairwise class distance
        dist_mat = np.zeros([len(self.feats_sorted), len(self.feats_sorted)])
        for i in self.feats_sorted.keys():
            for j in self.feats_sorted.keys():
                if i > j:
                    dist_mat[i, j] = dist_mat[j, i]
                    j += 1
                    continue
                row = self.indicator == i
                col = self.indicator == j
                r_max, r_min = torch.max(row.nonzero()).item(), \
                               torch.min(row.nonzero()).item()
                c_max, c_min = torch.max(col.nonzero()).item(), \
                               torch.min(col.nonzero()).item()
                mat_ij = dist[r_min:r_max + 1, c_min:c_max + 1]
                dist_ij = torch.sum(mat_ij).data.item() / (
                            mat_ij.size(0) * mat_ij.size(0))
                dist_mat[i, j] = dist_ij
                dist_mat[j, i] = dist_ij
        self.inter_class_dist = dist_mat

    def __iter__(self):
        if (self.strategy == 'alternating' and self.epoch % 5 == 2) or (
                self.strategy == 'only' and self.epoch % 5 == 2) or (
                self.strategy == 'pre' and self.epoch % 5 == 2) or (
                self.strategy == 'pre_soft' and self.epoch % 5 == 2):
            logger.info('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()
        # sort class distances
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()

        # for each anchor class sample hardest classes and hardest features
        for cl in range(indices.shape[0]):
            possible_classes = indices[cl, :].tolist()
            possible_classes.remove(possible_classes.index(cl))

            # get classes
            if self.strategy != 'pre':
                sample_margin = max(
                    int(len(possible_classes) * (1 - (self.epoch / self.m))),
                    self.num_classes - 1)
                classes = np.random.randint(sample_margin,
                                            size=self.num_classes - 1).tolist()
                cls = [possible_classes[i] for i in classes]
            else:
                cls = possible_classes[:self.num_classes - 1]

            # randomly sample anchor class samples
            ind_cl = [s for s in
                      random.sample(range(len(self.index_sorted[cl])),
                                    self.num_samples)]
            batch = [[self.index_sorted[cl][i] for i in ind_cl]]
            for c in cls:
                # threshold for samples to be sampled
                thresh = self.inter_class_dist[cl, c]

                # extract pairwise sample distance of anchor and sampled class
                row = self.indicator == cl
                col = self.indicator == c
                r_max, r_min = torch.max(row.nonzero()).item(), \
                               torch.min(row.nonzero()).item()
                c_max, c_min = torch.max(col.nonzero()).item(), \
                               torch.min(col.nonzero()).item()
                mat_clc = self.sample_dist[r_min:r_max + 1, c_min:c_max + 1]

                # take only the rows corresponding to sampled anchor samples
                mat_clc = [mat_clc[i, :] for i in range(mat_clc.shape[0]) if
                           i in ind_cl]
                if len(mat_clc) == 0:
                    print(mat_clc, ind_cl, r_min, r_max, c_min, c_max)
                    print("Error: sub matrix can not have zero entries")
                    quit()

                mat_clc = torch.stack(mat_clc)

                # samples from sampled class only if distance > than thresh
                possible_samples = torch.unique(
                    (mat_clc > thresh).nonzero()[:, 1])
                if possible_samples.shape[0] >= self.num_samples:
                    ind = random.sample(range(possible_samples.shape[0]),
                                        self.num_samples)
                elif possible_samples.shape[0] == 0:
                    ind = random.sample(range(c_max - c_min + 1),
                                        self.num_samples)
                else:
                    # if not enough samples with distance > tresh --> random
                    other_samps = [i for i in range(c_max - c_min + 1) if
                                   i not in possible_samples]
                    ind = random.sample(other_samps, self.num_samples -
                                        possible_samples.shape[0])
                    [ind.append(possible_samples[i]) for i in
                     range(possible_samples.shape[0])]

                samps = [self.index_sorted[c][i] for i in ind]
                batch.append(samps)
            batch = [s for c in batch for s in c]
            batches.append(batch)
        random.shuffle(batches)
        if len(batches) % self.num_classes != 0:
            b = np.random.choice(np.arange(len(batches)), size=len(batches) % self.num_classes, replace=False).tolist()
            [batches.append(batches[m]) for m in b]

        self.flat_list = [s for batch in batches for s in batch]

        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSamplerEnsure(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy, m):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.strategy = strategy
        self.m = m
        self.max = -1
        self.feature_dict = dict()
        self.index_dict = dict()
        self.epoch = 0

        # add until num_samples
        for c, inds in self.samples.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.num_samples:
                self.samples[c] += [random.choice(choose)]

        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # sort dicts and generate indicator tensor
        self.feats_sorted = defaultdict(list)
        self.index_sorted = defaultdict(list)
        self.indicator = list()
        feats = list()
        for k in sorted(self.feature_dict.keys()):
            #print(len(self.feature_dict[k]), self.feature_dict[k].keys())
            for i, f in self.feature_dict[k].items():
                self.feats_sorted[k].append(f)
                self.index_sorted[k].append(i)
            while len(self.feats_sorted[k]) < self.num_samples:
                ind = random.sample(range(len(self.feats_sorted[k])), 1)
                self.feats_sorted[k].append(self.feats_sorted[k][ind[0]])
                self.index_sorted[k].append(self.index_sorted[k][ind[0]])

            assert len(self.index_sorted[k]) == len(self.feats_sorted[k])
            self.indicator.append([k]*len(self.index_sorted[k]))
            feats.append(self.feats_sorted[k])
        
        del self.feature_dict

        self.indicator = [i for l in self.indicator for i in l]
        feats = [f for c in feats for f in c]
        # self.feature_dict = {k: self.feature_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.index_dict = {k: self.index_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.indicator = torch.tensor([k for k, v in self.feature_dict.items() for i in range(len(v))])

        self.indicator = torch.tensor(self.indicator)

        # stack features and generate pairwise sample dist
        #for f in feats:
        #    print(f, type(f))
        feats = torch.stack(feats)
        m, n = feats.size(0), feats.size(0)
        x = feats.view(m, -1)
        y = feats.view(n, -1)
        # use x^TX + y^Ty - 2x^Ty
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        self.sample_dist = dist

        # generate pairwise class distance
        dist_mat = np.zeros([len(self.feats_sorted), len(self.feats_sorted)])
        for i in self.feats_sorted.keys():
            for j in self.feats_sorted.keys():
                if i > j:
                    dist_mat[i, j] = dist_mat[j, i]
                    j += 1
                    continue
                row = self.indicator == i
                col = self.indicator == j
                r_max, r_min = torch.max(row.nonzero()).item(), \
                               torch.min(row.nonzero()).item()
                c_max, c_min = torch.max(col.nonzero()).item(), \
                               torch.min(col.nonzero()).item()
                mat_ij = dist[r_min:r_max+1, c_min:c_max+1]
                dist_ij = torch.sum(mat_ij).data.item() / (mat_ij.size(0)*mat_ij.size(0))
                dist_mat[i, j] = dist_ij
                dist_mat[j, i] = dist_ij
        self.inter_class_dist = dist_mat

    def __iter__(self):

        if (self.strategy == 'alternating' and self.epoch % 5 == 2) or (
                self.strategy == 'only' and self.epoch % 5 == 2) or (
                        self.strategy == 'pre' and self.epoch % 5 == 2) or (
                                self.strategy == 'pre_soft' and self.epoch % 5 == 2):
            print('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()

        if self.strategy == 'pre_soft':
            self.epoch = self.epoch + 30
        # sort class distances
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()

        # for each anchor class sample hardest classes and hardest features
        anchor_samples = copy.deepcopy(self.index_sorted)
        for cl in range(indices.shape[0]):
            while len(anchor_samples[cl]) > 0:
                possible_classes = indices[cl, :].tolist()
                possible_classes.remove(possible_classes.index(cl))

                # get classes
                if self.strategy != 'pre':
                    sample_margin = max(int(len(possible_classes) * (1-(self.epoch/self.m))), self.num_classes - 1)
                    classes = np.random.randint(sample_margin, size=self.num_classes-1).tolist()
                    cls = [possible_classes[i] for i in classes]
                else:
                    cls = possible_classes[:self.num_classes -1]

                # randomly sample anchor class samples
                if len(anchor_samples[cl]) < self.num_samples:
                    ind_cl_1 = [self.index_sorted[cl].index(i) for i in anchor_samples[cl]]
                    ind_cl = random.sample(range(len(self.index_sorted[cl])),
                                           self.num_samples - len(anchor_samples[cl]))
                    ind_cl += ind_cl_1
                    anchor_samples[cl] = list()
                else:
                    samps = random.sample(anchor_samples[cl], self.num_samples)
                    ind_cl = [self.index_sorted[cl].index(i) for i in samps]
                    [anchor_samples[cl].remove(sam) for sam in samps]

                #ind_cl = [s for s in random.sample(range(len(anchor_samples[cl])), self.num_samples)]
                batch = [[self.index_sorted[cl][i] for i in ind_cl]]

                for c in cls:
                    # threshold for samples to be sampled
                    thresh = self.inter_class_dist[cl, c]

                    # extract pairwise sample distance of anchor and sampled class
                    row = self.indicator == cl
                    col = self.indicator == c
                    r_max, r_min = torch.max(row.nonzero()).item(), \
                                   torch.min(row.nonzero()).item()
                    c_max, c_min = torch.max(col.nonzero()).item(), \
                                   torch.min(col.nonzero()).item()
                    mat_clc = self.sample_dist[r_min:r_max + 1, c_min:c_max + 1]

                    # take only the rows corresponding to sampled anchor samples
                    mat_clc = [mat_clc[i, :] for i in range(mat_clc.shape[0]) if i in ind_cl]
                    if len(mat_clc) == 0:
                        print(mat_clc, ind_cl, r_min, r_max, c_min, c_max)
                        print("Error: sub matrix can not have zero entries")
                        quit()

                    mat_clc = torch.stack(mat_clc)

                    # samples from sampled class only if distance > than thresh
                    possible_samples = torch.unique((mat_clc > thresh).nonzero()[:, 1])
                    #print(possible_samples, ind_cl, r_min, r_max, c_min, c_max, mat_clc.shape)
                    if possible_samples.shape[0] >= self.num_samples:
                        ind = random.sample(range(possible_samples.shape[0]), self.num_samples)
                    elif possible_samples.shape[0] == 0:
                        ind = random.sample(range(c_max-c_min+1), self.num_samples)
                    else:
                        # if not enough samples with distance > tresh --> random
                        other_samps = [i for i in range(c_max-c_min+1) if i not in possible_samples]
                        ind = random.sample(other_samps, self.num_samples-possible_samples.shape[0])
                        [ind.append(possible_samples[i]) for i in range(possible_samples.shape[0])]

                    samps = [self.index_sorted[c][i] for i in ind]
                    batch.append(samps)
                batch = [s for c in batch for s in c]
                batches.append(batch)
        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        print('SAMPLES')
        print(len(set(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSamplerMean(Sampler):
    def __init__(self, num_classes, num_samples, samples):
        print("USING DIST MEAN")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.max = -1
        self.feature_dict = dict()
        self.epoch = 0

        for inds, samp in samples.items():
            if len(samp) > self.max:
                self.max = len(samp)
        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        features_mean = [torch.mean(torch.stack(self.feature_dict[k], 0), 0) for
                             k in sorted(self.feature_dict.keys())]
        x = y = torch.stack(features_mean, 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        # use x^TX + y^Ty - 2x^Ty
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,
                                                               m).t()
        dist.addmm_(1, -2, x, y.t())

        self.inter_class_dist = dist.cpu().data.numpy()

    def __iter__(self):
        if self.epoch % 5 == 2:
            print('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()

        # shuffle elements inside each class
        l_inds = {ind: random.sample(sam, len(sam)) for ind, sam in self.samples.items()}

        for c, inds in l_inds.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.max:
                l_inds[c] += [random.choice(choose)]
        # get clostest classes for each class
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()
        for cl in range(indices.shape[0]):
            possible_classes = indices[cl, :].tolist()
            possible_classes.remove(possible_classes.index(cl))
            cls = possible_classes[:self.num_classes - 1]
            cls.append(cl)
            batch = [s for c in cls for s in random.sample(l_inds[c], self.num_samples)]
            batches.append(batch)

        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PretraingSampler(Sampler):
    def __init__(self, samples):
        self.samples = samples
        self.flat_list = list()
        self.max = 0

        for inds in samples:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        samples = list(map(lambda a: random.sample(a, len(a)), self.samples))

        for samp in samples:
            choose = copy.deepcopy(samp)
            while len(samp) < self.max:
                samp += [random.choice(choose)]

        self.flat_list = [item for sublist in samples for item in sublist]
        random.shuffle(self.flat_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerAdvanced(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """
    def __init__(self, l_inds, num_classes, num_elements_class, dict_class_distances, iterations_for_epoch):
        self.l_inds = l_inds
        self.num_classes = num_classes
        self.num_elements_class = num_elements_class
        self.batch_size = self.num_classes * self.num_elements_class
        self.flat_list = []
        self.iterations_for_epoch = iterations_for_epoch
        self.dict_class_distances = dict_class_distances

    def __iter__(self):
        self.flat_list = []
        for ii in range(int(self.iterations_for_epoch)):
            temp_list = []

            # get the class
            pivot_class_index = random.randint(0, self.num_classes - 1)
            pivot_class = self.l_inds[pivot_class_index]

            # put the elements of the class in a temp list
            pivot_elements = random.sample(pivot_class, self.num_elements_class)
            temp_list.extend(pivot_elements)

            # get the k nearest neighbors of the class
            other_class_indices = self.dict_class_distances[pivot_class_index][:self.num_classes - 1]

            # for each neighbor, put the elements of it in a temp list
            for class_index in other_class_indices:
                other_class = self.l_inds[class_index]
                # toDO - try/except error if class has less than k elements, in which case get all of them
                other_elements = random.sample(other_class, self.num_elements_class)
                temp_list.extend(other_elements)

            # shuffle the temp list
            random.shuffle(temp_list)
            self.flat_list.extend(temp_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerSuperclass(Sampler):
   """
   l_inds (list of lists)
   cl_b (int): classes in a batch
   n_cl (int): num of obs per class inside the batch
   """
   def __init__(self, l_inds, num_classes, num_elements_class, dict_superclass, iterations_for_epoch):
       self.l_inds = l_inds
       self.num_classes = num_classes
       self.num_elements_class = num_elements_class
       self.flat_list = []
       self.iterations_for_epoch = iterations_for_epoch
       self.dict_superclass = dict_superclass
   def __iter__(self):
       self.flat_list = []
       for ii in range(int(self.iterations_for_epoch)):
           temp_list = []
           # randomly sample the superclass
           superclass = random.choice(list(self.dict_superclass.keys()))
           list_of_potential_classes = self.dict_superclass[superclass]
           # randomly sample k classes for the superclass
           classes = random.sample(list_of_potential_classes, self.num_classes)
           # get the n objects for each class
           for class_index in classes:
               # classes are '141742158611' etc instead of 1, 2, 3, ..., this should be fixed by finding a mapping between two types of names
               class_ = self.l_inds[class_index]
               # check if the number of elements is >= self.num_elements_class
               if len(class_) >= self.num_elements_class:
                   elements = random.sample(class_, self.num_elements_class)
               else:
                   elements = random.choices(class_, k=self.num_elements_class)
               temp_list.extend(elements)
           # shuffle the temp list
           random.shuffle(temp_list)
           self.flat_list.extend(temp_list)
       return iter(self.flat_list)
   def __len__(self):
       return len(self.flat_list)


class CombineSamplerSuperclass2(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, num_classes, num_elements_class, dict_superclass, iterations_for_epoch):
        self.l_inds = l_inds
        self.num_classes = num_classes
        self.num_elements_class = num_elements_class
        self.flat_list = []
        self.iterations_for_epoch = iterations_for_epoch
        self.dict_superclass = dict_superclass

    def __iter__(self):
        self.flat_list = []
        for ii in range(int(self.iterations_for_epoch)):
            temp_list = []

            # randomly sample the superclass
            superclass_1 = random.choice(list(self.dict_superclass.keys()))
            list_of_potential_classes_1 = self.dict_superclass[superclass_1]

            superclass_2 = superclass_1
            while superclass_2 == superclass_1:
                superclass_2 = random.choice(list(self.dict_superclass.keys()))
            list_of_potential_classes_2 = self.dict_superclass[superclass_2]

            # randomly sample k classes for the superclass
            classes = random.sample(list_of_potential_classes_1, self.num_classes // 2)
            classes_2 = random.sample(list_of_potential_classes_2, self.num_classes // 2)
            classes.extend(classes_2)

            # get the n objects for each class
            for class_index in classes:
                # classes are '141742158611' etc instead of 1, 2, 3, ..., this should be fixed by finding a mapping between two types of names
                class_ = self.l_inds[class_index]
                # check if the number of elements is >= self.num_elements_class
                if len(class_) >= self.num_elements_class:
                    elements = random.sample(class_, self.num_elements_class)
                else:
                    elements = random.choices(class_, k=self.num_elements_class)
                temp_list.extend(elements)

            # shuffle the temp list
            random.shuffle(temp_list)
            self.flat_list.extend(temp_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


if __name__ == '__main__':

    feature_dict = {0: [torch.tensor([0.3376, 0.4358, 1.0508, 0.5083, 0.4756, 0.2441]),
                       torch.tensor([0.3079, 0.9692, 0.4834, 0.3521, 0.7603, 0.5337])],
                1: [torch.tensor([0.8047, 0.4968, 0.3809, 0.8354, 0.5464, 0.1128]),
                      torch.tensor([0.4136, 0.4592, 0.8535, 0.4343, 0.8022, 0.5010])],
                2: [torch.tensor([0.4578, 1.0303, 0.5601, 0.3042, 0.0160, 0.1713])]}

    samples = {0: [0, 1], 1: [2, 4], 2: [3]}

    samp = DistanceSampler(num_classes=3, num_samples=5, samples=samples, strategy='alternating', m=100)
    samp.feature_dict = feature_dict
    samp.index_dict = samples
    samp.get_inter_class_distances()
