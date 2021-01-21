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

    def __init__(self, l_inds, cl_b, n_cl, batch_sampler=None):
        logger.info("Combine Sampler")
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

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(cl_b, n_cl)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()

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


class NumberSampler():
    def __init__(self, num_classes, num_samples):
        self.bs = num_classes * num_samples
        self.possible_denominators = [i for i in range(2, int(self.bs/2+1)) if self.bs%i == 0]

    def sample(self):
        num_classes = random.choice(self.possible_denominators)
        num_samples = int(self.bs/num_classes)
        logger.info("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples


class BatchSizeSampler():
    def sample(self):
        num_classes = random.choice(range(2, 20))
        num_samples = random.choice(range(2, 20))
        logger.info("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples


class KReciprocalSampler(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        logger.info("Pseudo sampler - kNN reciprocal")
        self.feature_dict = None
        self.bs = 7 #num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.k1 = 30
        self.k2 = self.bs
        print(self.bs, self.k1, self.k2)
        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.bs = self.num_classes * self.num_samples
            quality_checker.num_samples = self.bs
        
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            y =  torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            indices = [k for k in self.feature_dict.keys()]

        # generate distance mat for all classes as in Hierachrical Triplet Loss
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        dist = dist.cpu().numpy()
        sorted_dist = np.argsort(dist, axis=1)
        batches = list()
        exp = 0
        no = 0
        for i in range(sorted_dist.shape[0]):
            e = 0
            # k-reciprocal neighbors
            forward = sorted_dist[i, :self.k1 + 1]
            backward = sorted_dist[forward, :self.k1 + 1]
            rr = np.where(backward == i)[0]
            reciprocal = forward[rr]
            reciprocal_expansion = reciprocal
            for cand in reciprocal:
                cand_forward = sorted_dist[cand, :int(np.around(self.k1 / 2)) + 1]
                cand_backward = sorted_dist[cand_forward, :int(np.around(self.k1 / 2)) + 1]
                fi_cand = np.where(cand_backward == cand)[0]
                cand_reciprocal = cand_forward[fi_cand]
                if len(np.intersect1d(cand_reciprocal, reciprocal)) > 2 / 3 * len(
                        cand_reciprocal):
                    reciprocal_expansion = np.append(reciprocal_expansion, cand_reciprocal)
                    e =1
            if e == 1:
                exp +=1
            else: 
                no +=1
            reciprocal_expansion = np.unique(reciprocal_expansion)
            batch = reciprocal_expansion[np.argsort(dist[i, reciprocal_expansion])[:self.bs]].tolist()
            k = 0
            while len(batch) < self.bs:
                if sorted_dist[i, k] not in batch:
                    batch.append(sorted_dist[i, k])
                k += 1
            batch = [indices[k] for k in batch]
            assert len(batch) == self.bs
            batches.append(batch)
        logger.info("Number batches {}".format(len(batches)))

        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        logger.info("{} Number of samples to process".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class ClusteringSampler(Sampler):
    def __init__(self, num_classes, num_samples, nb_clusters=None, batch_sampler=None):
        # kmeans
        logger.info("Pseudo sampler III - kmeans")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.cl_b = num_classes
        self.n_cl = num_samples
        self.epoch = 0
        self.nb_clusters = nb_clusters

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def get_clusters(self):
        logger.info(self.nb_clusters)
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            self.indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            self.indices = [k for k in self.feature_dict.keys()]
        #logger.info("Kmeans")
        #self.cluster = sklearn.cluster.KMeans(self.nb_clusters).fit(x).labels_        
        #logger.info('spectral')
        #self.cluster = sklearn.cluster.SpectralClustering(self.nb_clusters, assign_labels="discretize", random_state=0).fit(x).labels_
        #self.nb_clusters = 600
        #logger.info('ward')
        #self.cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=self.nb_clusters).fit(x).labels_
        #logger.info('DBSCAN')
        #eps = 0.9
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x).labels_
        logger.info("Optics")
        eps = 0.7
        min_samples = 5
        logger.info("Eps {}, min samples {}".format(eps, min_samples))
        self.cluster = sklearn.cluster.OPTICS(min_samples=min_samples, eps=eps).fit(x).labels_
        #logger.info("Birch")
        #self.cluster = sklearn.cluster.Birch(n_clusters=self.nb_clusters).fit(x).labels_

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()
            quality_checker.num_samps=self.n_cl
        if self.epoch % 5 == 2:
            self.get_clusters()
        
        ddict = defaultdict(list)
        for idx, label in zip(self.indices, self.cluster):
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
        logger.info("{} blocks".format(len(split_list_of_indices)))
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            logger.info("Add {} blocks".format(self.cl_b - len(split_list_of_indices) % self.cl_b))
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        assert len(split_list_of_indices) % self.cl_b == 0

        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        logger.info("{} samples to process".format(len(set(self.flat_list)))) 
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class DistanceSampler(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy, m, batch_sampler=None):
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
        print("In distance constructor")
        print(len(samples))
        # add until num_samples
        for c, inds in self.samples.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.num_samples:
                self.samples[c] += [random.choice(choose)]

        self.inter_class_dist = np.ones([len(samples), len(samples)])
        
        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

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

        self.indicator = torch.tensor(self.indicator)

        # stack features and generate pairwise sample dist
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
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
        self.get_inter_class_distances()

        #if self.epoch % 5 == 2:
        #    logger.info('recompute dist at epoch {}'.format(self.epoch))
        #    self.get_inter_class_distances()
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
                classes = np.random.choice(sample_margin,
                                            size=self.num_classes - 1, replace=False).tolist()
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

