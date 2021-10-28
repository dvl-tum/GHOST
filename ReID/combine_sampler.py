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

    def __init__(self, l_inds, cl_b, n_cl, batch_sampler=None, distractor_idx=None):
        logger.info("Combine Sampler")
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.feature_dict = None
        self.distractor_idx = distractor_idx
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

        if self.distractor_idx is not None:
            split_list_of_distractor_indices = []
            distractor_idx = copy.deepcopy(self.distractor_idx)
            random.shuffle(distractor_idx)
            while len(distractor_idx) >= self.n_cl:
                split_list_of_distractor_indices.append(distractor_idx[:self.n_cl])
                distractor_idx = distractor_idx[self.n_cl:]

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
            b = np.random.choice(np.arange(len(split_list_of_indices)), \
                size=self.cl_b - len(split_list_of_indices) % self.cl_b, \
                    replace=False).tolist()

            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]

        if self.distractor_idx is not None:
            split_list_of_indices = [s + split_list_of_distractor_indices[i] \
                for i, s in enumerate(split_list_of_indices)]
        
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


class KNNSampler(Sampler):
    def __init__(self, nn=50):
        # kNN
        logger.info("KNN")
        self.nn = nn
        self.dist = None
        self.indices_query = None
        self.indices_gallery = None

    def __iter__(self):

        sorted_dist = np.argsort(self.dist, axis=1)
        batches = list()
        for i in range(sorted_dist.shape[0]):  
            batch = [self.indices_query[i]] + [self.indices_gallery[k] for k in sorted_dist[i, :self.nn-1]]         
            batches.append(batch)
        logger.info("Number batches {}".format(len(batches)))

        self.flat_list = [s for batch in batches for s in batch]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class QueryGuidedSampler(Sampler):
    def __init__(self, batch_size):
        self.indices_gallery = None
        self.indices_query = None
        self.current_query_index = None
        self.batch_size = batch_size

    def __iter__(self):
        batches = list()
        import copy
        gallery_inds = copy.deepcopy(self.indices_gallery)
        while len(gallery_inds):
            batch = [self.indices_query[self.current_query_index]] + gallery_inds[:self.batch_size-1]
            gallery_inds = gallery_inds[self.batch_size-1:]
            batches.append(batch)

        self.flat_list = [s for batch in batches for s in batch]
        return iter(self.flat_list)
        

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

