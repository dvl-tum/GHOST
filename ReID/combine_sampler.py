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
        
