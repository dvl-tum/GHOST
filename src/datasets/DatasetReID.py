import copy
import random
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os.path as osp
import csv
from collections import defaultdict
import numpy as np
from ReID.dataset.utils import make_transform_bot
import PIL.Image as Image
from torchvision.transforms import ToTensor
from torchvision.ops.boxes import clip_boxes_to_image, nms
from torch.utils.data.sampler import Sampler


def collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    visibility = [item[2] for item in batch]
    im_path = [item[3] for item in batch]
    dets = [item[4] for item in batch]

    return [data, target, visibility, im_path, dets]


class ClassBalancedSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, num_classes_iter, num_elements_class, batch_sampler=None):
        print("Class Balanced Sampling")
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = num_classes_iter
        self.n_cl = num_elements_class
        self.batch_size = self.cl_b * self.n_cl
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

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]

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
    def __init__(self, num_classes, num_samples, seed=None):
        self.bs = num_classes * num_samples
        self.possible_denominators = [i for i in range(2, int(self.bs/2+1)) if self.bs%i == 0]
        seed = random.randint(0, 100) if seed is None else seed
        #seed = 4
        random.seed(seed)
        print("Using seed {}".format(seed))

    def sample(self):
        num_classes = random.choice(self.possible_denominators)
        num_samples = int(self.bs/num_classes)
        print("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples


class BatchSizeSampler():
    def __init__():
        seed = random.randint(0, 100)
        random.seed(seed)
        print("Using seed {}".format(seed))
    def sample(self):
        num_classes = random.choice(range(2, 20))
        num_samples = random.choice(range(2, 20))
        print("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples
