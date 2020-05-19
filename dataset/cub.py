import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile


class Birds(torch.utils.data.Dataset):
    def __init__(self, root, labels, paths, transform=None,
                 eval_reid=False):
        if os.path.basename(root) == 'cuhk03' or os.path.basename(root) == 'cuhk03-np':
            roots = [os.path.join(root, 'labeled'), os.path.join(root, 'detected')]
        else:
            roots = [root]

        self.eval_reid = eval_reid
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        map = {lab: i for i, lab in enumerate(sorted(set(self.labels)))}
        self.ys = list()
        self.im_paths = list()
        for root in roots:
            for i, y in enumerate(self.labels):
                self.ys.append(map[y])
                self.im_paths.append(os.path.join(root, 'images', '{:05d}'.format(
                    int(paths[i].split('_')[0])), paths[i]))

        if transform: self.transform = transform

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        im = self.transform(im)
        if self.eval_reid:
            return im, self.ys[index], os.path.basename(self.im_paths[index])
        return im, self.ys[index]

