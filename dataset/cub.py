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
        self.eval_reid = eval_reid
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        self.ys = list()
        self.im_paths = list()

        if os.path.basename(root) == 'cuhk03' or os.path.basename(root) == 'cuhk03-np':
            typ = ['labeled', 'detected']
            assert len(set(labels[typ[0]]).difference(set(labels[typ[1]]))) == 0

            self.map = {lab: i for i, lab in enumerate(sorted(set(self.labels[typ[0]])))}
            self.ys = list()
            self.im_paths = list()
            for t in typ:
                for i, y in enumerate(self.labels[t]):
                    self.ys.append(self.map[y])
                    self.im_paths.append(os.path.join(root, t, 'images', '{:05d}'.format(
                    int(paths[t][i].split('_')[0])), paths[t][i]))

        else:
            self.map = {lab: i for i, lab in enumerate(sorted(set(self.labels)))}
            for i, y in enumerate(self.labels):
                self.ys.append(self.map[y])
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

            return im, self.ys[index], self.im_paths[index]
        return im, self.ys[index]

