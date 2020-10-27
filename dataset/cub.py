import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile
import imageio
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import copy
import logging

logger = logging.getLogger('GNNReID.Dataset')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

def show_dataset(img, y):
    im = np.array(img)
    if im.shape[2] != 3:
        im = im.transpose((1, 2, 0))
    plt.imshow(im)
    plt.axis('off')
    plt.title('Image of label {}'.format(y))
    plt.show()

class Birds(torch.utils.data.Dataset):
    def __init__(self, root, labels, paths, trans=None,
                 eval_reid=False, magnitude=15, number_aug=0,
                 labels_train=None, paths_train=None, labels_gallery=None,
                 paths_gallery=None):
        self.trans = trans
        self.magnitude = magnitude
        self.number_aug = number_aug
        self.eval_reid = eval_reid
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        self.ys = list()
        self.im_paths = list()
        self.labels_train = labels_train
        self.paths_train = paths_train
        self.labels_gallery = labels_gallery
        self.paths_gallery = paths_gallery

        # when cuhk03 detected and labeled should be used
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
        # when only detected or labeled
        else:
            self.map = {lab: i for i, lab in enumerate(sorted(set(self.labels)))}
            for i, y in enumerate(self.labels):
                self.ys.append(self.map[y])
                self.im_paths.append(os.path.join(root, 'images', '{:05d}'.format(
                    int(paths[i].split('_')[0])), paths[i]))

        if self.labels_train is not None:
            self.ys_query = copy.deepcopy(self.ys) 
            self.ys_train = list()
            max_query = max(list(self.map.values()))
            map_train = {lab: i + 1 + max_query for i, lab in
                        enumerate(sorted(set(self.labels_train)))}
            self.map.update(map_train)
            for i, y in enumerate(self.labels_train):
                self.ys.append(self.map[y])
                self.ys_train.append(self.map[y])
                self.im_paths.append(
                    os.path.join(root, 'images', '{:05d}'.format(
                        int(paths_train[i].split('_')[0])), paths_train[i]))

        if self.labels_gallery is not None:
            self.ys_gallery = list()
            for i, y in enumerate(self.labels_gallery):
                if y not in self.map.keys():
                    self.map[y] = max(list(self.map.values())) + 1
                self.ys.append(self.map[y])
                self.ys_gallery.append(self.map[y])
                self.im_paths.append(
                    os.path.join(root, 'images', '{:05d}'.format(
                        int(paths_gallery[i].split('_')[0])), paths_gallery[i]))

        self.transform = self.get_transform()

    def get_transform(self):
        print(self.trans)
        if self.trans == 'norm':
            trans = utils.make_transform(is_train=not self.eval_reid)
        elif self.trans == 'bot':
            trans = utils.make_transform_bot(is_train=not self.eval_reid)
        elif self.trans == 'imgaug':
            trans = utils.make_transform_imaug(is_train=not self.eval_reid)
        elif self.trans == 'randaug':
            trans = utils.make_rand_aug(is_train=not self.eval_reid, M=self.magnitude, N=self.number_aug)
        elif self.trans == 'GLorig':
            trans = utils.make_transform_GL_orig(is_train=not self.eval_reid)
        elif self.trans == 'appearance':
            ddict = defaultdict(list)
            for idx, label in enumerate(self.ys):
                ddict[label].append(idx)
            self.occurance = {k: len(v) for k, v in ddict.items()}
            num_im = set(self.occurance.values())
            ps = [i / max(num_im) for i in num_im]
            trans = dict()
            for p, n in zip(ps, num_im):
                trans[n] = utils.appearance_proportional_augmentation1(is_train=not self.eval_reid, app=p)

        return trans

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        #show_dataset(im, self.ys[index])
        if self.trans == 'appearance':
            im = self.transform[self.occurance[self.ys[index]]](im)
        else:
            im = self.transform(im)
        #show_dataset(im, self.ys[index])
        if self.labels_train is not None:
            return im, self.ys[index], index, self.im_paths[index]
        if self.eval_reid:
            return im, self.ys[index], index, self.im_paths[index]
        return im, self.ys[index], index, self.im_paths[index]


class All(torch.utils.data.Dataset):
    def __init__(self, root, labels, paths, trans=None,
                 eval_reid=False):
        root = os.path.dirname(root)
        self.dirs = {'market': os.path.join(os.path.dirname(root), 'Market-1501-v15.09.15'),
                     'cuhk03': os.path.join(os.path.dirname(root), 'cuhk03', 'detected')}
        self.trans = trans
        self.eval_reid = eval_reid
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        self.ys = list()
        self.im_paths = list()

        # when cuhk03 and market should be used
        i = 0
        self.map = dict()
        self.ys = list()
        self.im_paths = list()
        for dataset in labels.keys():
            for lab, dat in zip(labels[dataset], paths[dataset]):
                id = str(lab) + '_' + dataset
                if id not in self.map.keys():
                    self.map[id] = i
                    i += 1
                self.ys.append(self.map[id])
                self.im_paths.append(os.path.join(self.dirs[dataset], 'images', '{:05d}'.format(
                int(dat.split('_')[0])), dat))

        self.transform = self.get_transform()

    def get_transform(self):
        if self.trans == 'norm':
            trans = utils.make_transform(is_train=not self.eval_reid)
        elif self.trans == 'bot':
            trans = utils.make_transform_bot(is_train=not self.eval_reid)
        elif self.trans == 'imgaug':
            trans = utils.make_transform_imaug(is_train=not self.eval_reid)
        elif self.trans == 'appearance':
            ddict = defaultdict(list)
            for idx, label in enumerate(self.ys):
                ddict[label].append(idx)
            self.occurance = {k: len(v) for k, v in ddict.items()}
            num_im = set(self.occurance.values())
            ps = [i / max(num_im) for i in num_im]
            trans = dict()
            for p, n in zip(ps, num_im):
                trans[n] = utils.appearance_proportional_augmentation1(is_train=not self.eval_reid, app=p)

        return trans

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        #show_dataset(im, self.ys[index])
        if self.trans == 'appearance':
            im = self.transform[self.occurance[self.ys[index]]](im)
        else:
            im = self.transform(im)
        #show_dataset(im, self.ys[index])

        if self.eval_reid:
            return im, self.ys[index], self.im_paths[index]
        return im, self.ys[index], index


import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile


class Birds_DML(torch.utils.data.Dataset):
    def __init__(self, root, labels, transform=None,
                 eval_reid=False, net_type='resnet50', labels_train=None):
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        self.eval_reid = eval_reid
        self.trans = transform
        self.labels_train = labels_train
        self.net_type = net_type
        self.ys, self.im_paths = [], []
        if os.path.basename(root) == 'In_shop':
            self.map = dict()
            j = 0

        for i in torchvision.datasets.ImageFolder(
                root=os.path.join(root, 'images')
        ).imgs:
            # i[1]: label, i[0]: path to file, including root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.labels and fn[:2] != '._':
                if os.path.basename(root) == 'In_shop':
                    if y not in self.map.keys():
                        self.map[y] = j
                        j += 1
                    y = self.map[y]
                self.ys += [y]
                self.im_paths.append(i[0])

        if labels_train is not None:
            self.ys_test = copy.deepcopy(self.ys)
            self.ys_train = list()
            for i in torchvision.datasets.ImageFolder(
                    root=os.path.join(root, 'images')
            ).imgs:
                # i[1]: label, i[0]: path to file, including root
                y = i[1]
                # fn needed for removing non-images starting with `._`
                fn = os.path.split(i[0])[1]
                if y in self.labels_train and fn[:2] != '._':
                    self.ys += [y]
                    self.ys_train += [y]
                    self.im_paths.append(i[0])

        self.transform = self.get_transform()

    def get_transform(self):
        if self.trans == 'norm':
            trans = utils.make_transform(is_train=not self.eval_reid, net_type=self.net_type)
        elif self.trans == 'bot':
            trans = utils.make_transform_bot(is_train=not self.eval_reid, net_type=self.net_type)
        elif self.trans == 'imgaug':
            trans = utils.make_transform_imaug(is_train=not self.eval_reid, net_type=self.net_type)
        elif self.trans == 'GLorig':
            trans = utils.make_transform_GL_orig(is_train=not self.eval_reid, net_type=self.net_type)
        elif self.trans == 'GL_orig_RE':
            trans = utils.GL_orig_RE(is_train=not self.eval_reid, net_type=self.net_type)
        elif self.trans == 'appearance':
            ddict = defaultdict(list)
            for idx, label in enumerate(self.ys):
                ddict[label].append(idx)
            self.occurance = {k: len(v) for k, v in ddict.items()}
            num_im = set(self.occurance.values())
            ps = [i / max(num_im) for i in num_im]
            trans = dict()
            for p, n in zip(ps, num_im):
                trans[n] = utils.appearance_proportional_augmentation1(
                    is_train=not self.eval_reid, app=p)

        return trans

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        #show_dataset(im, self.ys[index])
        if self.trans == 'appearance':
            im = self.transform[self.occurance[self.ys[index]]](im)
        else:
            im = self.transform(im)
        #show_dataset(im, self.ys[index])
        if self.labels_train is not None:
            return im, self.ys[index], index, self.im_paths[index]
        if self.eval_reid:
            return im, self.ys[index], index, self.im_paths[index]
        return im, self.ys[index], index, self.im_paths[index]
