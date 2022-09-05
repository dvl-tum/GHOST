import os
from typing import Sized
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile
#import imageio
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import copy
import logging
import random 

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
                 paths_gallery=None, rand_scales=False, sz_crop=None):
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
        self.rand_scales = rand_scales
        self.distractor_idx = None
        self.no_imgs = False
        
        print('Randomly scales images {}'.format(rand_scales))
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
            i = 0
            self.map = dict()
            for lab in sorted(set(self.labels)):
                if lab != -2:
                    self.map[lab] = i 
                    i += 1
            for i, y in enumerate(self.labels):
                if y == -2:
                    if self.distractor_idx is None:
                        self.distractor_idx = list()
                    self.distractor_idx.append(i)
                    self.ys.append(y)
                else:
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

        self.transform = self.get_transform(sz_crop=sz_crop)

    def get_transform(self, sz_crop):

        if self.trans == 'norm':
            trans = utils.make_transform(is_train=not self.eval_reid, sz_crop=sz_crop)
        elif self.trans == 'bot':
            trans = utils.make_transform_bot(is_train=not self.eval_reid, sz_crop=sz_crop)
        print(trans)
        return trans

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        if self.no_imgs:
            return self.ys[index], index, self.im_paths[index]
        im = pil_loader(self.im_paths[index])
        
        if not self.eval_reid and self.rand_scales:
            if random.randint(0, 1):
                r = random.randint(2, 4)
                #print(im.size)
                im = im.resize((int(im.size[0]/r), int(im.size[1]/r)))
                #print(r, im.size) 
        im = self.transform(im)

        if self.labels_train is not None:
            return im, self.ys[index], index, self.im_paths[index]
        if self.eval_reid:
            return im, self.ys[index], index, self.im_paths[index]
        return im, self.ys[index], index, self.im_paths[index]


class All(torch.utils.data.Dataset):
    def __init__(self, root, labels, paths, trans=None,
                 eval_reid=False, sz_crop=None):
        root = os.path.dirname(root)
        self.dirs = {'market': os.path.join('/storage/slurm/seidensc/datasets', 'Market-1501-v15.09.15'),
                     'cuhk03': os.path.join('/storage/slurm/seidensc/datasets', 'cuhk03', 'detected')}
        self.trans = trans
        self.eval_reid = eval_reid
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        self.ys = list()
        self.im_paths = list()
        self.distractor_idx = None

        # when cuhk03 and market should be used
        i = 0
        self.map = dict()
        self.ys = list()
        self.im_paths = list()
        for dataset in labels.keys():
            for lab in sorted(set(labels[dataset])):
                if lab != -2:
                    id = str(lab) + '_' + dataset
                    self.map[id] = i 
                    i += 1
            
            for j, (y, dat) in enumerate(zip(labels[dataset], paths[dataset])):
                id = str(y) + '_' + dataset
                if y == -2:
                    if self.distractor_idx is None:
                        self.distractor_idx = list()
                    self.distractor_idx.append(j)
                    self.ys.append(y)
                else:
                    self.ys.append(self.map[id])
                self.im_paths.append(os.path.join(self.dirs[dataset], 'images', '{:05d}'.format(
                    int(dat.split('_')[0])), dat))

        self.transform = self.get_transform(sz_crop=sz_crop)

    def get_transform(self, sz_crop):
        if self.trans == 'norm':
            trans = utils.make_transform(is_train=not self.eval_reid, sz_crop=sz_crop)
        elif self.trans == 'bot':
            trans = utils.make_transform_bot(is_train=not self.eval_reid, sz_crop=sz_crop)
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
        print(trans)
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
            return im, self.ys[index], index, self.im_paths[index]
        return im, self.ys[index], index, self.im_paths[index]


