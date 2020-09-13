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
            self.ys_train = list()
            self.im_paths_train = list()
            self.map_train = {lab: i for i, lab in
                        enumerate(sorted(set(self.labels_train)))}
            for i, y in enumerate(self.labels_train):
                self.ys_train.append(self.map_train[y])
                self.im_paths_train.append(
                    os.path.join(root, 'images', '{:05d}'.format(
                        int(paths_train[i].split('_')[0])), paths_train[i]))

        if self.labels_gallery is not None:
            self.ys_gallery = list()
            self.im_paths_gallery = list()
            for i, y in enumerate(self.labels_gallery):
                self.ys_gallery.append(self.map[y])
                self.im_paths_gallery.append(
                    os.path.join(root, 'images', '{:05d}'.format(
                        int(paths_gallery[i].split('_')[0])), paths_gallery[i]))

    self.transform = self.get_transform()

    def get_transform(self):
        if self.trans == 'norm':
            trans = utils.make_transform(is_train=not self.eval_reid)
        elif self.trans == 'bot':
            trans = utils.make_transform_bot(is_train=not self.eval_reid)
        elif self.trans == 'imgaug':
            trans = utils.make_transform_imaug(is_train=not self.eval_reid)
        elif self.trans == 'randaug':
            trans = utils.make_rand_aug(is_train=not self.eval_reid, M=self.magnitude, N=self.number_aug)
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
        if self.ys_train is not None:
            return im, self.ys_train[index[:-2]] + self.ys_gallery[index[-2]] + self.ys[index[-1]], \
                   self.im_paths_train[index[:-2]] + self.im_paths_gallery[-2] + self.im_paths[index[-1]]
        if self.eval_reid:
            return im, self.ys[index], self.im_paths[index]
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


'''     
if self.imaug:
    im = imageio.imread(self.im_paths[index])
    im = TF.to_tensor(self.transform.augment_image(im))

else:
    im = PIL.Image.open(self.im_paths[index])
    im = self.transform(im)
            
'''
