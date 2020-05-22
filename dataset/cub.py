import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile
import imageio
import matplotlib.pyplot as plt


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
    def __init__(self, root, labels, paths, transform=None,
                 eval_reid=False, imgaug=False):
        self.imgaug = imgaug
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
        im = pil_loader(self.im_paths[index])
        #show_dataset(im, self.ys[index])
        im = self.transform(im)
        #show_dataset(im, self.ys[index])

        if self.eval_reid:

            return im, self.ys[index], self.im_paths[index]
        return im, self.ys[index]


'''     
if self.imaug:
    im = imageio.imread(self.im_paths[index])
    im = TF.to_tensor(self.transform.augment_image(im))

else:
    im = PIL.Image.open(self.im_paths[index])
    im = self.transform(im)
            
'''