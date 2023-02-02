import os
from . import utils
import torch
import PIL.Image
import random 


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


class Birds(torch.utils.data.Dataset):
    def __init__(self, root, labels, paths, trans=None,
                 eval_reid=False, rand_scales=False, sz_crop=None):
        # init variables
        self.eval_reid = eval_reid
        self.ys = list()
        self.im_paths = list()
        self.rand_scales = rand_scales
        self.distractor_idx = None
        
        # all labels
        self.labels = labels

        # map labels from 0-#classes
        i = 0
        self.map = dict()
        for lab in sorted(set(self.labels)):
            if lab != -2:
                self.map[lab] = i
                i += 1

        # get img paths and distractor idxs by iterating over labels
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
        
        # get transforms
        self.transform = self.get_transform(trans, sz_crop=sz_crop)

    def get_transform(self, trans, sz_crop):
        if trans == 'norm':
            trans = utils.make_transform(
                is_train=not self.eval_reid, sz_crop=sz_crop)
        elif trans == 'bot':
            # with random erasing
            trans = utils.make_transform_bot(
                is_train=not self.eval_reid, sz_crop=sz_crop)
        elif trans == 'heavy':
            # heavy transformations
            trans = utils.make_transform_heavy(
                    is_train=not self.eval_reid, sz_crop=sz_crop)
        return trans

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])

        # if using random scale, randomly resize image
        if not self.eval_reid and self.rand_scales:
            if random.randint(0, 1):
                r = random.randint(2, 4)
                im = im.resize((int(im.size[0]/r), int(im.size[1]/r)))
        im = self.transform(im)

        return im, self.ys[index], index, self.im_paths[index]
