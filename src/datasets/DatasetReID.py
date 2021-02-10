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


class DatasetReID(Dataset):
    def __init__(self, sequences, dataset_cfg, dir):
        super(DatasetReID, self).__init__()
        self.sequences = self.add_detector(sequences, dataset_cfg['detector'])
        
        self.mot_dir = osp.join(dataset_cfg['mot_dir'], dir)

        self._vis_threshold = 0
        self.to_tensor = ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.transform = make_transform_bot(is_train=False)
        
        self.ys = list()
        self.samples = list()

        self.current_id = 0
        self.ids_dict = dict()

        self.data = self.get_samps()

    def add_detector(self, sequences, detector):
        if detector == 'all':
            dets = ('DPM', 'FRCNN', 'SDP')
            sequences = ['-'.join([s, d]) for s in sequences for d in dets]
        elif detector == '':
            pass
        else:
            sequences = ['-'.join([s, detector]) for s in sequences]

        return sequences

    def get_samps(self):
        
        for s in self.sequences:
            gt_file = osp.join(self.mot_dir, s, 'gt', 'gt.txt')
            no_gt, boxes, visibility = self.get_gt(gt_file)
            self.get_samples_from_seq(boxes, visibility, s)

    def get_gt(self, gt_file):
        no_gt = False
        boxes, visibility = defaultdict(dict), defaultdict(dict)
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(
                            row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(row[4]) - 1
                        y2 = y1 + int(row[5]) - 1
                        bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                        # frame, person 
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        return no_gt, boxes, visibility

    def get_samples_from_seq(self, boxes, visibility, s):
        img_dir = osp.join(self.mot_dir, s, 'img1')

        for i in range(1, len(boxes)+1):
            for ind, bb in boxes[i].items():
                
                # map frame id to global labels
                if ind in self.ids_dict.keys():
                    lab = self.ids_dict[ind]
                else:
                    self.ids_dict[ind] = self.current_id
                    lab = self.ids_dict[ind]
                    self.current_id += 1

                self.samples.append({'bb': bb, 
                    'img_path': osp.join(img_dir, f"{i:06d}.jpg"),
                    'visibility': visibility[i][ind]})
                self.ys.append(lab)

    def get_image(self, image, rois):
        # tracktor resize (256,128)
        
        res = list()
        for r in rois:
            x0 = int(r[0])
            y0 = int(r[1])
            x1 = int(r[2])
            y1 = int(r[3])
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            im = image[:, y0:y1, x0:x1]
            im = self.to_pil(im)
            im = self.transform(im)
            res.append(im)
        res = res[0]
        return res

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        sample = self.samples[idx]
        img = self.to_tensor(Image.open(sample['img_path']).convert("RGB"))
        bb = torch.tensor([sample['bb'][:4]])
        bb = clip_boxes_to_image(bb, img.shape[-2:])
        
        img = self.get_image(img, bb)
        y = self.ys[idx]

        return img, y

    def __len__(self):
        return len(self.samples)


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
