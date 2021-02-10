from torch.utils.data.sampler import Sampler
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


class MOT17Test(Dataset):
    def __init__(self, sequences, dataset_cfg, dir):
        super(MOT17Test, self).__init__()
        self.sequences = self.add_detector(sequences, dataset_cfg['detector'])
        
        self.mot_dir = osp.join(dataset_cfg['mot_dir'], dir)
        self.det_dir = osp.join(dataset_cfg['det_dir'], dir)
        self.det_file = dataset_cfg['det_file']

        self._vis_threshold = 0
        self.to_tensor = ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.transform = make_transform_bot(is_train=False)

        self.data = self.get_seqs()

    def add_detector(self, sequences, detector):
        if detector == 'all':
            dets = ('DPM', 'FRCNN', 'SDP')
            sequences = ['-'.join([s, d]) for s in sequences for d in dets]
        elif detector == '':
            pass
        else:
            sequences = ['-'.join([s, detector]) for s in sequences]

        return sequences

    def get_seqs(self):
        sequences = list()
        
        for s in self.sequences:
            gt_file = osp.join(self.mot_dir, s, 'gt', 'gt.txt')
            det_file = osp.join(self.det_dir, s, 'det', self.det_file)
            print(gt_file, det_file)
            no_gt, boxes, visibility = self.get_gt(gt_file)
            dets, labs = self.get_dets(det_file)
            
            samples = self.get_sample_list(boxes, visibility, dets, s, labs)
            sequences.append(samples)

        return sequences

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
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        return no_gt, boxes, visibility

    def get_dets(self, det_file):
        dets = defaultdict(list)
        labs = defaultdict(list)
        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)
                    labs[int(float(row[0]))].append(row[1])
        return dets, labs

    def get_sample_list(self, boxes, visibility, dets, s, labs):
        img_dir = osp.join(self.mot_dir, s, 'img1')
        
        samples = [{'gt': boxes[i],
                    'im_path': osp.join(img_dir, f"{i:06d}.jpg"),
                    'vis': visibility[i],
                    'dets': dets[i],
                    'labs': labs[i]} for i in range(1, len(dets)+1)]
        return samples

    def get_images(self, image, rois):
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
        
        res = torch.stack(res, 0)
        res = res.cuda()
        
        return res

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        seq = self.data[idx]

        seq_imgs = list()
        seq_gt = list()
        seq_vis = list()
        seq_paths = list()
        seq_dets = list()
        seq_labs = list() 
        for frame in seq:
            img = self.to_tensor(Image.open(frame['im_path']).convert("RGB"))
            dets = torch.tensor([det[:4] for det in frame['dets']])
            dets = clip_boxes_to_image(dets, img.shape[-2:])

            seq_imgs.append(self.get_images(img, dets))
            seq_gt.append(frame['gt'])
            seq_vis.append(frame['vis'])
            seq_paths.append(frame['im_path'])
            seq_dets.append(frame['dets'])
            seq_labs.append(frame['labs'])
        return seq_imgs, seq_gt, seq_vis, seq_paths, seq_dets, seq_labs

    def __len__(self):
        return len(self.data)


def collate_test(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    visibility = [item[2] for item in batch]
    im_path = [item[3] for item in batch]
    dets = [item[4] for item in batch]
    labs = [item[5] for item in batch]

    return [data, target, visibility, im_path, dets, labs]


class MOT17Train(Dataset):
    def __init__(self, sequences, dataset_cfg, dir):
        super(MOT17Train, self).__init__()
        self.sequences = self.add_detector(sequences, dataset_cfg['detector'])
        
        self.mot_dir = osp.join(dataset_cfg['mot_dir'], dir)
        self.det_dir = osp.join(dataset_cfg['det_dir'], dir)
        self.det_file = dataset_cfg['det_file']
        print(self.mot_dir, self.det_dir, self.det_file)
        self._vis_threshold = 0
        self.to_tensor = ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.transform = make_transform_bot(is_train=False)

        self.data = self.get_seqs()
        self.make_n_frame_packs()

    def add_detector(self, sequences, detector):
        if detector == 'all':
            dets = ('DPM', 'FRCNN', 'SDP')
            sequences = ['-'.join([s, d]) for s in sequences for d in dets]
        elif detector == '':
            pass
        else:
            sequences = ['-'.join([s, detector]) for s in sequences]

        return sequences

    def get_seqs(self):
        sequences = list()

        for s in self.sequences:
            gt_file = osp.join(self.mot_dir, s, 'gt', 'gt.txt')
            det_file = osp.join(self.det_dir, s, 'det', self.det_file)
            
            no_gt, boxes, visibility = self.get_gt(gt_file)
            dets, labs = self.get_dets(det_file)

            samples = self.get_sample_list(boxes, visibility, dets, s, labs)
            sequences.append(samples)

        return sequences

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
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        return no_gt, boxes, visibility

    def get_dets(self, det_file):
        dets = defaultdict(list)
        labs = defaultdict(list)
        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)
                    labs[int(float(row[0]))].append(row[1])
        return dets, labs

    def get_sample_list(self, boxes, visibility, dets, s, labs):
        img_dir = osp.join(self.mot_dir, s, 'img1')

        samples = [{'gt': boxes[i],
                    'im_path': osp.join(img_dir, f"{i:06d}.jpg"),
                    'vis': visibility[i],
                    'dets': dets[i],
                    'labs': labs[i]} for i in range(1, len(dets)+1)]
        return samples
    
    def make_n_frame_packs(self, n=2):
        n_packs = list()
        for seq in self.data:
            for i in range(len(seq)-n):
                n_packs.append([seq[i+j] for j in range(n)])
        self.data = n_packs

    def get_images(self, image, rois):
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

        res = torch.stack(res, 0)
        res = res.cuda()

        return res

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        n_pack = self.data[idx]

        seq_imgs = list()
        seq_gt = list()
        seq_vis = list()
        seq_paths = list()
        seq_dets = list()
        seq_labs = list()

        for frame in n_pack:
            img = self.to_tensor(Image.open(frame['im_path']).convert("RGB"))
            dets = torch.tensor([det[:4] for det in frame['dets']])
            dets = clip_boxes_to_image(dets, img.shape[-2:])
            seq_imgs.append(self.get_images(img, dets))
            seq_gt.extend(list(frame['gt'].values()))
            print(frame['gt'])
            print(frame['vis'])
            print(frame['dets'])
            seq_vis.extend(list(frame['vis'].values()))
            seq_paths.extend([frame['im_path']] * len(frame['dets']))
            seq_dets.extend(frame['dets'])
            seq_labs.extend(frame['labs'])

        seq_imgs = torch.cat(seq_imgs)
        print(seq_imgs.shape)
        print(len(seq_gt))
        print(len(seq_vis))
        print(len(seq_paths))
        print(len(seq_dets))
        print(len(seq_labs))
        quit()
        return seq_imgs, seq_gt, seq_vis, seq_paths, seq_dets, seq_labs

    def __len__(self):
        return len(self.data)


class TwoFrameSampler(Sampler):
    def __init__(self, fames):
        pass
