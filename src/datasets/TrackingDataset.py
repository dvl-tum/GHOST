from cProfile import label
import os

from numpy.core.fromnumeric import shape
import torch
from .MOT17_parser import MOTLoader
from .bdd100k_parser import BDDLoader
import pandas as pd
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import csv
from torchvision.ops.boxes import clip_boxes_to_image
import random
import torch.nn.functional as F
import logging
import copy
from torchvision.transforms import ToTensor
from torchvision import transforms
from ReID.data.utils import make_transform_bot, make_transfor_obj_det, make_transform_IBN


logger = logging.getLogger('AllReIDTracker.TrackingDataset')


class TrackingDataset():
    def __init__(
            self,
            split,
            sequences,
            dataset_cfg,
            dir,
            datastorage='data',
            net_type='resnet50',
            dev=None,
            add_detector=True,
            assign_gt=False):

        self.device = dev
        self.assign_gt = assign_gt

        self.split = split
        if add_detector:
            self.sequences = self.add_detector(
                sequences, dataset_cfg['detector'])
        else:
            self.sequences = sequences

        self.dataset_cfg = dataset_cfg
        self.dir = dir
        self.datastorage = datastorage
        self.data = list()

        self.to_tensor = ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.transform_det = make_transfor_obj_det(is_train=False)
        self.net_type = net_type

        # different preprocessing
        if net_type == "IBN":
            self.transform = make_transform_IBN()
        else:
            self.transform = make_transform_bot(
                is_train=False, sz_crop=dataset_cfg['sz_crop'])

        self.process()

    def __len__(self):
        return len(self.data)

    def add_detector(self, sequence, detector):
        """
        Add detector to sequence names
        """
        if detector == 'all':
            dets = ('DPM', 'FRCNN', 'SDP')
            sequence = ['-'.join([s, d]) for s in sequence for d in dets]
        elif detector == '':
            pass
        else:
            sequence = ['-'.join([s, detector]) for s in sequence]

        return sequence

    def process(self):
        self.data = list()
        for seq in self.sequences:
            # load and preprocess clipped (to image size) and unclipped
            # detections
            if 'bdd' in self.dataset_cfg['mot_dir']:
                loader = BDDLoader([seq], self.dataset_cfg, self.dir)
            else:
                loader = MOTLoader([seq], self.dataset_cfg, self.dir)
            exist_gt = loader.get_seqs(self.assign_gt)
            dets = loader.dets
            dets_unclipped = loader.dets_unclipped

            # save ground truth bbs and ground truth bbs corresponding to
            # detecionts
            if exist_gt and not 'bdd' in loader.mot_dir:
                gt = loader.gt
                corresponding_gt = loader.corresponding_gt
            else:
                corresponding_gt = None
                gt = None

            self.data.append(Sequence(name=seq, dets=dets, gt=gt,
                                      to_pil=self.to_pil,
                                      to_tensor=self.to_tensor,
                                      transform=self.transform,
                                      dev=self.device,
                                      dets_unclipped=dets_unclipped,
                                      corresponding_gt=corresponding_gt,
                                      transform_det=self.transform_det,
                                      fixed_aspect_ratio=self.dataset_cfg['fixed_aspect_ratio'],
                                      net_type=self.net_type))

    def __getitem__(self, idx):
        """Return the ith sequence converted"""
        seq = self.data[idx]

        return [seq]


class Sequence():
    def __init__(
            self,
            name,
            dets,
            gt,
            to_pil,
            to_tensor,
            transform,
            dev=None,
            dets_unclipped=None,
            corresponding_gt=None,
            padding=True,
            transform_det=None,
            use_unclipped_for_eval=True,
            fixed_aspect_ratio=0,
            net_type='resnet50'):
        # detections and ground truth
        self.dets = dets
        self.dets_unclipped = dets_unclipped
        self.gt = gt
        self.corresponding_gt = corresponding_gt
        self.name = name
        self.net_type = net_type

        # transformations
        self.to_pil = to_pil
        self.to_tensor = to_tensor
        self.transform = transform

        # parameters
        self.device = dev
        self.num_frames = len(self.dets['frame'].unique())
        self.padding = padding
        self.random_patches = False
        self.transform_det = transform_det
        self.use_unclipped_for_eval = use_unclipped_for_eval
        self.fixed_aspect_ratio = fixed_aspect_ratio

        logger.info("Padding of images {}".format(self.padding))
        logger.info("Using unclipped detections for evaluation {}".format(
            use_unclipped_for_eval))

    def _get_random_patches(self, img, height_max: int = 256,
                            height_min: int = 64, width_max: int = 256,
                            width_min: int = 64, num_patches: int = 128,
                            frame_size: tuple = (560, 1024)):

        # generate random patches for BatchNorm statistics if needed
        frame_size = (img.shape[1], img.shape[2])
        patches = list()
        heights = random.choices(range(height_min, height_max), k=num_patches)
        widths = random.choices(range(width_min, width_max), k=num_patches)
        for h, w in zip(heights, widths):
            x_pos = random.choice(
                range(1 + int(w / 2), frame_size[1] - int(w / 2)))
            y_pos = random.choice(
                range(1 + int(h / 2), frame_size[0] - int(h / 2)))

            im = img[:, y_pos:y_pos + h, x_pos:x_pos + w]
            im = self.to_pil(im)
            im = self.transform(im)
            patches.append(im)

        return torch.stack(patches, 0).to(self.device)

    def pads(self, img, row_unclipped):
        left_pad = abs(int(row_unclipped['bb_left'])) if int(
            row_unclipped['bb_left']) < 0 else 0
        right_pad = abs(
            int(
                row_unclipped['bb_right']) -
            img.shape[2]) if int(
            row_unclipped['bb_right']) > img.shape[2] else 0
        top_pad = abs(int(row_unclipped['bb_top'])) if int(
            row_unclipped['bb_top']) < 0 else 0
        bot_pad = abs(int(row_unclipped['bb_bot']) - img.shape[1]
                      ) if int(row_unclipped['bb_bot']) > img.shape[1] else 0

        to_pad = sum([left_pad, right_pad, top_pad, bot_pad])

        return left_pad, right_pad, top_pad, bot_pad, to_pad

    def pad_bbs(
            self,
            padding,
            im,
            row_unclipped,
            left_pad,
            right_pad,
            top_pad,
            bot_pad):

        # if keeping fixed aspect ratio
        if self.fixed_aspect_ratio:
            w = row_unclipped['bb_right'] + right_pad - \
                row_unclipped['bb_left'] - left_pad
            h_fixed = w * self.fixed_aspect_ratio
            h = row_unclipped['bb_bot'] + bot_pad - \
                row_unclipped['bb_top'] - top_pad
            dh = h_fixed - h
            bot_pad += dh
            bot_pad = round(bot_pad)

        # zero padding
        if padding == 'zero':
            m = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bot_pad))
            im = m(im)

        # padding with image mean
        elif padding == 'mean':
            im = F.pad(
                im,
                (left_pad,
                 right_pad,
                 top_pad,
                 bot_pad),
                "constant",
                im.mean())

        # padding with channel wise mean
        elif padding == 'channel_wise_mean':
            im = torch.stack([F.pad(im[i, :, :], (left_pad, right_pad, top_pad, bot_pad), "constant", im.mean(
                dim=1).mean(dim=1)[i]) for i in range(im.shape[0])])

        # others like 'circular'
        else:
            left_pad = left_pad if left_pad < im.shape[2] else im.shape[2] - 1
            right_pad = right_pad if right_pad < im.shape[2] else im.shape[2] - 1
            top_pad = top_pad if top_pad < im.shape[1] else im.shape[1] - 1
            bot_pad = bot_pad if bot_pad < im.shape[1] else im.shape[1] - 1
            im = F.pad(
                im.unsqueeze(0),
                (left_pad,
                 right_pad,
                 top_pad,
                 bot_pad),
                padding).squeeze()

        return im

    def get_fixed_ratio(self, img, row):
        w = row['bb_right'] - row['bb_left']
        h_fixed = w * self.fixed_aspect_ratio
        h = row['bb_bot'] - row['bb_top']
        dh = h_fixed - h
        row['bb_bot'] += dh

        im = img[:, int(row['bb_top']):int(row['bb_bot']), int(
            row['bb_left']):int(row['bb_right'])]

        return im, row

    def _get_images(self, path, dets_frame, dets_uncl_frame, padding='zero'):
        # get image
        if self.net_type == 'IBN':
            img = Image.open(path)
            img = np.asarray(img)
            img = Image.fromarray(img)
            img = self.to_tensor(img)
        else:
            img = self.to_tensor(Image.open(path).convert("RGB"))
        img_for_det = copy.deepcopy(img)

        # initialize return lists
        res, dets, ids, vis, areas_out, conf, label = \
            list(), list(), list(), list(), list(), list(), list()

        # generate random patches if BatchNorm stats are updated with those
        if self.random_patches:
            random_patches = self._get_random_patches(img)
        else:
            random_patches = None

        # iterate over bbs in frame
        for ind, row in dets_frame.iterrows():
            # get unclipped detections and bb (im)
            row_unclipped = dets_uncl_frame.loc[ind]

            # if padding get size of pads
            if self.padding:
                left_pad, right_pad, top_pad, bot_pad, to_pad = \
                    self.pads(img, row_unclipped)
            else:
                left_pad, right_pad, top_pad, bot_pad, to_pad = \
                    0, 0, 0, 0, False
        
            # if keep fixed aspect ratio if not padding
            if self.fixed_aspect_ratio and not (to_pad and self.padding):
                im, row = self.get_fixed_ratio(img, row)
            # get crop of image
            else:
                im = img[:, int(row['bb_top']):int(row['bb_bot']), int(
                    row['bb_left']):int(row['bb_right'])]

            # pad if part of bb outside of image
            if self.padding and to_pad:
                im = self.pad_bbs(
                    padding,
                    im,
                    row_unclipped,
                    left_pad,
                    right_pad,
                    top_pad,
                    bot_pad)

            # transform bb
            im = self.to_pil(im)
            # im.save(f'{ind}.jpg')
            im = self.transform(im)

            # append to bbs, detections, tracktor ids, ids and visibility
            res.append(im)
            if self.padding or self.use_unclipped_for_eval:
                dets.append(np.array([row_unclipped['bb_left'],
                                      row_unclipped['bb_top'],
                                      row_unclipped['bb_right'],
                                      row_unclipped['bb_bot']],
                                     dtype=np.float32))
            else:
                dets.append(np.array([row['bb_left'], row['bb_top'], row[
                    'bb_right'], row['bb_bot']], dtype=np.float32))

            ids.append(row['id'])
            vis.append(row['vis'])
            conf.append(row['conf'])
            label.append(row['label'])

        # different scaling of networks
        if self.net_type == "IBN":
            res = torch.stack(res, 0) * 255
        else:
            res = torch.stack(res, 0)

        res = res.to(self.device)

        return res, dets, ids, vis, random_patches, img_for_det.to(
            self.device), conf, label
    
    def __len__(self):
        return self.num_frames

    def __iter__(self):
        self.frames = self.dets['frame'].unique()
        self.i = 0
        return self

    def __next__(self):
        # iterate over frames
        if self.i < self.num_frames:
            out = self._get(self.i)
            self.i += 1
            return out
        else:
            raise StopIteration

    def _get(self, idx):
        frame = self.frames[idx]
        dets_frame = self.dets[self.dets['frame'] == frame]
        dets_uncl_frame = self.dets_unclipped[self.dets_unclipped['frame'] == frame]

        assert len(dets_frame['frame_path'].unique()) == 1

        img, dets_f, ids, vis, random_patches, img_for_det, conf, label = self._get_images(
            dets_frame['frame_path'].unique()[0], dets_frame, dets_uncl_frame)

        return (img, dets_frame['frame_path'].unique(
            )[0], dets_f, ids, vis, random_patches, img_for_det, conf, label)
