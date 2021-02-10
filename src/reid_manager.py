from tracking_wo_bnw.src.tracktor.datasets import factory
from ReID import net
import os.path as osp
import os
import shutil
from torch.utils.data import DataLoader
import ReID.RAdam
import ReID.utils.losses
from torchvision.ops.boxes import clip_boxes_to_image, nms
from ReID.dataset.utils import make_transform_bot
import PIL.Image
from torchvision import transforms
import torch
import sklearn.metrics
import scipy
from collections import defaultdict
import numpy as np
from src.datasets.DatasetReID import ClassBalancedSampler 
from src.datasets.MOT17_parser import ReIDDataset
from data.splits import _SPLITS
from src.nets.proxy_gen import ProxyGenMLP, ProxyGenRNN
from .tracker import Tracker
import sklearn.metrics.pairwise
from sklearn.metrics import average_precision_score
import os.path as osp
import time
import logging
from .manager import Manager
from src.utils import eval_metrics

logger = logging.getLogger('AllReIDTracker.ReIDManager')

class ManagerReID(Manager):
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg):
        super(ManagerReID, self).__init__(device, timer, dataset_cfg, reid_net_cfg, tracker_cfg)
        self.eval1 = 'rank-1'
        self.eval2 = 'mAP'

    def _get_loaders(self, dataset_cfg):
        loaders = dict()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            dir = _SPLITS[dataset_cfg['splits']][mode]['dir']
            dataset = ReIDDataset(dataset_cfg['splits'], seqs, dataset_cfg, dir)

            if mode == 'train':
                ddict = defaultdict(list)
                for idx, label in enumerate(dataset.ys):
                    ddict[label].append(idx)

                list_of_indices_for_each_class = []
                for key in ddict:
                    list_of_indices_for_each_class.append(ddict[key])

                sampler = ClassBalancedSampler(list_of_indices_for_each_class,
                                **self.reid_net_cfg['dl_params'])
                
                bs = self.reid_net_cfg['dl_params']['num_classes_iter']*self.reid_net_cfg['dl_params']['num_elements_class']

                dataloader = DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True)
            else:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=50,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True
                )

            loaders[mode] = dataloader

        return loaders

    def _evaluate(self, mode='val'):
        feats = list()
        ys = list()
        for img, Y in self.loaders[mode]:
            Y, img = Y.to(self.device), img.to(self.device)
            with torch.no_grad():
                _, f = self.encoder(img, output_option='plain')
            feats.extend(f)
            ys.extend(Y)
        feats = torch.stack(feats)
        ys = torch.stack(ys)
        print(feats.shape, ys.shape)

        return eval_metrics(feats, ys)
