from tracking_wo_bnw.src.tracktor.datasets import factory
from ReID import net
import os.path as osp
import os
import shutil
from torch.utils.data import DataLoader
import ReID
from torchvision.ops.boxes import clip_boxes_to_image, nms
from ReID.dataset.utils import make_transform_bot
import PIL.Image
from torchvision import transforms
import torch
import sklearn.metrics
import scipy
from collections import defaultdict
import numpy as np
from tracking_wo_bnw.src.tracktor.utils import interpolate, get_mot_accum, \
    evaluate_mot_accums
from src.datasets.MOT import MOT17, collate
from data.splits import _SPLITS
from src.nets.proxy_gen import ProxyGenMLP, ProxyGenRNN



class Manager():
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg):
        self.device = device

        #load ReID net
        self.encoder, sz_embed = ReID.net.load_net(
            reid_net_cfg['trained_on']['name'],
            reid_net_cfg['trained_on']['num_classes'], 'test',
            **reid_net_cfg['encoder_params'])
        
        self.encoder = self.encoder.to(self.device)
        self.proxy_gen = ProxyGenMLP(sz_embed)

        self.tracker(tracker_cfg)

        self.loaders = self.get_data_loaders(dataset_cfg)

    def get_data_loaders(self, dataset_cfg):
        loaders = dict()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            print(mode)
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            dir = _SPLITS[dataset_cfg['splits']][mode]['dir']
            dataset = MOT17(seqs, dataset_cfg, dir)

            loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=True,
                                       collate_fn=collate)
        return loaders

    def train(self):
        for data in self.loaders['train']:
            data, target, visibility = data

    def evaluate(self, mode='val'):
        mot_accums = list()
        for data in self.loaders[mode]:
            mot_accums.append(self.tracker.track())

    def make_results(self):
        results = defaultdict(dict)
        for id, ts in self.tracks.items():
            for t in ts:
                results[id][t['im_index']] = np.concatenate([t['bbox'].numpy(), np.array([-1])])

        return results
