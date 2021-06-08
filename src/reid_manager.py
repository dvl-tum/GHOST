from typing import Tuple
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
from src.datasets.DatasetReID import ReIDDataset
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
from csv import writer
from datetime import date
from collections import defaultdict

logger = logging.getLogger('AllReIDTracker.ReIDManager')

class ManagerReID(Manager):
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg, separate_seqs=True):
        self.tracker_cfg = tracker_cfg
        self.dataset_cfg = dataset_cfg
        self.separate_seqs = separate_seqs
        self.features = defaultdict(dict)
        super(ManagerReID, self).__init__(device, timer, dataset_cfg, reid_net_cfg, tracker_cfg)
        self.eval1 = 'rank-1'
        self.eval2 = 'mAP'
        if not os.path.isfile('experiments.csv'):
            columns = ['date', 'Sequence', 'vis thresh 1', 'vis thresh 2', 'size thresh 1', 'size thresh 2', 'frame dist thresh 1', 'frame dist thresh 2', 'rel size thresh 1', 'rel size thresh 2', 'number of samples', 'mAP', 'rank-1']
            with open('experiments.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(columns)

    def _get_loaders(self, dataset_cfg):
        loaders = dict()
        seqs = list()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            dir = _SPLITS[dataset_cfg['splits']][mode]['dir']

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
                if self.separate_seqs:
                    query_dls = list()
                    gallery_dls = list()
                    seq_list = list()
                    for seq in seqs:
                        seq_list.append(seq)
                        query_dataset = dataset = ReIDDataset(dataset_cfg['splits'], [seq], dataset_cfg, self.tracker_cfg, dir, 'query')
                        query_dl = torch.utils.data.DataLoader(
                            query_dataset,
                            batch_size=265,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True
                        )
                        query_dls.append(query_dl)
                        if query_dataset.different_gallery_set:
                            gallery_dataset = dataset = ReIDDataset(dataset_cfg['splits'], [seq], dataset_cfg, self.tracker_cfg, dir, 'gallery')
                            gallery_dl = torch.utils.data.DataLoader(
                                gallery_dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True
                            )
                            gallery_dls.append(gallery_dl)
                        else:
                            gallery_dls.append([])
                        break
                        
                    loaders['query'] = query_dls
                    loaders['gallery'] = gallery_dls
                    loaders['seqs_test'] = seq_list
                else:
                    query_dataset = dataset = ReIDDataset(dataset_cfg['splits'], seqs, dataset_cfg, self.tracker_cfg, dir, 'query')
                    query_dl = torch.utils.data.DataLoader(
                        query_dataset,
                        batch_size=265,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True
                    )
                    if query_dataset.gallery_mask is not None:
                        gallery_dl = torch.utils.data.DataLoader(
                            gallery_dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True
                        )
                        gallery_dataset = dataset = ReIDDataset(dataset_cfg['splits'], seqs, dataset_cfg, self.tracker_cfg, dir, 'gallery')
                    loaders['query'] = [query_dl]
                    loaders['gallery'] = []

        return loaders

    def get_distance_once(self, i):

        if len(self.loaders['gallery'][i]) != 0:
            loader = self.loaders['gallery'][i]
        else:
            loader = self.loaders['query'][i]



    def _evaluate(self, mode='test'):
        seqs = _SPLITS[self.dataset_cfg['splits']][mode]['seq']
        for i, seq in enumerate(self.loaders['seqs_test']):

            print("Sequence {}, num query samples {}".format(seq, len(self.loaders['query'][i].dataset))) #, len(self.loaders['gallery'][i].dataset) if len(self.loaders['gallery']) > 0 else len(self.loaders['query'][i].dataset)))
            if len(self.loaders['query'][i].dataset) == 0:
                print("Sequence has no query samples fulfilling the current constraint {}".format(len(self.loaders['query'][i])))
                continue

            import time
            s = time.time()
            q_feats = list()
            q_ys = list()
            for img, idx, Y in self.loaders['query'][i]:
                Y, img = Y.to(self.device), img.to(self.device)
                with torch.no_grad():
                    f = self.encoder(img)
                q_feats.extend(f)
                q_ys.extend(Y)
            q_feats = torch.stack(q_feats)
            q_ys = torch.stack(q_ys)
            print(q_feats.shape, q_ys.shape)

            s = time.time()
            if len(self.loaders['gallery'][i]) != 0:
                g_feats = list()
                g_ys = list()
                for img, idx, Y in self.loaders['gallery'][i]:
                    Y, img = Y.to(self.device), img.to(self.device)
                    with torch.no_grad():
                        f = self.encoder(img)
                    g_feats.extend(f)
                    g_ys.extend(Y)
                g_feats = torch.stack(g_feats)
                g_ys = torch.stack(g_ys)

                gallery_mask = self.loaders['query'][i].dataset.gallery_mask
                rank, mAP = eval_metrics(q_feats, q_ys, g_feats, g_ys, gallery_mask=gallery_mask)
            else:
                gallery_mask = self.loaders['query'][i].dataset.gallery_mask
                rank, mAP = eval_metrics(q_feats, q_ys, gallery_mask=gallery_mask)
            print("Time for galleries {}".format(time.time()-s))
            row = self.get_row(q_feats.shape[0], mAP, rank, seq)

            with open('experiments.csv', 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(row)
            

    def get_row(self, num_samps, mAP, rank, seq):
        new_row = [date.today().strftime("%d.%m.%Y"), seq]
        def add_row(val):  
            if type(val) != tuple:
                if val == 0:
                    new_row.extend(['-', '-'])
                else:
                    new_row.extend([val, 'open'])
            else:
                new_row.extend([val[0], val[1]])

        add_row(self.tracker_cfg['iou_thresh'])
        add_row(self.tracker_cfg['size_thresh'])
        add_row(self.tracker_cfg['frame_dist_thresh'])
        add_row(self.tracker_cfg['size_diff_thresh'])
        new_row.extend([num_samps, mAP, rank[0]])

        return new_row



