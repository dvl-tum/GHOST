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












class ManagerReIDOld():
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg):
        self.device = device
        self.reid_cfg, self.dataset_cfg = reid_net_cfg, dataset_cfg
        self.save_folder_nets = reid_net_cfg['save_folder_nets'] + '_inter'
        self.save_folder_nets_final = reid_net_cfg['save_folder_nets']
        if not os.path.isdir(self.save_folder_nets):
            os.mkdir(self.save_folder_nets)
        if not os.path.isdir(self.save_folder_nets_final):
            os.mkdir(self.save_folder_nets_final)
        self.fn = str(time.time())

        #load ReID net
        encoder, sz_embed = ReID.net.load_net(
            reid_net_cfg['trained_on']['name'],
            reid_net_cfg['trained_on']['num_classes'], 'test',
            **reid_net_cfg['encoder_params'])
        
        self.encoder = encoder.to(self.device)
        self.proxy_gen = ProxyGenMLP(sz_embed)

        self.tracker = Tracker(tracker_cfg, encoder)
        if reid_net_cfg['gnn']:
            self.gnn = ReID.net.GNNReID(self.device,
                                   reid_net_cfg['gnn_params'],
                                   sz_embed).cuda(self.device)

            if reid_net_cfg['gnn_params']['pretrained_path'] != "no":
                load_dict = torch.load(
                    reid_net_cfg['gnn_params']['pretrained_path'],
                    map_location='cpu')
                self.gnn.load_state_dict(load_dict)

            graph_gen = ReID.net.GraphGenerator(self.device,
                                                        **reid_net_cfg[
                                                          'graph_params'])
        else:
            self.gnn = None

        self.loaders = self.get_data_loaders(dataset_cfg)
        
        if self.gnn:
            params = list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
        else:
            params = list(set(self.encoder.parameters()))

        param_groups = [{'params': params,
                             'lr': self.reid_cfg['train_params']['lr']}]
        self.optimizer = ReID.RAdam.RAdam(param_groups,
                             weight_decay=self.reid_cfg['train_params']['weight_decay']) 
        
        num_classes = len(set(self.loaders['train'].dataset.ys))
        self.loss1 = ReID.utils.losses.CrossEntropyLabelSmooth(
                        num_classes=num_classes, dev=self.device).to(self.device) 
        self.loss2 = ReID.utils.losses.CrossEntropyLabelSmooth(
                        num_classes=num_classes, dev=self.device).to(self.device)
        self.loss3 = None

    def get_data_loaders(self, dataset_cfg):
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
                                **self.reid_cfg['dl_params'])
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.reid_cfg['dl_params']['num_classes_iter']*self.reid_cfg['dl_params']['num_elements_class'],
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

    def train(self):
        best_rank = 0
        best_mAP = 0
        for e in range(self.reid_cfg['train_params']['epochs']):
            self._train(e)
            rank_1, mAP = self.evaluate()
            if rank > best_rank:
                best_rank = rank
                best_mAP = mAP
                torch.save(self.encoder.state_dict(),
                            osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))
                torch.save(self.gnn.state_dict(),
                            osp.join(self.save_folder_nets,
                                    'gnn_' + self.fn + '.pth'))
        
        logger.info("Best Rank {} and mAP {}".format(best_rank, best_mAP))

        os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
                    osp.join(self.save_folder_nets_final, str(best_rank) + self.net_type + '_' + 
                        self.dataset_cfg['splits'] + '.pth'))
        os.rename(osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
                    osp.join(self.save_folder_nets_final, str(best_rank) + 'gnn_' + self.net_type + '_' +
                        self.dataset_cfg['splits'] + '.pth'))

    def _train(self, e):
        temp = self.reid_cfg['train_params']['temperature']
        for img, Y in self.loaders['train']:
            if e == 31 or e == 51:
                logger.info("reduce learning rate")
                self.encoder.load_state_dict(torch.load(
                    osp.join(self.save_folder_nets, self.fn + '.pth')))
                self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
                    'gnn_' + self.fn + '.pth')))
                for g in self.opt.param_groups:
                    g['lr'] = train_params['lr'] / 10.

            Y, img = Y.to(self.device), img.to(self.device)
            self.optimizer.zero_grad()
             
            preds1, feats1 = self.encoder(img, output_option='plain')
            print(preds1.shape, Y.shape)
            loss = self.loss1(preds1/temp, Y) 
            if self.gnn:
                edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
                preds2, feats2 = self.gnn(feats, edge_ind, edge_attr, 'plain')
                loss += self.loss2(preds2/temp, Y)
            loss.backward()
            self.optimizer.step()
            

    def evaluate(self, mode='val'):
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
        return self.eval_metrics(feats, ys)

    def eval_metrics(self, X, y, topk=20):
        X, y = X.cpu(), y.cpu()
        dist = sklearn.metrics.pairwise.pairwise_distances(X)
        if type(dist) != np.ndarray:
            dist = dist.cpu().numpy()

        indices = np.argsort(dist, axis=1)
        matches = (y[indices] == y[:, np.newaxis])

        aps = []
        ret = np.zeros(topk)
        num_valid_queries = 0
        topk = 1
        for k in range(dist.shape[0]):
            # map
            y_true = matches[k, :]
            y_score = -dist[k][indices[k]]
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        
            # rank
            index = np.nonzero(matches[i, :])[0]
            delta = 1. / len(index)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
            num_valid_queries += 1

        rank_1 = ret.cumsum() / num_valid_queries
        mAP = np.mean(aps)

        logger.info("Rank-1: {}, mAP: {}".format(rank_1, mAP))

        return rank_1, mAP
