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
from tracking_wo_bnw.src.tracktor.utils import interpolate, get_mot_accum
from src.utils import evaluate_mot_accums
from src.datasets.MOT import MOT17Test, MOT17Train, collate_test
from data.splits import _SPLITS
from src.nets.proxy_gen import ProxyGenMLP, ProxyGenRNN
from .tracker import Tracker
from src.datasets.MOT17_parser import MOTDataset, collate_train
import time
import random
import logging

logger = logging.getLogger('AllReIDTracker.Manager')


class Manager():
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg, train=False):
        self.device = device
        self.reid_net_cfg = reid_net_cfg
        self.dataset_cfg = dataset_cfg
        self.tracker_cfg = tracker_cfg
        
        self.loaders = self._get_loaders(dataset_cfg)
        if 'train' in self.loaders.keys():
            self.num_classes = self.loaders['train'].dataset.id
        else:
            self.num_classes = self.reid_net_cfg['trained_on']['num_classes']
        self.num_iters = 30 if self.reid_net_cfg['mode'] == 'hyper_search' else 1
        
        #load ReID net
        self._get_models()

        # name of eval criteria for logging
        self.eval1 = 'MOTA'
        self.eval2 = 'IDF1'

    def _get_models(self):
        self._get_encoder()
        if self.reid_net_cfg['gnn']:
            self._get_gnn()
            self.tracker = Tracker(self.tracker_cfg, self.encoder, self.gnn, self.graph_gen)
        else:
            self.gnn, self.graph_gen = None, None
            self.tracker = Tracker(self.tracker_cfg, self.encoder)

    def _get_encoder(self):
        encoder, self.sz_embed = ReID.net.load_net(
            self.reid_net_cfg['trained_on']['name'],
            self.num_classes, 'test',
            **self.reid_net_cfg['encoder_params'])

        self.encoder = encoder.to(self.device)
        self.proxy_gen = ProxyGenMLP(self.sz_embed)
    
    def _get_gnn(self):
        self.reid_net_cfg['gnn_params']['classifier']['num_classes'] = self.num_classes

        self.gnn = ReID.net.GNNReID(self.device,
                               self.reid_net_cfg['gnn_params'],
                               self.sz_embed).cuda(self.device)

        if self.reid_net_cfg['gnn_params']['pretrained_path'] != "no":
            load_dict = torch.load(
                self.reid_net_cfg['gnn_params']['pretrained_path'],
                map_location='cpu')

            load_dict = {k: v for k, v in load_dict.items() if 'fc' not in k.split('.')}
            gnn_dict = self.gnn.state_dict()
            gnn_dict.update(load_dict)
                
            self.gnn.load_state_dict(gnn_dict)

        self.graph_gen = ReID.net.GraphGenerator(self.device,
                                                    **self.reid_net_cfg[
                                                      'graph_params'])
        self.tracker = Tracker(self.tracker_cfg, self.encoder, self.gnn, self.graph_gen)
    
    def _get_loaders(self, dataset_cfg):
        loaders = dict()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            dir = _SPLITS[dataset_cfg['splits']][mode]['dir']
            if mode != 'train':
                dataset = MOT17Test(seqs, dataset_cfg, dir)
                loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=True,
                                        collate_fn=collate_test)
            else:
                dataset = MOTDataset(dataset_cfg['splits'], seqs, dataset_cfg, dir) #MOT17Train(seqs, dataset_cfg, dir)
                loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=True,
                                        collate_fn=collate_train)
        return loaders
    
    def _sample_params(self):

        config = {'lr': 10 ** random.uniform(-5, -3),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'temperatur': random.random(),
                  'epochs': 5}
        self.reid_net_cfg['train_params'].update(config)
        
        if self.reid_net_cfg['gnn_params']['pretrained_path'] == 'no':
            config = {'num_layers': random.randint(1, 4)}
            config = {'num_heads': random.choice([1, 2, 4, 8])}

            self.reid_net_cfg['gnn_params']['gnn'].update(config)
            self._get_gnn()
        
        logger.info("Updated Hyperparameter:")
        logger.info(self.reid_net_cfg)

    def _setup_training(self):
        self.best_mota = 0
        self.best_idf1 = 0
        
        if self.reid_net_cfg['mode'] == 'hyper_search':
            self._sample_params()

        self.save_folder_nets = self.reid_net_cfg['save_folder_nets'] + '_inter'
        self.save_folder_nets_final = self.reid_net_cfg['save_folder_nets']
        if not os.path.isdir(self.save_folder_nets):
            os.mkdir(self.save_folder_nets)
        if not os.path.isdir(self.save_folder_nets_final):
            os.mkdir(self.save_folder_nets_final)
        self.fn = str(time.time())
        
        if self.gnn:
            params = list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
        else:
            params = list(set(self.encoder.parameters()))

        param_groups = [{'params': params,
                            'lr': self.reid_net_cfg['train_params']['lr']}]
        self.optimizer = ReID.RAdam.RAdam(param_groups,
                            weight_decay=self.reid_net_cfg['train_params']['weight_decay']) 
        
        self.loss1 = ReID.utils.losses.CrossEntropyLabelSmooth(
                        num_classes=self.num_classes, dev=self.device).to(self.device) 
        self.loss2 = ReID.utils.losses.CrossEntropyLabelSmooth(
                        num_classes=self.num_classes, dev=self.device).to(self.device)
        self.loss3 = None
    
    def _check_best(self, mota_overall, idf1_overall):
        if mota_overall > self.best_mota:
                self.best_mota = mota_overall
                self.best_idf1 = idf1_overall
                torch.save(self.encoder.state_dict(),
                            osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))
                torch.save(self.gnn.state_dict(),
                            osp.join(self.save_folder_nets,
                                    'gnn_' + self.fn + '.pth'))

    def _save_best(self):
        os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
            osp.join(self.save_folder_nets_final, 
                    str(self.best_mota) + 
                    self.reid_net_cfg['encoder_params']['net_type'] + '_' + 
                    self.dataset_cfg['splits'] + '.pth'))
        os.rename(osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
            osp.join(self.save_folder_nets_final, 
                    str(self.best_mota) + 'gnn_' + 
                    self.reid_net_cfg['encoder_params']['net_type'] + '_' +
                    self.dataset_cfg['splits'] + '.pth'))

    def _reduce_lr(self):
        logger.info("reduce learning rate")
        self.encoder.load_state_dict(torch.load(
            osp.join(self.save_folder_nets, self.fn + '.pth')))
        self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
            'gnn_' + self.fn + '.pth')))
        for g in self.optimizer.param_groups:
            g['lr'] = self.reid_net_cfg['train_params']['lr'] / 10.

    def train(self):
        for i in range(self.num_iters):
            if self.num_iters > 1:    
                logger.info("Search iteration {}/{}".format(i, self.num_iters))
            self._setup_training()
            best_mota_iter, best_idf1_iter = 0, 0
            for e in range(self.reid_net_cfg['train_params']['epochs']):
                logger.info("Epoch {}/{}".format(e, self.reid_net_cfg['train_params']['epochs']))
                self._train(e)
                mota_overall, idf1_overall = self._evaluate()
                self._check_best(mota_overall, idf1_overall)
                if mota_overall > best_mota_iter:
                    best_mota_iter = mota_overall
                    best_idf1_iter = idf1_overall
            logger.info("Iteration {}: Best {} {} and {} {}".format(i, self.eval1, best_mota_iter, self.eval2, best_idf1_iter))

        # save best
        logger.info("Overall Results: Best {} {} and {} {}".format(self.eval1, self.best_mota, self.eval2, self.best_idf1))
        self._save_best()

    def _train(self, e):
        temp = self.reid_net_cfg['train_params']['temperature']
        for data in self.loaders['train']:
            if e == 31 or e == 51:
                self._reduce_lr()
            self.optimizer.zero_grad()

            if type(data[0]) == list: # data == graph of n frames
                img, Y = data[0][0], data[1][0]
            else: # data == bboxes of persons (reid setting)
                img, Y = data[0], data[1]

            Y, img = Y.to(self.device), img.to(self.device)
            preds1, feats1 = self.encoder(img, output_option='plain')
            loss = self.loss1(preds1/temp, Y) 
            if self.gnn:
                edge_attr, edge_ind, feats1 = self.graph_gen.get_graph(feats1)
                preds2, feats2 = self.gnn(feats1, edge_ind, edge_attr, 'plain')
                loss += self.loss2(preds2[-1]/temp, Y)
            loss.backward()
            self.optimizer.step()

    def _evaluate(self, mode='val'):
        mot_accums = list()
        names = list()
        for data in self.loaders[mode]:
            data, target, visibility, im_paths, dets, labs = data
            self.tracker.gnn, self.tracker.encoder = self.gnn, self.encoder
            mot = self.tracker.track(data[0], target[0], visibility[0], im_paths[0], dets[0])
            if mot:
                mot_accums.append(mot)
                names.append(im_paths[0][0].split(os.sep)[-3])

        if len(mot_accums):
            logger.info("Evaluation:")
            summary = evaluate_mot_accums(mot_accums,
                            names,
                            generate_overall=True)

            mota_overall = summary.iloc[summary.shape[0]-1]['mota']
            idf1_overall = summary.iloc[summary.shape[0]-1]['idf1']
            return mota_overall, idf1_overall

        return None, None 

