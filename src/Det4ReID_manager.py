from numpy import ma
from scipy.sparse import data
from torch.utils.data import DataLoader
from src.datasets.Det4ReIDDataset import Det4ReIDDataset
from data.splits import _SPLITS
import os.path as osp
import logging
from .manager import Manager
from src.utils import eval_metrics
from csv import writer
from collections import defaultdict
import torch
import sklearn
import random
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger('AllReIDTracker.ReIDManager')

class ManagerDet4ReID(Manager):
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg, 
            separate_seqs=True, experiment_name='experiment', train=False, write=True):
        self.tracker_cfg = tracker_cfg 
        print("Eval mode {}".format(self.tracker_cfg['eval_bb']))
        self.dataset_cfg = dataset_cfg
        self.reid_net_cfg = reid_net_cfg
        print("Experiment: {}".format(experiment_name))
        self.separate_seqs = separate_seqs
        self.experiment_name = experiment_name
        self.write = write

        # compute ys and dists only once
        self.ys = dict()
        self.dist = dict()
        self.computed = dict()

        super(ManagerDet4ReID, self).__init__(device, timer, dataset_cfg, reid_net_cfg, tracker_cfg)
        self.eval1 = 'rank-1'
        self.eval2 = 'mAP'

    def add_detector(self, sequence, detector='all', mode='train'):
        if detector == 'all':
            dets = ('DPM', 'FRCNN', 'SDP')
            if mode != 'train':
                self.samp_seqs = [['-'.join([s, d]) for d in dets] for s in sequence]  
            sequence = ['-'.join([s, d]) for s in sequence for d in dets]
        elif detector == '':
            if mode != 'train':
                self.samp_seqs = [[seq] for seq in sequence]
        else:
            sequence = ['-'.join([s, detector]) for s in sequence]
            if mode != 'train':
                self.samp_seqs = [[seq] for seq in sequence]

        return sequence

    def _get_models(self):
        self._get_encoder()

    def _get_loaders(self, dataset_cfg):
        loaders = dict()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            # get information from splits and add detector if nessecary
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            seqs = self.add_detector(seqs, self.dataset_cfg['detector'], mode)
            dir = _SPLITS[dataset_cfg['splits']][mode]['dir']

            logger.info("{} sequences {} from {}".format(mode, seqs, dir))

            if mode == 'train': # get train loader
                dataset = Det4ReIDDataset(dataset_cfg['splits'], seqs, dataset_cfg, \
                    self.tracker_cfg, dir, 'train', augment=False, train=True)
                self.num_classes = dataset.num_classes()
                bs = self.reid_net_cfg['dl_params']['batch_size']

                dataloader = DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                    collate_fn=collate_fn) 
                
                loaders['train'] = dataloader  
                loaders['seqs_train'] = seqs  

                logger.info("Training sequences {} in total: {} frames".format(seqs, len(dataset)))

            else: # get val/test loader
                if mode == 'val':
                    datastorage = 'train'
                else:
                    datastorage = 'test'
                
                # separate test sequences to have per sequence results
                if self.separate_seqs:
                    dataloader = list()
                    for seq in seqs:
                        dataset = Det4ReIDDataset(dataset_cfg['splits'], [seq], dataset_cfg, \
                            self.tracker_cfg, dir, datastorage, augment=False)
                        bs = self.reid_net_cfg['dl_params']['batch_size']

                        _dataloader = DataLoader(
                            dataset,
                            batch_size=bs,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            collate_fn=collate_fn) 
                        
                        dataloader.append(_dataloader)
                        logger.info("Validation sequence {}: {} frames".format(seq, len(dataset)))
                  
                else:
                    dataset = Det4ReIDDataset(dataset_cfg['splits'], seqs, dataset_cfg, self.tracker_cfg, \
                        dir, datastorage)
                    bs = self.reid_net_cfg['dl_params']['batch_size']

                    dataloader = DataLoader(
                        dataset,
                        batch_size=bs,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

                loaders[mode] = dataloader
                loaders['seqs_' + mode] = seqs  

        return loaders

    def set_loss_fns(self):
        # every box refers to if features for every porposal should be computed on every level
        trip = True if 'trip' in self.reid_net_cfg['every_box'] else False
        mult_pos_contr = True if 'mult_pos_contr' in self.reid_net_cfg['every_box'] else False
        self.every_box = trip | mult_pos_contr

        self._get_loss_fns(ce=True, trip=trip, mult_pos_contr=mult_pos_contr)
        self.encoder.roi_heads.embedding_loss = self.loss1
        if trip:
            self.encoder.roi_heads.embedding_loss_trip = self.loss2
            
        if mult_pos_contr:
            self.encoder.roi_heads.embedding_loss_mult = self.loss3

    def train(self):
        # set loss function inside of model
        self.set_loss_fns()
        for i in range(self.num_iters):
            if self.num_iters > 1:    
                logger.info("Search iteration {}/{}".format(i, self.num_iters))

            # set up training
            if self.write:
                self.writer = SummaryWriter('runs/' + self.dataset_cfg['splits'] + "_hyper_search_gt_along_no_aug" + str(i))
            self._setup_training()
            best_rank_iter, best_mAP_iter = 0, 0
            for e in range(self.reid_net_cfg['train_params']['epochs']):
                # train and eval step in epoch i
                logger.info("Epoch {}/{}".format(e, self.reid_net_cfg['train_params']['epochs']))
                print('Train')
                self._train(e)
                print('eval')
                rank, mAP, _ = self._evaluate(e, mode='val')
                logger.info("Performance averaged over all sequences at epoch \
                    {}: Best {} {} and {} {}".format(e, self.eval1, rank, self.eval2, mAP))
                
                # check if results in current epoch better than previous best
                self._check_best(rank, mAP) # rank = mota, mAP = idf1
                if rank > best_rank_iter:
                    best_rank_iter = rank
                    best_mAP_iter = mAP

            logger.info("Iteration {}: Best {} {} and {} {}".format(i, self.eval1, \
                best_rank_iter, self.eval2, best_mAP_iter))
            if self.write:
                self.writer.close()

        # save best
        logger.info("Overall Results: Best {} {} and {} {}".format(self.eval1, \
            self.best_mota, self.eval2, self.best_idf1))
        self._save_best()

    def _evaluate(self, e, mode='test'):
        print("Started evaluating")
        seq_eval_dict = dict()
        if self.tracker_cfg['eval_bb']:
            self.encoder.eval()

        sample_val_seq = [random.choice(v) for v in self.samp_seqs]

        for i, seq in enumerate(self.loaders['seqs_' + mode]):
            if seq not in sample_val_seq:
                continue
            logger.info("Evaluating sequence {}".format(seq))
            feats, ys = list(), list()
            for data in self.loaders[mode][i]:
                # get data
                imgs = data[0]
                labels = data[1]
                labels = [{k: torch.from_numpy(v).to(self.device) for k, v in \
                    labs.items()} for labs in labels]
                imgs = [img.to(self.device) for img in imgs]

                # get embeddings
                with torch.no_grad():
                    results = self.encoder(imgs, labels, every_box=self.every_box)
                embeddings = [r['embeddings'] for r in results]
                id_targets = [r['id_targets'] for r in results]
                feats.extend(embeddings)
                ys.extend(id_targets)
            
            # get dist and evaluate
            feats = torch.stack(feats).cpu()
            dist = sklearn.metrics.pairwise.pairwise_distances(feats)

            self.dist[seq] = dist
            self.ys[seq] = torch.stack(ys).cpu().numpy()
            rank, mAP, num_valid_queries = eval_metrics(y=self.ys[seq], y_g=self.ys[seq], 
                            gallery_mask=None, dist=self.dist[seq])

            # write per sequence results
            if self.write:
                for t, r in zip([rank, mAP], ['rank-1', 'mAP']):
                    self.writer.add_scalar('Accuracy/test/' + seq + '/' + str(r), t, e)

            seq_eval_dict[seq] = {'rank': rank, 'mAP': mAP, 'num_valid_queries': num_valid_queries}
        
        self.encoder.train()

        print(seq_eval_dict)
        # get and wirte overall results
        rank_sum = sum([v['rank'] * v['num_valid_queries'] for k, v in seq_eval_dict.items()])
        map_sum = sum([v['mAP'] * v['num_valid_queries'] for k, v in seq_eval_dict.items()])
        val_queries_sum = sum([v['num_valid_queries'] for k, v in seq_eval_dict.items()])
        print(rank_sum, map_sum, val_queries_sum)
        if self.write:
            for t, r in zip([rank_sum/val_queries_sum, map_sum/val_queries_sum], ['rank-1', 'mAP']):
                self.writer.add_scalar('Accuracy/test/' + 'overall' + '/' + str(r), t, e)
        
        return rank_sum/val_queries_sum, map_sum/val_queries_sum, seq_eval_dict

    def _train(self, e):
        temp = self.reid_net_cfg['train_params']['temperature']
        losses = defaultdict(list)
        self.encoder.train()

        for data in self.loaders['train']:
            imgs = data[0]
            labels = data[1]

            if e == 31 or e == 51:
                self._reduce_lr()
            self.optimizer.zero_grad()

            labels = [{k: torch.from_numpy(v).to(self.device) for k, v in labs.items()} for labs in labels]
            imgs = [img.to(self.device) for img in imgs]

            _losses = self.encoder(imgs, labels, every_box=self.every_box)
            loss = 0
            for k, v in _losses.items():
                losses[k].append(v)
                loss += v

            losses['Total'].append(loss)
            if self.write:
                for k, v in losses.items():
                    self.writer.add_scalar('Loss/train/' + k, v[-1], e)
            
            loss.backward()
            self.optimizer.step()

        logger.info({k: sum(v)/len(v) for k, v in losses.items()})  
            

def collate_fn(batch):
    return tuple(zip(*batch))

