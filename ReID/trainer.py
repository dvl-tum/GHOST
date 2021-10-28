import os.path as osp
import logging
import random
import net
import dataset
from RAdam import RAdam
from collections import defaultdict
import torch.nn as nn
from utils import losses
import torch
from apex import amp
import random
import data_utility
import time
import torch.nn.functional as F
import copy
import sys
from evaluation import Evaluator
import utils.utils as utils
import matplotlib.pyplot as plt
import os
import json
from torch import autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
autograd.set_detect_anomaly(True)

logger = logging.getLogger('GNNReID.Training')

torch.manual_seed(0)


class Trainer():
    def __init__(self, config, save_folder_nets, save_folder_results,
                 device, timer, write=True):
        
        self.make_det()

        self.config = config
        if 'attention' in self.config['train_params']['loss_fn']['fns']:
            self.config['models']['attention'] = 1
        else:
            self.config['models']['attention'] = 0
        self.device = device
        self.save_folder_results = save_folder_results
        self.save_folder_nets = save_folder_nets
        self.timer = timer
        self.fn = self.config['dataset'][
                      'dataset_short'] + '_intermediate_model_' + str(
            timer)
        self.net_type = self.config['models']['encoder_params']['net_type']
        self.dataset_short = self.config['dataset']['dataset_short']

        self.best_hypers = None
        self.num_iter = 30 if 'hyper' in config['mode'].split('_') else 1
        self.write = write

    def make_det(self, seed=1):
        logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        #torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
    
    def init_iter(self, i):
        self.train_params = self.config['train_params']
        self.model_params = self.config['models']

        if self.write:
            self.writer = SummaryWriter('runs/' + "lsce1.0_small_eval") # + str(i))
        self.make_det(random.randint(0, 100))
        logger.info('Search iteration {}'.format(i + 1))
        mode = self.get_save_name()
        self.update_params()
        logger.info(self.config)
        logger.info(self.timer)
        self.device = 0
        if torch.cuda.device_count() > 1:
            self.gnn_dev = 1
        else:
            self.gnn_dev = 0
        
        return mode

    def train(self):
        # initialize rank and hypers
        best_rank = 0
        best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])

        for i in range(self.num_iter):
            # init writer, make det det save name and update params
            mode = self.init_iter(i)

            # get models
            encoder, sz_embed = net.load_net(
                self.config['dataset']['dataset_short'],
                self.config['dataset']['num_classes'],
                self.config['mode'],
                self.model_params['attention'],
                add_distractors=self.config['dataset']['add_distractors'],
                **self.model_params['encoder_params'])
            self.encoder = encoder.cuda(self.device)  # to(self.device)
            params = self.get_gnn(sz_embed)
            
            # get evaluator
            self.evaluator = Evaluator(**self.config['eval_params'])    

            # get optimizer
            param_groups = [{'params': params,
                             'lr': self.train_params['lr']}]
            self.opt = RAdam(param_groups,
                             weight_decay=self.train_params[
                                 'weight_decay'])

            # get loss functions
            self.get_loss_fn(self.train_params['loss_fn'],
                             self.config['dataset']['num_classes'],
                             self.config['dataset']['add_distractors'])

            # Do training in mixed precision
            if self.train_params['is_apex']:
                [self.encoder, self.gnn], self.opt = amp.initialize(
                    [self.encoder, self.gnn], self.opt,
                    opt_level="O1")

            # do paralel training in case there are more than one gpus
            if torch.cuda.device_count() > 1:
                self.encoder = nn.DataParallel(self.encoder)

            # get data
            self.get_data(self.config['dataset'], self.train_params,
                          self.config['mode'])

            # execute training
            best_rank_iter, model = self.execute(
                self.train_params,
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Rank-1: {}'.format(best_rank_iter))

            # save best model of iterations
            if best_rank_iter > best_rank:
                os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
                          str(best_rank_iter) + mode + self.net_type + '_' +
                          self.dataset_short + '.pth')

                if self.gnn is not None or self.query_guided_attention is not None:
                    os.rename(
                        osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
                        str(best_rank_iter) + 'gnn_' + mode + self.net_type + '_' +
                        self.dataset_short + '.pth')

                best_rank = best_rank_iter
                
                best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])
            
            if self.write:
                self.writer.close()

        logger.info("Best Hyperparameters found: " + best_hypers)
        logger.info(
            "Achieved {} with this hyperparameters".format(best_rank))
        logger.info("-----------------------------------------------------\n")

    def execute(self, train_params, eval_params):
        best_rank_iter = 0
        scores = list()

        for e in range(1, train_params['num_epochs'] + 1):
 
            if 'test' in self.config['mode'].split('_'):
                best_rank_iter = self.evaluate(eval_params, scores, 0, 0)
            # If not testing
            else:
                logger.info(
                    'Epoch {}/{}'.format(e, train_params['num_epochs']))

                self.milestones(e, train_params, logger)

                for i, (x, Y, I, P) in enumerate(self.dl_tr):
                    loss = self.forward_pass(x, Y, I, P, train_params, e)

                    if i % 100 == 0:
                        logger.info("Iteration {}/{}: {}".format(i, len(self.dl_tr), \
                            {k: l[-1] for k, l in self.losses.items()}))

                    if train_params['is_apex']:
                        with amp.scale_loss(loss, self.opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    self.opt.step()

            self.log()
            # compute ranks and mAP at the end of each epoch
            if e > 5 and best_rank_iter < 0.1 and self.num_iter > 1:
                break

        return best_rank_iter, self.encoder

    def forward_pass(self, x, Y, I, P, train_params, e):
        Y = Y.cuda(self.device)
        self.opt.zero_grad()
        
        # bb for spatial attention (same bb but different output)
        if self.model_params['attention']:
            probs, fc7, attention_features = self.encoder(x.cuda(self.device),
                                output_option=train_params['output_train_enc'])
        else:
            # bb for training with distractors (same bb but different output)
            if self.config['dataset']['add_distractors']:
                probs, fc7, distractor_bce = self.encoder(x.cuda(self.device),
                                output_option=train_params['output_train_enc'])
            # normal training (same bb but different output)
            else:
                probs, fc7 = self.encoder(x.cuda(self.device),
                                output_option=train_params['output_train_enc'])
            
        # query guided attention output
        if self.model_params['attention'] and not self.gnn:
            _, _, _, qs, gs, attended_feats, fc_x = \
                self.query_guided_attention(attention_features)

        # Compute CE Loss
        loss = 0
        if self.ce:
            if type(probs) != list:
                probs = [probs]

            for i, p in enumerate(probs):
                # -2 is the distractor label
                p = p[Y!=-2]
                _Y = Y[Y!=-2]
                loss0 = self.ce(p/self.train_params['temperatur'], _Y)
                loss += train_params['loss_fn']['scaling_ce'] * loss0
                self.losses['Cross Entropy ' + str(i)].append(loss0.item())
        
        if self.bce_distractor:
            # -2 again is distractor label
            bce_lab = torch.zeros(Y.shape)
            bce_lab[Y!=-2] = 1
            
            distrloss = self.bce_distractor(distractor_bce.squeeze(), bce_lab.squeeze())
            loss += train_params['loss_fn']['scaling_bce'] * distrloss
            self.losses['BCE Loss Distractors ' + str(i)].append(distrloss.item())

        if self.tripletattention:
            _Y = Y[Y!=-2]
            triploss, _ = self.tripletattention(dist=dist, targets=_Y)
            loss += train_params['loss_fn']['scaling_triplet'] * triploss
            self.losses['Triplet'].append(triploss.item())

        if self.bceattention:
            ql = Y[qs]
            gl = Y[gs]
            same_class = (ql == gl).float().cuda(self.device)
            bceattloss = self.bceattention(dist[qs, gs].squeeze(), same_class.squeeze())
            loss += train_params['loss_fn']['scaling_bce'] * bceattloss
            self.losses['BCE Loss Atttention ' + str(i)].append(bceattloss.item())
        
        if self.multiattention:
            # only take samples that were attended by positive queries, query label == gallery label
            mask = Y[gs] == Y[qs]
            multattloss = self.multiattention(fc_x[mask]/self.train_params['temperatur'], Y[gs][mask])
            # multattloss = self.multiattention(fc_x/self.train_params['temperatur'], Y[gs])
            scale = train_params['loss_fn']['scaling_multiattention']
            loss += scale * multattloss
            self.losses['CE Loss Atttention ' + str(i)].append(scale * multattloss.item())
        
        # Compute MultiPositiveContrastive
        if self.multposcont:
            dist = self.get_dist(attended_feats, attended_feats)
            loss0 = self.multposcont(dist, Y[gs])
            scale = train_params['loss_fn']['scaling_multiposcont']
            loss += scale * loss0
            self.losses['Multi Positive Contrastive ' + str(i)].append(scale * loss0.item())

        if self.gnn_loss:

            data = self.graph_generator.get_graph(fc7, Y)
            edge_attr = data[0].cuda(self.gnn_dev)
            edge_index = data[1].cuda(self.gnn_dev)
            if self.model_params['attention']:
                fc7 = attention_features.cuda(self.gnn_dev)
            else:
                fc7 = data[2].cuda(self.gnn_dev)

            if type(loss) != int:
                loss = loss.cuda(self.gnn_dev)
            pred, feats, Y = self.gnn(fc7, edge_index, edge_attr, Y,
                                        train_params['output_train_gnn'],
                                        mode='train')
            

            if self.gnn_loss:
                if self.every:
                    loss1 = [gnn_loss(
                        pr / self.train_params['temperatur'],
                        Y.cuda(self.gnn_dev)) for gnn_loss, pr in
                                zip(self.gnn_loss, pred)]
                else:
                    loss1 = [self.gnn_loss(
                        pred[-1] / self.train_params[
                            'temperatur'], Y.cuda(self.gnn_dev))]

                loss += sum([train_params['loss_fn']['scaling_gnn'] * l
                                for l in loss1])

                [self.losses['GNN' + str(i)].append(l.item()) for i, l in
                    enumerate(loss1)]


            # Compute Triplet Loss
            if self.triplet:
                _fc7 = fc7[Y!=-2]
                _Y = Y[Y!=-2]
                triploss, _ = self.triplet(_fc7, _Y)
                loss += train_params['loss_fn']['scaling_triplet'] * triploss
                self.losses['Triplet'].append(triploss.item())

        if self.write:
            logger.info("Write losses")
            for k, v in self.losses.items():
                self.writer.add_scalar('Loss/train/' + k, v[-1], e)

        return loss

    def evaluate(self, eval_params, scores, e, best_rank_iter):
        if not self.config['mode'] == 'pretraining':
            with torch.no_grad():
                logger.info('EVALUATION')
                if self.config['mode'] in ['train', 'test', 'hyper_search']:
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       gallery=self.gallery,
                                                       add_dist= 
                                                       self.config['dataset'][
                                                            'add_distractors'])
                elif self.gnn is not None:
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       self.gallery, self.gnn,
                                                       self.graph_generator
                                                       )
                else: #'spatialattention' in self.train_params['loss_fn']['fns']
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       self.gallery, 
                                                       self.query_guided_attention,
                                                       query_guided=True,
                                                       knn='knn' in self.config['mode'].split('_'),
                                                       queryguided = 'queryguided' in self.config['mode'].split('_'),
                                                       dl_ev_gnn=self.dl_ev_gnn
                                                       )

                logger.info('Mean AP: {:4.1%}'.format(mAP))

                logger.info('CMC Scores{:>12}'.format('Market'))

                for k in (1, 5, 10):
                    logger.info('  top-{:<4}{:12.1%}'.format(k, top['Market'][k - 1]))

                if self.dataset_short in ['cuhk03-np', 'dukemtmc']:
                    setting = 'Market'
                else:
                    setting = self.dataset_short

                if self.write:
                    logger.info("write evaluation")
                    for t, rank in zip([top[setting][0], top[setting][4], top[setting][9], mAP], ['rank-1', 'rank-5', 'rank-10', 'mAP']):
                        self.writer.add_scalar('Accuracy/test/' + str(rank), t, e)

                scores.append((mAP, [top[setting][k - 1] for k in [1, 5, 10]]))
                rank = top[setting][0]

                self.encoder.current_epoch = e
                if rank > best_rank_iter:
                    best_rank_iter = rank
                    torch.save(self.encoder.state_dict(),
                               osp.join(self.save_folder_nets,
                                        self.fn + '.pth'))
                    if self.gnn is not None:
                        torch.save(self.gnn.state_dict(),
                               osp.join(self.save_folder_nets,
                                        'gnn_' + self.fn + '.pth'))
                    
                    elif self.query_guided_attention is not None:
                        torch.save(self.query_guided_attention.state_dict(),
                               osp.join(self.save_folder_nets,
                                        'gnn_' + self.fn + '.pth'))
                
        else:
            logger.info(
                'Loss {}, Accuracy {}'.format(torch.mean(loss.cpu()),
                                              self.running_corrects / self.denom))

            scores.append(self.running_corrects / self.denom)
            self.denom, self.running_corrects = 0, 0
            if scores[-1] > best_rank_iter:
                best_rank_iter = scores[-1]
                torch.save(self.encoder.state_dict(),
                           osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))

        return best_rank_iter

    def check_sampling_strategy(self, e):
        # Different Distance Sampling Strategies
        if self.distance_sampling != 'no':
            if e > 1:
                self.dl_tr = self.dl_tr2
                self.dl_ev = self.dl_ev2
                self.gallery = self.gallery2
                self.query = self.query2
                self.dl_ev_gnn = self.dl_ev_gnn2
            elif e == 1:
                self.dl_tr = self.dl_tr1
                self.dl_ev = self.dl_ev1
                self.gallery = self.gallery1
                self.query = self.query1
                self.dl_ev_gnn = self.dl_ev_gnn1

            # set feature dict of sampler = feat dict of previous epoch
            self.dl_tr.sampler.feature_dict = self.feature_dict
            self.dl_tr.sampler.epoch = e


    def get_loss_fn(self, params, num_classes, add_distractors=False):
        self.losses = defaultdict(list)
        self.losses_mean = defaultdict(list)
        self.every = self.model_params['gnn_params']['every']

        if add_distractors:
            self.bce_distractor = nn.BCELoss()
        else:
            self.bce_distractor = None

        # GNN loss
        if not self.every:  # normal GNN loss
            if 'gnn' in params['fns'].split('_'):
                self.gnn_loss = nn.CrossEntropyLoss().cuda(self.gnn_dev)
            elif 'lsgnn' in params['fns'].split('_') or 'gnnspatialattention' in params['fns'].split('_'):
                self.gnn_loss = losses.CrossEntropyLabelSmooth(
                    num_classes=num_classes, dev=self.gnn_dev).cuda(
                    self.gnn_dev)
            elif 'focalgnn' in params['fns'].split('_'):
                self.gnn_loss = losses.FocalLoss().cuda(self.gnn_dev)
            else:
                self.gnn_loss = None

        # GNN loss after every layer
        else:  
            if 'gnn' in params['fns'].split('_'):
                self.gnn_loss = [nn.CrossEntropyLoss().cuda(self.gnn_dev) for
                                 _ in range(
                        self.model_params['gnn_params']['gnn'][
                            'num_layers'])]
            elif 'lsgnn' in params['fns'].split('_'):
                self.gnn_loss = [losses.CrossEntropyLabelSmooth(
                    num_classes=num_classes, dev=self.gnn_dev).cuda(
                    self.gnn_dev) for
                                 _ in range(
                        self.model_params['gnn_params']['gnn'][
                            'num_layers'])]

        # Label smoothing for CrossEntropy Loss
        if 'lsce' in params['fns'].split('_'):
            self.ce = losses.CrossEntropyLabelSmooth(
                num_classes=num_classes, dev=self.gnn_dev).cuda(self.gnn_dev)
        elif 'focalce' in params['fns'].split('_'):
            self.ce = losses.FocalLoss().cuda(self.gnn_dev)
        elif 'ce' in params['fns'].split('_'):
            self.ce = nn.CrossEntropyLoss().cuda(self.gnn_dev)
        else:
            self.ce = None

        # Add MultiPositiveContrastive
        if 'multposcont' in params['fns'].split('_'):
            self.multposcont = losses.MultiPosCrossEntropyLoss().cuda(self.gnn_dev)
        else:
            self.multposcont = None

        # for distance computation
        if 'bceattention' in params['fns'].split('_'):
            self.bceattention = nn.BCELoss()
        else:
            self.bceattention = None
        
        # loss for query guided 
        if 'cespatialattention' in params['fns'].split('_'):
            self.multiattention = nn.CrossEntropyLoss().cuda(self.gnn_dev)
        elif 'lsspatialattention' in params['fns'].split('_'):
            self.multiattention = losses.CrossEntropyLabelSmooth(
                num_classes=num_classes, dev=self.gnn_dev).cuda(
                self.gnn_dev)
        else:
            self.multiattention = None

        # Add triplet loss
        if 'triplet' in params['fns'].split('_'):
            self.triplet = losses.TripletLoss(margin=0.3).cuda(self.gnn_dev)
        else:
            self.triplet = None

        # Add triplet loss after attention
        if 'tripletattention' in params['fns'].split('_'):
            self.tripletattention = losses.TripletLoss(margin=0.3).cuda(self.gnn_dev)
        else:
            self.tripletattention = None

    def get_save_name(self):
        if self.config['mode'] == 'pretraining':
            mode = 'finetuned_'
        else:
            mode = ''
        if self.model_params['encoder_params']['neck']:
            mode = mode + 'neck_'

        return mode

    def update_params(self):
        self.sample_hypers() if 'hyper' in self.config['mode'].split('_') else None

        if self.config['dataset']['val']:
            self.config['dataset']['num_classes'] -= 100

        if 'test' in self.config['mode'].split('_'):
            self.train_params['num_epochs'] = 1

        if self.config['dataset']['sampling'] != 'no':
            self.train_params['num_epochs'] += 10

    def sample_hypers(self):
        config = {'lr': 10 ** random.uniform(-8, -2),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'num_classes_iter': random.randint(6, 9),  # 100
                  'num_elements_class': random.randint(3, 4),
                  'temperatur': random.random(),
                  'num_epochs': 40}
        self.train_params.update(config)

        config = {'num_layers': random.randint(1, 4)}
        config = {'num_heads': random.choice([1, 2, 4, 8])}

        self.model_params['gnn_params']['gnn'].update(config)

        logger.info("Updated Hyperparameters:")
        logger.info(self.config)

    def get_data(self, config, train_params, mode):
        # If distance sampling
        self.distance_sampling = config['sampling']
        if config['sampling'] != 'no':
            seed = random.randint(0, 100)
            # seed = 19
            self.dl_tr2, self.dl_ev2, self.query2, self.gallery2, self.dl_ev_gnn2 = data_utility.create_loaders(
                data_root=config['dataset_path'],
                num_workers=config['nb_workers'],
                num_classes_iter=train_params['num_classes_iter'],
                num_elements_class=train_params['num_elements_class'],
                mode=mode,
                trans=config['trans'],
                distance_sampler=config['sampling'],
                val=config['val'],
                seed=seed,
                bssampling=self.config['dataset']['bssampling'])
            self.dl_tr1, self.dl_ev1, self.query1, self.gallery1, self.dl_ev_gnn1 = data_utility.create_loaders(
                data_root=config['dataset_path'],
                num_workers=config['nb_workers'],
                num_classes_iter=train_params['num_classes_iter'],
                num_elements_class=train_params['num_elements_class'],
                mode=mode,
                trans=config['trans'],
                distance_sampler='no',
                val=config['val'],
                seed=seed,
                bssampling=self.config['dataset']['bssampling'])
        # If testing or normal training
        else:
            self.dl_tr, self.dl_ev, self.query, self.gallery, self.dl_ev_gnn = data_utility.create_loaders(
                data_root=config['dataset_path'],
                num_workers=config['nb_workers'],
                num_classes_iter=train_params['num_classes_iter'],
                num_elements_class=train_params['num_elements_class'],
                mode=mode,
                trans=config['trans'],
                distance_sampler=config['sampling'],
                val=config['val'],
                bssampling=self.config['dataset']['bssampling'],
                rand_scales=config['rand_scales'],
                add_distractors=config['add_distractors'])

    def get_gnn(self, sz_embed):
        # pretrined gnn path
        path = self.model_params['gnn_params']['pretrained_path']
        if 'gnnspatialattention' in self.train_params['loss_fn']['fns']:
            if self.net_type == 'resnet50FPN':
                in_channels = 1024
            else:
                in_channels = sz_embed[0]

            self.gnn = net.SpatialGNNReIDTransformer(self.gnn_dev,
                                self.model_params['gnn_params'],
                                in_channels).cuda(self.gnn_dev)

            if path != "no":
                load_dict = torch.load(path, map_location='cpu')
                self.gnn.load_state_dict(load_dict)

            self.graph_generator = net.GraphGenerator(self.device,
                                                    **self.config[
                                                        'graph_params'])

            params = list(set(self.encoder.parameters())) + \
                list(set(self.gnn.parameters()))

        elif 'gnn' in self.train_params['loss_fn']['fns']:
            self.gnn = net.GNNReID(self.gnn_dev,
                                self.model_params['gnn_params'],
                                sz_embed).cuda(self.gnn_dev)

            if path != "no":
                load_dict = torch.load(path, map_location='cpu')
                self.gnn.load_state_dict(load_dict)

            self.graph_generator = net.GraphGenerator(self.device,
                                                    **self.config[
                                                        'graph_params'])

            params = list(set(self.encoder.parameters())) + \
                list(set(self.gnn.parameters()))

        elif 'spatialattention' in self.train_params['loss_fn']['fns']:
            if self.net_type == 'resnet50FPN':
                in_channels = 1024
            else:
                in_channels = sz_embed[0]

            self.query_guided_attention = net.Query_Guided_Attention_Layer(in_channels, \
                gnn_params=self.model_params['gnn_params']['gnn'],
                num_classes=self.config['dataset']['num_classes'],
                non_agg=True, class_non_agg=True).cuda(self.gnn_dev)

            if path != "no":
                load_dict = torch.load(path, map_location='cpu')
                self.query_guided_attention.load_state_dict(load_dict)

            params = list(set(self.encoder.parameters())) + \
                list(set(self.query_guided_attention.parameters()))
            self.gnn = None

        else:
            params = list(set(self.encoder.parameters()))
            self.gnn = None

        return params

    def milestones(self, e, train_params, logger):
        if e == 31:
            logger.info("reduce learning rate")
            self.encoder.load_state_dict(torch.load(
                osp.join(self.save_folder_nets, self.fn + '.pth')))
            if self.gnn is not None:
                self.gnn.load_state_dict(
                    torch.load(osp.join(self.save_folder_nets,
                                        'gnn_' + self.fn + '.pth')))
            elif self.query_guided_attention is not None:
                self.query_guided_attention.load_state_dict(
                    torch.load(osp.join(self.save_folder_nets,
                                        'gnn_' + self.fn + '.pth')))
            for g in self.opt.param_groups:
                g['lr'] = train_params['lr'] / 10.

        if e == 51:
            logger.info("reduce learning rate")
            if self.gnn is not None:
                self.gnn.load_state_dict(
                    torch.load(osp.join(self.save_folder_nets,
                                        'gnn_' + self.fn + '.pth')))
            elif self.query_guided_attention is not None:
                self.query_guided_attention.load_state_dict(
                    torch.load(osp.join(self.save_folder_nets,
                                        'gnn_' + self.fn + '.pth')))

            self.encoder.load_state_dict(torch.load(
                osp.join(self.save_folder_nets, self.fn + '.pth')))
            for g in self.opt.param_groups:
                g['lr'] = train_params['lr'] / 100.

    def log(self):
        [self.losses_mean[k].append(sum(v) / len(v)) for k, v in
             self.losses.items()]
        self.losses = defaultdict(list)
        logger.info('Loss Values: ')
        logger.info(', '.join([str(k) + ': ' + str(v[-1]) for k, v in
                                self.losses_mean.items()]))

    def get_dist(self, x, y, temp=-1):
        if temp > 0:
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)
            return torch.mm(x, y.t())/temp
        else:
            return torch.mm(x, y.t())