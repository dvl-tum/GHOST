import os.path as osp
import logging
import random

from numpy.core.fromnumeric import argmax
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
#from evaluation.utils import visualize_att_map
from torch.utils.tensorboard import SummaryWriter
autograd.set_detect_anomaly(True)

logger = logging.getLogger('GNNReID.Training')

torch.manual_seed(0)


class Trainer():
    def __init__(self, config, save_folder_nets, save_folder_results,
                 device, timer, write=True):
        
        self.make_det()
        self.update_configs = list()

        self.config = config
        if 'attention' in self.config['train_params']['loss_fn']['fns']:
            self.config['models']['attention'] = 1
            self.attention = 1
        else:
            self.config['models']['attention'] = 0
            self.attention = 0 

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
        logger.info("Writing and saving weights {}".format(write))

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

        if self.write and not 'hyper' in self.config['mode'].split('_'):
            pretrained = str(self.model_params['encoder_params']['pretrained_path'] != 'no')
            path = 'runs/' + "_".join([str(self.model_params['freeze_bb']), \
                str(self.model_params['gnn_params']['gnn']['sig']),\
                    str(self.model_params['gnn_params']['gnn']['inter_chan']), \
                        str(self.model_params['gnn_params']['gnn']['inter_chan_fin']), \
                            self.config['dataset']['corrupt'],
                            self.train_params['loss_fn']['fns'], pretrained, \
                                str(self.model_params['encoder_params']['last_stride']), 
                                self.train_params['loss_fn']['which_MPC'],
                                self.train_params['output_train_enc'],
                                self.train_params['output_train_gnn'],
                                self.config['eval_params']['output_test_enc'],
                                self.config['eval_params']['output_test_gnn'],
                                self.model_params['encoder_params']['pool'],
                                self.model_params['gnn_params']['gnn']['pool'],
                                str(self.model_params['gnn_params']['gnn']['transformer']),
                                self.config['mode'], str(self.model_params['encoder_params']['red']),
                                str(self.model_params['encoder_params']['neck'])])
            self.writer = SummaryWriter(path) # + str(i))
            logger.info("Writing to {}.".format(path))  

        logger.info('Search iteration {}'.format(i + 1))
        mode = self.get_save_name()
        self.update_params()

        if not 'hyper' in self.config['mode'].split('_'):
            self.make_det(random.randint(0, 100))

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
            self.corrupt = self.config['dataset']['corrupt'] != 'no'

            if 'all' in self.config['mode']:
                # get data
                self.get_data(self.config['dataset'], self.train_params,
                            self.config['mode'])

            # get models
            encoder, sz_embed = net.load_net(
                self.config['dataset']['dataset_short'],
                self.config['dataset']['num_classes'],
                self.config['mode'],
                self.model_params['attention'],
                add_distractors=self.config['dataset']['add_distractors'],
                **self.model_params['encoder_params'])
            self.encoder = encoder.cuda(self.device)  # to(self.device)
            param_groups = self.get_gnn(sz_embed)

            if 'all' not in self.config['mode']:
                # get data
                self.get_data(self.config['dataset'], self.train_params,
                            self.config['mode'])
            
            # get evaluator
            self.evaluator = Evaluator(**self.config['eval_params'])    

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

            # execute training
            best_rank_iter, model = self.execute(
                self.train_params,
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Rank-1: {}'.format(best_rank_iter))

            # save best model of iterations
            if best_rank_iter > best_rank and self.write and not 'hyper' in self.config['mode'].split('_'):
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
            
            if self.write and not 'hyper' in self.config['mode'].split('_'):
                self.writer.close()

        logger.info("Best Hyperparameters found: " + best_hypers)
        logger.info(
            "Achieved {} with this hyperparameters".format(best_rank))
        logger.info("-----------------------------------------------------\n")

    def execute(self, train_params, eval_params):
        best_rank_iter = 0
        scores = list()

        #best_rank_iter = self.evaluate(eval_params, scores, 0,
        #                        best_rank_iter)
        if 'test' in self.config['mode'].split('_'):
                best_rank_iter = self.evaluate(eval_params, scores, 0, 0)
        else:

            best_rank_iter = self.evaluate(eval_params, scores, 0,
                                best_rank_iter)

            for e in range(1, train_params['num_epochs'] + 1):
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
                
                best_rank_iter = self.evaluate(eval_params, scores, e,
                                best_rank_iter)

                self.log()
                # compute ranks and mAP at the end of each epoch
                if e > 5 and best_rank_iter < 0.1 and self.num_iter > 1:
                    break

        return best_rank_iter, self.encoder

    def forward_pass(self, x, Y, I, P, train_params, e):
        Y = Y.cuda(self.device)
        self.opt.zero_grad()
        # get sencond image to corrupt
        '''cost_mat = (torch.atleast_2d(Y) == torch.atleast_2d(Y).T).float() * 100000000
        from lapsolver import solve_dense
        _, corruption_img_ind = solve_dense(cost_mat.cpu().numpy())
        corruption_img_ind = torch.from_numpy(corruption_img_ind).to(Y.get_device()).long()'''
        corruption_img_ind = torch.stack([random.choice(torch.arange(x.shape[0])[Y[i] != Y]) for i in range(x.shape[0])])
        x_cor = x[corruption_img_ind]

        # randomly get position
        position = torch.tensor([random.randint(0, 1) for _ in range(x.shape[0])]).bool()
        imgs = list()
        
        # corrupt
        if self.corrupt:
            for i in range(x.shape[0]):
                if self.config['dataset']['corrupt'] == 'strong':
                    pos_a = [random.randint(-32, 32), random.randint(-16, 182)]
                    if pos_a[1] > 128:
                        pos_b = random.randint(-32, pos_a[1]-96)
                    else:
                        pos_b = random.randint(pos_a[1]+54, 182)

                    pos_b = [random.randint(min(pos_a[1] + 32, 40), 64), pos_b]
                    
                    corrupted = torch.zeros(3, x.shape[-2], x.shape[-1]*2)
                    if position[i]:
                        _x = x[i][:, -pos_a[0]:, :] if pos_a[0] < 0 else x[i][:, :x[i].shape[1]-pos_a[0], :]
                        _x = _x[:, :, -pos_a[1]:] if pos_a[1] < 0 else _x[:, :, :min(_x.shape[1], x[i].shape[-1]*2-pos_a[1])]
                        _xc = x_cor[i][:, -pos_b[0]:, :] if pos_b[0] < 0 else x_cor[i][:, :x_cor[i].shape[1]-pos_b[0], :]
                        _xc = _xc[:, :, -pos_b[1]:] if pos_b[1] < 0 else _xc[:, :, :min(_xc.shape[2], x_cor.shape[-1]*2-pos_b[1])]
                        
                        corrupted[:, max(0, pos_a[0]):max(0, pos_a[0])+_x.shape[1], max(0, pos_a[1]):max(0, pos_a[1])+_x.shape[2]] = _x
                        corrupted[:, max(0, pos_b[0]):max(0, pos_b[0])+_xc.shape[1], max(0, pos_b[1]):max(0, pos_b[1])+_xc.shape[2]] = _xc
                        
                    else:
                        _x = x_cor[i][:, -pos_a[0]:, :] if pos_a[0] < 0 else x_cor[i][:, :x_cor[i].shape[1]-pos_a[0], :]
                        _x = _x[:, :, -pos_a[1]:] if pos_a[1] < 0 else _x[:, :, :min(_x.shape[1], x_cor.shape[-1]*2-pos_a[1])]
                        _xc = x[i][:, -pos_b[0]:, :] if pos_b[0] < 0 else x[i][:, :x[i].shape[1]-pos_b[0], :]
                        _xc = _xc[:, :, -pos_b[1]:] if pos_b[1] < 0 else _xc[:, :, :min(_xc.shape[2], x[i].shape[-1]*2-pos_b[1])]
                        
                        corrupted[:, max(0, pos_a[0]):max(0, pos_a[0])+_x.shape[1], max(0, pos_a[1]):max(0, pos_a[1])+_x.shape[2]] = _x
                        corrupted[:, max(0, pos_b[0]):max(0, pos_b[0])+_xc.shape[1], max(0, pos_b[1]):max(0, pos_b[1])+_xc.shape[2]] = _xc
                elif self.config['dataset']['corrupt'] == 'randadd':
                    if position[i]:
                        corrupted = torch.cat([x[i], x_cor[i]], dim=-1)
                    else:
                        corrupted = torch.cat([x_cor[i], x[i]], dim=-1)  
                elif self.config['dataset']['corrupt'] == 'add':
                    corrupted = torch.cat([x[i], x_cor[i]], dim=-1)
                
                imgs.append(corrupted)    

            x_cor = torch.stack(imgs)

        # bb for spatial attention (same bb but different output)
        if self.model_params['attention']:
            probs, fc7, attention_features = self.encoder(x.cuda(self.device),
                                output_option=train_params['output_train_enc'])
            
            if self.corrupt:
                # if corruption
                _, _, attention_features_cor = self.encoder(x_cor.cuda(self.device),
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
        
        self.losses['Classification Accuracy'].append((torch.argmax(\
            probs, dim=1)==Y).float().mean())

        # query guided attention output
        if self.model_params['attention'] and not self.gnn:
            if not self.corrupt:
                _, _, dist, qs, gs, attended_feats, fc_x, x2, fc_x2, att_maps = \
                    self.query_guided_attention(attention_features, \
                        output_option=train_params['output_train_gnn'])
            else:
                _, _, dist, qs, gs, attended_feats, fc_x, x2, fc_x2, att_maps = \
                    self.query_guided_attention(attention_features, attention_features_cor, \
                        output_option=train_params['output_train_gnn'])
        
        if self.config['visualize']:
            att_g = copy.deepcopy(att_maps.detach().cpu())
            att_g = att_g.view(x.shape[0], -1, att_g.shape[-2], att_g.shape[-1])
            for i in range(x.shape[0]):
                visualize_att_map(att_g[i], x_cor[gs[qs==i]], P[i], \
                    [p for j, p in enumerate(P) if j in gs[qs==i]], \
                        [P[cp] for j, cp in enumerate(corruption_img_ind) if j in gs[qs==i]], \
                            save_dir='train_samples_2', sig=self.model_params['gnn_params']['gnn']['sig'])
            quit()

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
            bce_lab = torch.zeros(Y.shape).to(self.device)
            bce_lab[Y!=-2] = 1
            distrloss = self.bce_distractor(distractor_bce.squeeze(), bce_lab.squeeze())
            loss += train_params['loss_fn']['scaling_bce'] * distrloss
            self.losses['BCE Loss Distractors ' + str(i)].append(distrloss.item())

        if self.tripletattention:
            triploss, _ = self.tripletattention(label=Y[gs], label_cor=Y[corruption_img_ind[gs]], \
                label_2=Y, att_feats=attended_feats, fc7=fc7, qs=qs, gs=gs, \
                    corruption_img_ind=corruption_img_ind, ind=I)
            loss += train_params['loss_fn']['scaling_triplet'] * triploss
            self.losses['Triplet'].append(triploss.item())

        if self.bceattention:
            ql = Y[qs]
            gl = Y[gs]

            if not self.corrupt:
                same_class = (ql == gl).float().cuda(self.device)
            else:
                gl_cor = Y[corruption_img_ind[gs]]
                same_class = ((ql == gl)|(ql == gl_cor)).float().cuda(self.device)
            
            bceattloss = self.bceattention(dist.squeeze(), same_class.squeeze())
            loss += train_params['loss_fn']['scaling_bce'] * bceattloss
            self.losses['BCE Loss Atttention ' + str(i)].append(bceattloss.item())

        if self.l1regatt:
            neg = torch.logical_and(torch.logical_and(Y[gs] != Y[qs].T , Y[corruption_img_ind[gs]] != Y[qs].T), Y[qs] == Y[qs].T)
            neg_samps = att_maps[neg]
            l1_reg_att = torch.norm(neg_samps, 1)
            scale = train_params['loss_fn']['scaling_l1regatt']
            loss += scale * l1_reg_att

            self.losses['L1 Reg Atttention Maps Neg samples ' + str(i)].append(scale*l1_reg_att.item())
        
        if self.multiattention:
            for i, fc in enumerate([fc_x]): #, fc_x2
                # only take samples that were attended by positive queries, query label == gallery label
                if not self.corrupt:
                    mask = Y[gs] == Y[qs]
                else:
                    mask = (Y[gs] == Y[qs]) | (Y[corruption_img_ind[gs]] == Y[qs])

                multattloss = self.multiattention(fc[mask]/self.train_params['temperatur'], Y[gs][mask])
                # multattloss = self.multiattention(fc_x/self.train_params['temperatur'], Y[gs])
                scale = train_params['loss_fn']['scaling_multiattention']
                loss += scale * multattloss
                self.losses['CE Loss Atttention ' + str(i)].append(scale * multattloss.item())
                self.losses['Classification Accuracy after QG'].append((torch.argmax(\
                    fc_x[mask], dim=1)==Y[gs][mask]).float().mean())
        
        # Compute MultiPositiveContrastive
        if self.multposcont:
            for i, att_feats in enumerate([attended_feats]): #, x2
                if self.train_params['loss_fn']['which_MPC'] == 'MPCQG':
                    # only samples after spatial attention for pos con
                    dist = self.get_dist(att_feats, att_feats)
                    if not self.corrupt:
                        loss0 = self.multposcont(dist, Y[gs], which=self.config['train_params'\
                            ]['loss_fn']['which_MPC'])
                    else:
                        loss0 = self.multposcont(dist, Y[gs], label_corr=Y[corruption_img_ind[gs]\
                            ], label_2=Y[qs], which=self.config['train_params']['loss_fn']['which_MPC'])
                
                elif self.train_params['loss_fn']['which_MPC'] == 'MPCFC7':
                    # use features after bb as pos and neg samples
                    dist = self.get_dist(att_feats, fc7)
                    if not self.corrupt:
                        loss0 = self.multposcont(dist, Y[gs], Y, which=self.config['train_params'\
                            ]['loss_fn']['which_MPC'])
                    else:
                        loss0 = self.multposcont(dist, Y[gs], label_2=Y, \
                            label_corr=Y[corruption_img_ind[gs]], which=self.config['train_params'\
                                ]['loss_fn']['which_MPC'])
                
                elif self.train_params['loss_fn']['which_MPC'] == 'MPCFC7QG':
                    # use features after bb as pos and neg samples
                    dist1 = self.get_dist(att_feats, fc7)
                    dist2 = self.get_dist(att_feats, att_feats)
                    if not self.corrupt:
                        loss0 = self.multposcont(dist, _Y, Y, which=self.config['train_params'\
                            ]['loss_fn']['which_MPC'])
                    else:
                        loss0 = self.multposcont(dist1, Y[gs], label_2=Y[qs], \
                            label_corr=Y[corruption_img_ind[gs]], which=self.config['train_params'\
                                ]['loss_fn']['which_MPC'], dist_2=dist2, qs=qs, gs=gs)

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
            if self.bce_distractor_gnn:
                pred, feats, pred_person = self.gnn(fc7, edge_index, edge_attr,
                                            train_params['output_train_gnn'],
                                            mode='train')
            else:
                pred, feats = self.gnn(fc7, edge_index, edge_attr,
                                        train_params['output_train_gnn'],
                                        mode='train')
            pred = [p[Y!=-2] for p in pred]
            _Y = Y[Y!=-2]
            self.losses['Classification Accuracy after GNN'].append((torch.argmax(\
                pred[-1], dim=1)==_Y).float().mean())

            if self.every:
                loss1 = [gnn_loss(
                    pr / self.train_params['temperatur'],
                    _Y.cuda(self.gnn_dev)) for gnn_loss, pr in
                            zip(self.gnn_loss, pred)]
            else:
                loss1 = [self.gnn_loss(
                    pred[-1] / self.train_params[
                        'temperatur'], _Y.cuda(self.gnn_dev))]

            loss += sum([train_params['loss_fn']['scaling_gnn'] * l
                            for l in loss1])

            [self.losses['GNN' + str(i)].append(l.item()) for i, l in
                enumerate(loss1)]

            if self.bce_distractor_gnn:
                # -2 again is distractor label
                bce_lab = torch.zeros(Y.shape).to(self.device)
                bce_lab[Y!=-2] = 1
                distrloss = self.bce_distractor_gnn(pred_person.squeeze(), bce_lab.squeeze())
                loss += train_params['loss_fn']['scaling_bce'] * distrloss
                self.losses['BCE Loss Distractors GNN ' + str(i)].append(distrloss.item())


        # Compute Triplet Loss
        if self.triplet:
            _fc7 = fc7[Y!=-2]
            _Y = Y[Y!=-2]
            triploss, _ = self.triplet(_fc7, _Y)
            loss += train_params['loss_fn']['scaling_triplet'] * triploss
            self.losses['Triplet'].append(triploss.item())

        if self.write and not 'hyper' in self.config['mode'].split('_'):
            for k, v in self.losses.items():
                self.writer.add_scalar('Loss/train/' + k, v[-1], e)
            
        return loss

    def evaluate(self, eval_params, scores=None, e=None, best_rank_iter=None, mode='val'):
        if mode == 'val':
            with torch.no_grad():
                logger.info('EVALUATION')
                if self.config['mode'] in ['train', 'test', 'hyper_search', 'all', 'all_test']:
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       gallery=self.gallery,
                                                       add_dist= 
                                                       self.config['dataset'][
                                                            'add_distractors'],
                                                       attention=self.attention
                                                       )
                elif self.gnn is not None:
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       self.gallery, self.gnn,
                                                       self.graph_generator,
                                                       attention=self.attention
                                                       )
                else: #'spatialattention' in self.train_params['loss_fn']['fns']
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       self.gallery, 
                                                       self.query_guided_attention,
                                                       query_guided=True,
                                                       queryguided = 'queryguided' in self.config['mode'].split('_'),
                                                       dl_ev_gnn=self.dl_ev_gnn,
                                                       attention=self.attention,
                                                       visualize=self.config['visualize']
                                                       )

                logger.info('Mean AP: {:4.1%}'.format(mAP))

                logger.info('CMC Scores{:>12}'.format('Market'))

                for k in (1, 5, 10):
                    logger.info('  top-{:<4}{:12.1%}'.format(k, top['Market'][k - 1]))

                setting = 'Market'

                if self.write and not 'hyper' in self.config['mode'].split('_'):
                    logger.info("write evaluation")
                    for t, rank in zip([top[setting][0], top[setting][4], top[setting][9], mAP], ['rank-1', 'rank-5', 'rank-10', 'mAP']):
                        self.writer.add_scalar('Accuracy/test/' + str(rank), t, e)

                scores.append((mAP, [top[setting][k - 1] for k in [1, 5, 10]]))
                rank = top[setting][0]
                
                self.encoder.current_epoch = e
                if self.train_params['store_every'] or rank > best_rank_iter:
                    best_rank_iter = rank
                    if self.write and not 'hyper' in self.config['mode'].split('_'):
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
            
            if add_distractors:
                self.bce_distractor_gnn = nn.BCELoss()
            else:
                self.bce_distractor_gnn = None

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
            self.bceattention = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(270/36))
        else:
            self.bceattention = None
        
        if 'l1regatt' in params['fns'].split('_'):
            self.l1regatt = True
        else:
            self.l1regatt = False
        
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
            self.tripletattention = losses.TripletLossAtt(margin=0.3).cuda(self.gnn_dev)
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
        if not len(self.update_configs):
            for i in range(self.num_iter):
                config = dict()
                config['train'] = {'lr': 10 ** random.uniform(-8, -2),
                        'weight_decay': 10 ** random.uniform(-15, -6),
                        'num_classes_iter': random.randint(6, 9),  # 100
                        'num_elements_class': random.randint(3, 4),
                        'temperatur': random.random(),
                        'num_epochs': 30}
                config['gnn'] = {'num_layers': random.randint(1, 4),
                          'num_heads': random.choice([1, 2, 4, 8])}
                self.update_configs.append(config)
            self.iter_i = 0
        
        self.train_params.update(self.update_configs[self.iter_i]['train'])
        self.model_params['gnn_params']['gnn'].update(self.update_configs[\
            self.iter_i]['gnn'])

        logger.info("Updated Hyperparameters:")
        logger.info(self.config)
        self.iter_i += 1

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
                add_distractors=config['add_distractors'],
                split=config['split'],
                sz_crop=config['sz_crop'])
            
        self.config['dataset']['num_classes'] = len(set(self.dl_tr.dataset.ys)) - 1
        self.model_params['gnn_params']['classifier']['num_classes'] = len(set(self.dl_tr.dataset.ys)) - 1


    def get_gnn(self, sz_embed):
        # pretrined gnn path
        self.query_guided_attention = None
        self.gnn = None

        path = self.model_params['gnn_params']['pretrained_path']
        if 'gnnspatialattention' in self.train_params['loss_fn']['fns']:
            if self.net_type == 'resnet50FPN':
                in_channels = 1024
            else:
                in_channels = sz_embed[0]

            # SpatialGNNReIDTransformer
            self.gnn = net.SpatialGNNReID(self.gnn_dev, 
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
                                sz_embed, 
                                add_distractors=self.config['dataset']['add_distractors']).cuda(self.gnn_dev)

            if path != "no":
                load_dict = torch.load(path, map_location='cpu')
                self.gnn.load_state_dict(load_dict)

            self.graph_generator = net.GraphGenerator(self.device,
                                                    **self.config[
                                                        'graph_params'])

            params = list(set(self.encoder.parameters())) + \
                list(set(self.gnn.parameters()))

        elif 'spatialattention' in self.train_params['loss_fn']['fns'] or 'queryguided' in self.config['mode']:
            if self.net_type == 'resnet50FPN':
                in_channels = 1024
            else:
                in_channels = sz_embed[0]

            self.query_guided_attention = net.Query_Guided_Attention_Layer(in_channels, \
                gnn_params=self.model_params['gnn_params']['gnn'],
                num_classes=self.config['dataset']['num_classes'],
                non_agg=True, class_non_agg=True, 
                neck=self.model_params['gnn_params']['classifier']['neck'],
                last_stride=self.model_params['encoder_params']['last_stride'],
                sig=self.model_params['gnn_params']['gnn']['sig'],
                inter_channels=self.model_params['gnn_params']['gnn']['inter_chan'],
                corrupt=self.corrupt,
                inter_chan_fin=self.model_params['gnn_params']['gnn']['inter_chan_fin'],
                pool=self.model_params['gnn_params']['gnn']['pool'],
                transformer=self.model_params['gnn_params']['gnn']['transformer']).cuda(self.gnn_dev)

            self.query_guided_attention.apply(net.utils.weights_init_kaiming)

            if path != "no":
                load_dict = torch.load(path, map_location='cpu')
                load_dict = {k: v for k, v in load_dict.items() if 'fc_beforeRes' not in k}
                state_dict = self.query_guided_attention.state_dict()
                state_dict.update(load_dict)
                self.query_guided_attention.load_state_dict(state_dict)

            params = list(set(self.query_guided_attention.parameters()))
        
        else:
            self.gnn = None

        if not self.model_params['freeze_bb'] and self.gnn:
            params += list(set(self.encoder.parameters()))
        elif not self.gnn:
            params = list(set(self.encoder.parameters()))
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # get optimizer
        param_groups = [{'params': params,
                            'lr': self.train_params['lr']}]
        
        return param_groups

    def milestones(self, e, train_params, logger):
        if e in train_params['milestones']:
            logger.info("reduce learning rate")
            if self.write:
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
            else:
                logger.info("not loading weights as self.write = False")

            for g in self.opt.param_groups:
                g['lr'] = g['lr'] / 10.

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


import cv2
def visualize_att_map(att_g, x_cor, q_path, g_paths, corr_paths, save_dir='visualization_attention_maps', sig=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import PIL.Image as Image
    query = cv2.imread(q_path, 1)
    min_att = att_g.min().cpu().numpy()
    max_att = att_g.max().cpu().numpy()
    for g, gc, gallery, attention_map in zip(g_paths, corr_paths, x_cor, att_g):
        gallery = torch.stack([(img * std[i]) + mean[i] for i, img in enumerate(gallery)])
        gallery = np.transpose(gallery.cpu().numpy(), (1, 2, 0))[:,:,::-1]
        #print(attention_map)
        attention_map = cv2.resize(attention_map.squeeze().cpu().numpy(), (gallery.shape[1], gallery.shape[0]))
        if not sig:
            attention_map = (attention_map-min_att)/(max_att-min_att)

        cam = show_cam_on_image(gallery, attention_map)        
        
        fig = figure(figsize=(10, 6), dpi=80)
        # Create figure and axes
        fig.add_subplot(1,3,1)
        plt.imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        fig.add_subplot(1,3,2)
        plt.imshow(cv2.cvtColor(gallery, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        fig.add_subplot(1,3,3)
        plt.imshow(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(os.path.join(save_dir, \
            os.path.basename(q_path)[:-4] + '_' + os.path.basename(g)[:-4] + '_' + os.path.basename(gc)[:-4] + '.png'))


def show_cam_on_image(img, mask):
    img = np.float32(img) #/ 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
