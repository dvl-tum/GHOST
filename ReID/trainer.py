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
from evaluation import Evaluator, Evaluator_DML
import utils.utils as utils
import matplotlib.pyplot as plt
import os
import json
from torch import autograd

autograd.set_detect_anomaly(True)

logger = logging.getLogger('GNNReID.Training')

torch.manual_seed(0)


class Trainer():
    def __init__(self, config, save_folder_nets, save_folder_results,
                 device, timer):
        self.config = config
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

    def train(self):
        best_rank = 0

        for i in range(self.num_iter):
            self.nb_clusters = 900
            logger.info('Search iteration {}'.format(i + 1))
            mode = self.get_save_name()
            self.update_params()
            logger.info(self.config)
            logger.info(self.timer)

            encoder, sz_embed = net.load_net(
                self.config['dataset']['dataset_short'],
                self.config['dataset']['num_classes'],
                self.config['mode'],
                **self.config['models']['encoder_params'])

            self.device = 0
            self.encoder = encoder.cuda(self.device)  # to(self.device)
            if torch.cuda.device_count() > 1:
                self.gnn_dev = 1
            else:
                self.gnn_dev = 0
            print(self.device, self.gnn_dev)
            # 1 == dev
            self.gnn = net.GNNReID(self.gnn_dev,
                                   self.config['models']['gnn_params'],
                                   sz_embed).cuda(self.gnn_dev)

            if self.config['models']['gnn_params']['pretrained_path'] != "no":
                load_dict = torch.load(
                    self.config['models']['gnn_params']['pretrained_path'],
                    map_location='cpu')
                self.gnn.load_state_dict(load_dict)

            self.graph_generator = net.GraphGenerator(self.device,
                                                      **self.config[
                                                          'graph_params'])

            self.evaluator = Evaluator(**self.config['eval_params'])

            params = list(set(self.encoder.parameters())) + list(
                set(self.gnn.parameters()))

            param_groups = [{'params': params,
                             'lr': self.config['train_params']['lr']}]

            self.opt = RAdam(param_groups,
                             weight_decay=self.config['train_params'][
                                 'weight_decay'])

            self.get_loss_fn(self.config['train_params']['loss_fn'],
                             self.config['dataset']['num_classes'])

            # Do training in mixed precision
            if self.config['train_params']['is_apex']:
                [self.encoder, self.gnn], self.opt = amp.initialize(
                    [self.encoder, self.gnn], self.opt,
                    opt_level="O1")
            if torch.cuda.device_count() > 1:
                self.encoder = nn.DataParallel(self.encoder)

            self.get_data(self.config['dataset'], self.config['train_params'],
                          self.config['mode'])

            best_rank_iter, model = self.execute(
                self.config['train_params'],
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Rank-1: {}'.format(best_rank_iter))

            if best_rank_iter > best_rank and \
                    not 'test' in self.config['mode'].split('_'):
                os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
                          str(best_rank_iter) + mode + self.net_type + '_' +
                          self.dataset_short + '.pth')
                os.rename(
                    osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
                    str(best_rank_iter) + 'gnn_' + mode + self.net_type + '_' +
                    self.dataset_short + '.pth')
                best_rank = best_rank_iter
                best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])
            elif 'test' in self.config['mode'].split('_'):
                best_rank = best_rank_iter
                best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])

        logger.info("Best Hyperparameters found: " + best_hypers)
        logger.info(
            "Achieved {} with this hyperparameters".format(best_rank))
        logger.info("-----------------------------------------------------\n")

    def execute(self, train_params, eval_params):
        since = time.time()
        best_rank_iter = 0
        scores = list()
        self.preds, self.feats, self.labs, self.preds_before, self.feats_before = dict(), dict(), dict(), dict(), dict()
        self.best_preds, self.best_labs, self.best_feats, self.best_preds_before, self.best_feats_before = dict(), dict(), dict(), dict(), dict()

        # feature dict for distance sampling
        if self.distance_sampling != 'no':
            self.feature_dict = defaultdict()

        for e in range(1, train_params['num_epochs'] + 1):
            if 'test' in self.config['mode'].split('_'):
                best_rank_iter = self.evaluate(eval_params, scores, 0, 0)
            # If not testing
            else:
                logger.info(
                    'Epoch {}/{}'.format(e, train_params['num_epochs']))
                if e == 31:
                    logger.info("reduce learning rate")
                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    self.gnn.load_state_dict(
                        torch.load(osp.join(self.save_folder_nets,
                                            'gnn_' + self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                if e == 51:
                    logger.info("reduce learning rate")
                    self.gnn.load_state_dict(
                        torch.load(osp.join(self.save_folder_nets,
                                            'gnn_' + self.fn + '.pth')))

                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                self.check_sampling_strategy(e)

                # If distance_sampling == only, use first epoch to get features
                if (self.distance_sampling != 'no' and e == 1):
                    model_is_training = self.encoder.training
                    gnn_is_training = self.gnn.training
                    self.encoder.eval()
                    self.gnn.eval()
                    with torch.no_grad():
                        for x, Y, I, P in self.dl_tr:
                            Y = Y.cuda(self.device)
                            probs, fc7, student_feats = self.encoder(
                                x.cuda(self.device),
                                train_params['output_train_enc'])
                            if self.gnn_loss or self.of:
                                edge_attr, edge_index, fc7 = self.graph_generator.get_graph(
                                    fc7)
                                fc7 = fc7.cuda(self.gnn_dev)
                                edge_attr = edge_attr.cuda(self.gnn_dev)
                                edge_index = edge_index.cuda(self.gnn_dev)

                                pred, feats, _ = self.gnn(fc7, edge_index,
                                                          edge_attr,
                                                          train_params[
                                                              'output_train_gnn'])
                                features = feats[0]
                            else:
                                features = fc7

                            for y, f, i in zip(Y, features, I):
                                self.feature_dict[y.data.item()][i.item()] = f

                    self.encoder.train(model_is_training)
                    self.gnn.train(gnn_is_training)

                # Normal training with backpropagation
                else:
                    self.dl_tr.sampler.epoch = e
                    for x, Y, I, P in self.dl_tr:
                        loss = self.forward_pass(x, Y, I, P, train_params)

                        # Check possible net divergence
                        if torch.isnan(loss):
                            logger.error("We have NaN numbers, closing\n\n\n")
                            return 0.0, self.encoder

                        if train_params['is_apex']:
                            with amp.scale_loss(loss, self.opt) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        self.opt.step()

                        if self.center:
                            for param in self.center.parameters():
                                param.grad.data *= (
                                        1. /
                                        self.config['train_params']['loss_fn'][
                                            'scaling_center'])
                            self.opt_center.step()

                best_rank_iter = self.evaluate(eval_params, scores, e,
                                               best_rank_iter)

            [self.losses_mean[k].append(sum(v) / len(v)) for k, v in
             self.losses.items()]
            self.losses = defaultdict(list)
            logger.info('Loss Values: ')
            logger.info(', '.join([str(k) + ': ' + str(v[-1]) for k, v in
                                   self.losses_mean.items()]))
            # compute ranks and mAP at the end of each epoch

        end = time.time()

        self.save_results(train_params, since, end, best_rank_iter, scores)

        return best_rank_iter, self.encoder

    def forward_pass(self, x, Y, I, P, train_params):
        Y = Y.cuda(self.device)
        self.opt.zero_grad()
        if self.center:
            self.opt_center.zero_grad()
        probs, fc7 = self.encoder(x.cuda(self.device),
                                output_option=train_params['output_train_enc'])

        # Add feature vectors to dict if distance sampling
        if self.distance_sampling != 'no':
            for y, f, i in zip(Y, fc7, I):
                self.feature_dict[y.data.item()][i.item()] = f.detach()

        # Compute CE Loss
        loss = 0
        if self.ce:
            probs = probs.cuda(self.gnn_dev)
            Y = Y.cuda(self.gnn_dev)
            loss0 = self.ce(probs/self.config['train_params']['temperatur'], Y)
            loss += train_params['loss_fn']['scaling_ce'] * loss0
            self.losses['Cross Entropy'].append(loss.item())

        # Add other losses of not pretraining
        if not self.config['mode'] == 'pretraining':

            if self.gnn_loss or self.of:
                data = self.graph_generator.get_graph(fc7, Y)
                edge_attr = data[0].cuda(self.gnn_dev)
                edge_index = data[1].cuda(self.gnn_dev)
                fc7 = data[2].cuda(self.gnn_dev)

                if type(loss) != int:
                    loss = loss.cuda(self.gnn_dev)
                pred, feats, Y = self.gnn(fc7, edge_index, edge_attr, Y,
                                          train_params['output_train_gnn'],
                                          mode='train')

                for path, pre, f, pre_b, f_b, lab in zip(P, pred[-1],
                                                         feats[-1], fc7,
                                                         probs, Y):
                    self.preds[path] = pre.detach()
                    self.feats[path] = f.detach()
                    self.preds_before[path] = pre_b.detach()
                    self.feats_before[path] = f_b.detach()
                    self.labs[path] = lab

                if self.gnn_loss:
                    if self.every:
                        loss1 = [gnn_loss(
                            pr / self.config['train_params']['temperatur'],
                            Y.cuda(self.gnn_dev)) for gnn_loss, pr in
                                 zip(self.gnn_loss, pred)]
                    else:
                        loss1 = [self.gnn_loss(
                            pred[-1] / self.config['train_params'][
                                'temperatur'], Y.cuda(self.gnn_dev))]

                    loss += sum([train_params['loss_fn']['scaling_gnn'] * l
                                 for l in loss1])

                    [self.losses['GNN' + str(i)].append(l.item()) for i, l in
                     enumerate(loss1)]

            if self.distance_sampling != 'no':
                features = feats[-1] if self.gnn_loss or self.of else fc7
                for y, f, i in zip(Y, features, I):
                    self.feature_dict[y.data.item()][i.item()] = f.detach()

            # Compute center loss
            if self.center:
                loss2 = self.center(feats[-1], Y)
                loss += train_params['loss_fn']['scaling_center'] * loss2
                self.losses['Center'].append(loss2.item())

            # Compute Triplet Loss
            if self.triplet:
                triploss, _ = self.triplet(fc7, Y)
                loss += train_params['loss_fn']['scaling_triplet'] * triploss
                self.losses['Triplet'].append(triploss.item())

            # Compute MSE regularization
            if self.of:
                p = feats[0].detach().cuda(self.gnn_dev)
                of_reg = self.of(fc7, p)
                loss += train_params['loss_fn']['scaling_of'] * of_reg
                self.losses['OF'].append(of_reg.item())

            # Compute CE loss with soft targets = predictions of gnn
            if self.distill:
                target = torch.stack([self.soft_targets[p] for p in P]).cuda(
                    self.gnn_dev)
                distill = self.distill(
                    probs / self.config['train_params']['loss_fn'][
                        'soft_temp'], target)
                loss += train_params['loss_fn']['scaling_distill'] * distill
                self.losses['Distillation'].append(distill.item())

            # compute MSE loss with feature vectors from gnn
            if self.of_pre:
                target = torch.stack(
                    [torch.tensor(self.feat_targets[p]) for p in P]).cuda(
                    self.gnn_dev)
                of_pre = self.of_pre(fc7, target)
                loss += train_params['loss_fn']['scaling_of_pre'] * of_pre
                self.losses['OF Pretrained'].append(of_pre.item())

            # relational loss
            if self.distance:
                teacher = torch.stack(
                    [torch.tensor(self.feat_targets[p]) for p in P]).cuda(
                    self.gnn_dev)
                dist = self.distance(teacher, fc7)
                loss += train_params['loss_fn']['scaling_distance'] * dist
                self.losses['Distance'].append(dist.item())

            self.losses['Total Loss'].append(loss.item())

        else:
            # For pretraining just use acc as evaluation
            _, preds = torch.max(probs, 1)
            self.denom += Y.shape[0]
            self.running_corrects += torch.sum(
                preds == Y.data).cpu().data.item()

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
                                                       net_type=self.net_type,
                                                       dataroot=
                                                       self.config['dataset'][
                                                           'dataset_short'],
                                                       nb_classes=
                                                       self.config['dataset'][
                                                           'num_classes'])
                else:
                    mAP, top = self.evaluator.evaluate(self.encoder,
                                                       self.dl_ev,
                                                       self.query,
                                                       self.gallery, self.gnn,
                                                       self.graph_generator,
                                                       dl_ev_gnn=self.dl_ev_gnn,
                                                       net_type=self.net_type,
                                                       dataroot=
                                                       self.config['dataset'][
                                                           'dataset_short'],
                                                       nb_classes=
                                                       self.config['dataset'][
                                                           'num_classes'])

                logger.info('Mean AP: {:4.1%}'.format(mAP))

                logger.info('CMC Scores{:>12}{:>12}{:>12}'
                            .format('allshots', 'cuhk03', 'Market'))

                for k in (1, 5, 10):
                    logger.info('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                                .format(k, top['allshots'][k - 1],
                                        top['cuhk03'][k - 1],
                                        top['Market'][k - 1]))

                if self.dataset_short in ['cuhk03-np', 'dukemtmc']:
                    setting = 'Market'
                else:
                    setting = self.dataset_short

                scores.append((mAP, [top[setting][k - 1] for k in [1, 5, 10]]))
                rank = top[setting][0]

                self.encoder.current_epoch = e
                if rank > best_rank_iter:
                    best_rank_iter = rank
                    self.best_preds = self.preds
                    self.best_labs = self.labs
                    self.best_feats = self.feats
                    self.best_preds_before = self.preds_before
                    self.best_feats_before = self.feats_before
                    torch.save(self.encoder.state_dict(),
                               osp.join(self.save_folder_nets,
                                        self.fn + '.pth'))
                    torch.save(self.gnn.state_dict(),
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

    def save_results(self, train_params, since, end, best_rank_iter, scores):
        logger.info(
            'Completed {} epochs in {}s on {}'.format(
                train_params['num_epochs'],
                end - since,
                self.dataset_short))

        file_name = str(
            best_rank_iter) + '_' + self.dataset_short + '_' + str(self.timer)

        if 'test' in self.config['mode'].split('_'):
            file_name = 'test_' + file_name

        results_dir = osp.join('results', file_name)
        utils.make_dir(results_dir)

        if not self.config['mode'] == 'pretraining':
            with open(
                    osp.join(results_dir, file_name + '.txt'),
                    'w') as fp:
                fp.write(file_name + "\n")
                fp.write(str(self.config))
                fp.write('\n')
                fp.write('\n'.join('%s %s' % x for x in scores))
                fp.write("\n\n\n")

        # plot losses
        if not 'test' in self.config['mode'].split('_'):
            for k, v in self.losses_mean.items():
                eps = list(range(len(v)))
                plt.plot(eps, v)
                plt.xlim(left=0)
                plt.ylim(bottom=0, top=14)
                plt.xlabel('Epochs')
            plt.legend(self.losses_mean.keys(), loc='upper right')
            plt.grid(True)
            plt.savefig(osp.join(results_dir, k + '.png'))
            plt.close()

            # plot scores
            scores = [[score[0], score[1][0], score[1][1], score[1][2]] for score
                      in scores]
            for i, name in enumerate(['mAP', 'rank-1', 'rank-5', 'rank-10']):
                sc = [s[i] for s in scores]
                eps = list(range(len(sc)))
                plt.plot(eps, sc)
                plt.xlim(left=0)
                plt.ylim(bottom=0)
                plt.xlabel('Epochs')
                plt.ylabel(name)
                plt.grid(True)
                plt.savefig(osp.join(results_dir, name + '.png'))
                plt.close()

        with open(osp.join(results_dir, "preds.json"), "w") as f:
            save = {k: v.tolist() for k, v in self.best_preds.items()}
            json.dump(save, f)

        with open(osp.join(results_dir, "labs.json"), "w") as f:
            save = {k: v.tolist() for k, v in self.best_labs.items()}
            json.dump(save, f)

        with open(osp.join(results_dir, "feats.json"), "w") as f:
            save = {k: v.tolist() for k, v in self.best_feats.items()}
            json.dump(save, f)

        with open(osp.join(results_dir, "preds_before.json"), "w") as f:
            save = {k: v.tolist() for k, v in self.best_preds_before.items()}
            json.dump(save, f)

        with open(osp.join(results_dir, "feats_before.json"), "w") as f:
            save = {k: v.tolist() for k, v in self.best_feats_before.items()}
            json.dump(save, f)

    def get_loss_fn(self, params, num_classes):
        self.losses = defaultdict(list)
        self.losses_mean = defaultdict(list)
        self.every = self.config['models']['gnn_params']['every']

        # GNN loss
        if not self.every:  # normal GNN loss
            if 'gnn' in params['fns'].split('_'):
                self.gnn_loss = nn.CrossEntropyLoss().cuda(self.gnn_dev)
            elif 'lsgnn' in params['fns'].split('_'):
                self.gnn_loss = losses.CrossEntropyLabelSmooth(
                    num_classes=num_classes, dev=self.gnn_dev).cuda(
                    self.gnn_dev)
            elif 'focalgnn' in params['fns'].split('_'):
                self.gnn_loss = losses.FocalLoss().cuda(self.gnn_dev)
            else:
                self.gnn_loss = None

        else:  # GNN loss after every layer
            if 'gnn' in params['fns'].split('_'):
                self.gnn_loss = [nn.CrossEntropyLoss().cuda(self.gnn_dev) for
                                 _ in range(
                        self.config['models']['gnn_params']['gnn'][
                            'num_layers'])]
            elif 'lsgnn' in params['fns'].split('_'):
                self.gnn_loss = [losses.CrossEntropyLabelSmooth(
                    num_classes=num_classes, dev=self.gnn_dev).cuda(
                    self.gnn_dev) for
                                 _ in range(
                        self.config['models']['gnn_params']['gnn'][
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

        # Add Center Loss
        if 'center' in params['fns'].split('_'):
            self.center = losses.CenterLoss(num_classes=num_classes).cuda(
                self.gnn_dev)
            self.opt_center = torch.optim.SGD(self.center.parameters(),
                                              lr=0.5)
        else:
            self.center = None

        # Add triplet loss
        if 'triplet' in params['fns'].split('_'):
            self.triplet = losses.TripletLoss(margin=0.5).cuda(self.gnn_dev)
        else:
            self.triplet = None

        if 'of' in params['fns'].split('_'):
            self.of = nn.MSELoss().cuda(self.gnn_dev)
        else:
            self.of = None

        if 'distillSh' in params['fns'].split('_'):
            self.distill = losses.CrossEntropyDistill().cuda(self.gnn_dev)
            with open(params['preds'], 'r') as f:
                self.soft_targets = json.load(f)
            self.soft_targets = {
                k: F.softmax(torch.tensor(v) / params['soft_temp']) for k, v in
                self.soft_targets.items()}
        elif 'distillKL' in params['fns'].split('_'):
            self.distill = losses.KLDivWithLogSM().cuda(self.gnn_dev)
            with open(params['preds'], 'r') as f:
                self.soft_targets = json.load(f)
            self.soft_targets = {
                k: F.softmax(torch.tensor(v) / params['soft_temp']) for k, v in
                self.soft_targets.items()}
        else:
            self.distill = None

        if 'ofpre' in params['fns'].split('_'):
            self.of_pre = nn.SmoothL1Loss().cuda(self.gnn_dev)
            with open(params['feats'], 'r') as f:
                self.feat_targets = json.load(f)
        else:
            self.of_pre = None

        if 'distance' in params['fns'].split('_'):
            self.distance = losses.DistanceLoss().cuda(self.gnn_dev)
            with open(params['feats'], 'r') as f:
                self.feat_targets = json.load(f)
        else:
            self.distance = None

    def get_save_name(self):
        if self.config['mode'] == 'pretraining':
            mode = 'finetuned_'
        else:
            mode = ''
        if self.config['models']['encoder_params']['neck']:
            mode = mode + 'neck_'

        return mode

    def update_params(self):
        self.sample_hypers() if 'hyper' in self.config['mode'].split('_') else None

        if self.config['dataset']['val']:
            self.config['dataset']['num_classes'] -= 100

        if 'test' in self.config['mode'].split('_'):
            self.config['train_params']['num_epochs'] = 1

        if self.config['dataset']['sampling'] != 'no':
            self.config['train_params']['num_epochs'] += 10

    def sample_hypers(self):
        config = {'lr': 10 ** random.uniform(-8, -2),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'num_classes_iter': random.randint(6, 13),  # 100
                  'num_elements_class': random.randint(3, 7),
                  'temperatur': random.random(),
                  'num_epochs': 40}
        self.config['train_params'].update(config)

        config = {'num_layers': random.randint(1, 4)}
        config = {'num_heads': random.choice([1, 2, 4, 8])}

        self.config['models']['gnn_params']['gnn'].update(config)

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
                bssampling=self.config['dataset']['bssampling'])
