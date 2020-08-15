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
#from torch import autograd
#autograd.set_detect_anomaly(True)

logger = logging.getLogger('GNNReID.Training')


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

        self.best_recall = 0
        self.best_hypers = None
        self.num_iter = 30 if config['mode'] == 'hyper_search' else 1

    def train(self):
        best_recall = 0
        for i in range(self.num_iter):
            logger.info('Search iteration {}'.format(i + 1))
            mode = self.get_save_name()
            self.update_params()
            print(self.config)
            print(self.timer)

            encoder, sz_embed = net.load_net(
                self.config['dataset']['dataset_short'],
                self.config['dataset']['num_classes'],
                self.config['mode'],
                **self.config['models']['encoder_params'])
            self.encoder = encoder.to(self.device)

            """self.gnn = net.GNNReID(self.device, self.config['models']['gnn_params'], sz_embed).to(
                self.device)"""
            self.gnn = net.TransformerEncoder(d_embed=2048,
                                              nhead=self.config['models']['gnn_params']['gnn']['num_heads'],
                                              num_layers=self.config['models']['gnn_params']['gnn']['num_layers'],
                                              neck=0,
                                              num_classes=self.config['dataset']['num_classes']).to(self.device)

            self.graph_generator = net.GraphGenerator(self.device, **self.config['graph_params'])
            
            self.evaluator = Evaluator(**self.config['eval_params'])
            
            #for param in self.encoder.parameters():
            #    param.requires_grad = False
            params = list(self.gnn.parameters()) + list(self.encoder.parameters())
            # list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
            param_groups = [{'params': params,
                             'lr': self.config['train_params']['lr']}]

            self.opt = RAdam(param_groups,
                             weight_decay=self.config['train_params'][
                                 'weight_decay'])

            self.get_loss_fn(self.config['train_params']['loss_fn'], self.config['dataset']['num_classes'])

            # Do training in mixed precision
            if self.config['train_params']['is_apex']:
                self.encoder, self.opt = amp.initialize(self.encoder, self.opt,
                                                        opt_level="O1")

            if torch.cuda.device_count() > 1:
                self.encoder = nn.DataParallel(self.encoder)

            self.get_data(self.config['dataset'], self.config['train_params'],
                          self.config['mode'])

            best_accuracy, model = self.execute(
                self.config['train_params'],
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Recall: {}'.format(best_accuracy))
            if best_accuracy > best_recall and not self.config['mode'] == 'test':
                os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
                          str(best_accuracy) + mode + self.net_type + '_' +
                          self.dataset_short + '.pth')
                best_recall = best_accuracy
                best_hypers = ', '.join(
                        [str(k) + ': ' + str(v) for k, v in self.config.items()])
            elif self.config['mode'] == 'test':
                best_recall = best_accuracy
                best_hypers = ', '.join(
                        [str(k) + ': ' + str(v) for k, v in self.config.items()])

        logger.info("Best Hyperparameters found: " + best_hypers)
        logger.info(
            "Achieved {} with this hyperparameters".format(best_recall))
        logger.info("-----------------------------------------------------\n")

    def execute(self, train_params, eval_params):
        since = time.time()
        best_accuracy = 0
        scores = list()

        # feature dict for distance sampling
        if self.distance_sampling != 'no':
            self.feature_dict = dict()

        for e in range(1, train_params['num_epochs'] + 1):
            # If not testing
            if not self.config['mode'] == 'test':
                logger.info(
                    'Epoch {}/{}'.format(e, train_params['num_epochs']))
                if e == 1131:
                    print("reduces Learning rate")
                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                if e == 1161:
                    print("reduces Learning rate")
                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                self.check_sampling_strategy(e)

                # If distance_sampling == only, use first epoch to get features
                if (self.distance_sampling == 'only' and e == 1) \
                        or (self.distance_sampling == 'pre' and e == 1) \
                        or (self.distance_sampling == 'pre_soft' and e == 1):
                    model_is_training = self.encoder.training
                    self.encoder.eval()
                    loss = 0
                    with torch.no_grad():
                        for x, Y, I in self.dl_tr:
                            Y = Y.to(self.device)
                            probs, fc7 = self.encoder(x.to(self.device))
                            for y, f, i in zip(Y, fc7, I):
                                if y.data.item() in self.feature_dict.keys():
                                    self.feature_dict[y.data.item()][i.item()] = f
                                else:
                                    self.feature_dict[y.data.item()] = {i.item(): f}

                # Normal training with backpropagation
                else:
                    for x, Y, I in self.dl_tr:
                        loss = self.forward_pass(x, Y, I, train_params)

                        # Check possible net divergence
                        if torch.isnan(loss):
                            logger.error("We have NaN numbers, closing\n\n\n")
                            sys.exit(0)

                        # Backpropagation
                        if train_params['is_apex']:
                            with amp.scale_loss(loss, self.opt) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        self.opt.step()
                        
                        if self.gnn_loss:
                            for param in self.gnn.parameters():
                                if torch.isnan(param).any():
                                    print(param)
                                if torch.isnan(param.grad).any():
                                    print(param, param.grad)

                        if self.center:
                            for param in self.center.parameters():
                                param.grad.data *= (
                                        1. / self.args.scaling_center)
                            self.opt_center.step()

                # Set model to training mode again, if first epoch and only
                if (self.distance_sampling == 'only' and e == 1) \
                        or (self.distance_sampling == 'pre' and e == 1) \
                        or (self.distance_sampling == 'pre_soft' and e == 1):
                    self.encoder.train(model_is_training)
            
            [self.losses_mean[k].append(sum(v) / len(v)) for k, v in
             self.losses.items()]
            losses = defaultdict(list)
            logger.info('Loss Values: ')
            logger.info(', '.join([str(k) + ': ' + str(v[-1]) for k, v in self.losses_mean.items()]))
            # compute ranks and mAP at the end of each epoch
            best_accuracy = self.evaluate(eval_params, scores, e, loss,
                                          best_accuracy)

        end = time.time()

        self.save_results(train_params, since, end, best_accuracy, scores)

        return best_accuracy, self.encoder

    def forward_pass(self, x, Y, I, train_params):
        Y = Y.to(self.device)
        self.opt.zero_grad()
        if self.center:
            self.opt_center.zero_grad()
        probs, fc7, student_feats = self.encoder(x.to(self.device),
                                  output_option=train_params['output_train'])

        # Add feature vectors to dict if distance sampling
        if self.distance_sampling != 'no':
            for y, f, i in zip(Y, fc7, I):
                if y.data.item() in self.feature_dict.keys():
                    self.feature_dict[y.data.item()][i.item()] = f
                else:
                    self.feature_dict[y.data.item()] = {i.item(): f}

        # Compute CE Loss
        loss = 0
        if self.ce:
            loss0 = self.ce(probs, Y)
            loss+= train_params['loss_fn']['scaling_ce'] * loss0
            self.losses['Cross Entropy'].append(loss.item())

        # Add other losses of not pretraining
        if not self.config['mode'] == 'pretraining':
            
            if self.gnn_loss:
                #print("Next Batch")
                #print(torch.argmax(probs, dim=1), Y)
                #print(fc7)
                edge_attr, edge_index, fc7 = self.graph_generator.get_graph(fc7)
                #print(fc7)
                #pred, feats = self.gnn(fc7, edge_index, edge_attr)
                pred, feats = self.gnn(fc7, train_params['output_train'])
                
                loss1 = self.gnn_loss(pred, Y)
                #print(torch.argmax(pred, dim =1), Y)
                #print(loss1)

                loss += train_params['loss_fn']['scaling_gnn'] * loss1
                self.losses['GNN'].append(loss1.item())

            # Compute center loss
            if self.center:
                loss2 = self.center(fc7, Y)
                loss += train_params['loss_fn']['scaling_center'] * loss2
                self.losses['Center'].append(loss2.item())

            # Compute Triplet Loss
            if self.triplet:
                triploss, _ = self.triplet(fc7, Y)
                loss += train_params['loss_fn']['scaling_triplet'] * triploss
                self.losses['Triplet'].append(triploss.item())

            # Compute MSE regularization
            if self.of:
                p = feats.detach()
                of_reg = self.of(student_feats, p)
                loss += train_params['loss_fn']['scaling_of'] * of_reg
                self.losses['OF'].append(of_reg.item())
            self.losses['Total Loss'].append(loss.item())
        else:
            # For pretraining just use acc as evaluation
            _, preds = torch.max(probs, 1)
            self.denom += Y.shape[0]
            self.running_corrects += torch.sum(
                preds == Y.data).cpu().data.item()

        return loss

    def evaluate(self, eval_params, scores, e, loss, best_accuracy):
        if not self.config['mode'] == 'pretraining':
            with torch.no_grad():
                logger.info('EVALUATION')
                mAP, top = self.evaluator.evaluate_reid(self.encoder, self.dl_ev,
                        self.query, gallery=self.gallery)

                logger.info('Mean AP: {:4.1%}'.format(mAP))

                logger.info('CMC Scores{:>12}{:>12}{:>12}'
                            .format('allshots', 'cuhk03', 'Market'))

                for k in (1, 5, 10):
                    logger.info('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                                .format(k, top['allshots'][k - 1],
                                        top['cuhk03'][k - 1],
                                        top['Market'][k - 1]))

                scores.append((mAP,
                               [top[self.dataset_short][k - 1] for k
                                in
                                [1, 5, 10]]))
                self.encoder.current_epoch = e
                if top[self.dataset_short][0] > best_accuracy:
                    best_accuracy = top[self.dataset_short][0]
                    torch.save(self.encoder.state_dict(),
                               osp.join(self.save_folder_nets,
                                        self.fn + '.pth'))

        else:
            logger.info(
                'Loss {}, Accuracy {}'.format(torch.mean(loss.cpu()),
                                              self.running_corrects / self.denom))

            scores.append(self.running_corrects / self.denom)
            self.denom = 0
            self.running_corrects = 0
            if scores[-1] > best_accuracy:
                best_accuracy = scores[-1]
                torch.save(self.encoder.state_dict(),
                           osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))

        return best_accuracy

    def check_sampling_strategy(self, e):
        # Different Distance Sampling Strategies
        if (self.distance_sampling == 'only' and e > 1) \
                or (self.distance_sampling == 'alternating'
                    and e % 2 == 0) or (self.distance_sampling
                                        == 'pre' and e > 1) or (
                self.distance_sampling == 'pre_soft' and e > 1):
            self.dl_tr = self.dl_tr2
            self.dl_ev = self.dl_ev2
            self.gallery = self.gallery2
            self.query = self.query2
        elif (self.distance_sampling == 'only' and e == 1) \
                or (self.distance_sampling == 'alternating'
                    and e % 2 != 0) or (self.distance_sampling
                                        == 'pre' and e == 1) or (
                self.distance_sampling == 'pre_soft' and e == 1):
            self.dl_tr = self.dl_tr1
            self.dl_ev = self.dl_ev1
            self.gallery = self.gallery1
            self.query = self.query1
        if self.distance_sampling != 'no':
            # set feature dict of sampler = feat dict of previous epoch
            self.dl_tr.sampler.feature_dict = self.feature_dict
            self.dl_tr.sampler.epoch = e

    def save_results(self, train_params, since, end, best_accuracy, scores):
        logger.info(
            'Completed {} epochs in {}s on {}'.format(
                train_params['num_epochs'],
                end - since,
                self.dataset_short))

        file_name = str(
            best_accuracy) + '_' + self.dataset_short + '_' + str(self.timer)
        if self.config['mode'] == 'test':
            file_name = 'test_' + file_name
        if not self.config['mode'] == 'pretraining':
            with open(
                    osp.join(self.save_folder_results, file_name + '.txt'),
                    'w') as fp:
                fp.write(file_name + "\n")
                fp.write(str(self.config))
                fp.write('\n')
                fp.write('\n'.join('%s %s' % x for x in scores))
                fp.write("\n\n\n")

        results_dir = osp.join('results', file_name)
        utils.make_dir(results_dir)

        # plot losses
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

    def get_loss_fn(self, params, num_classes):
        self.losses = defaultdict(list)
        self.losses_mean = defaultdict(list)
        # Loss for GroupLoss

        if 'gnn' in params['fns'].split('_'):
            self.gnn_loss = nn.CrossEntropyLoss().to(self.device)
        elif 'lsgnn' in params['fns'].split('_'):
            self.gnn_loss = losses.CrossEntropyLabelSmooth(
                num_classes=num_classes)
        elif 'focalgnn' in params['fns'].split('_'):
            self.gnn_loss = losses.FocalLoss().to(self.device)
        else:
            self.gnn_loss = None

        # Label smoothing for CrossEntropy Loss
        if 'lsce' in params['fns'].split('_'):
            self.ce = losses.CrossEntropyLabelSmooth(
                num_classes=num_classes)
        elif 'focalce' in params['fns'].split('_'):
            self.ce = losses.FocalLoss().to(self.device)
        elif 'ce' in params['fns'].split('_'):
            self.ce = nn.CrossEntropyLoss().to(self.device)
        else:
            self.ce = None

        # Add Center Loss
        if 'center' in params['fns'].split('_'):
            self.center = losses.CenterLoss(num_classes=num_classes)
            self.opt_center = torch.optim.SGD(self.center.parameters(),
                                              lr=self.args.center_lr)
        else:
            self.center = None

        # Add triplet loss
        if 'triplet' in params['fns'].split('_'):
            self.triplet = losses.TripletLoss(margin=0.5)
        else:
            self.triplet = None

        if 'of' in params['fns'].split('_'):
            self.of = nn.MSELoss().to(self.device)
        else:
            self.of = None

    def get_save_name(self):
        if self.config['mode'] == 'pretraining':
            mode = 'finetuned_'
        else:
            mode = ''
        if self.config['models']['encoder_params']['neck']:
            mode = mode + 'neck_'

        return mode

    def update_params(self):
        self.sample_hypers() if self.config['mode'] == 'hyper_search' else None

        if self.config['dataset']['val']:
            self.config['dataset']['num_classes'] -= 100

        if self.config['mode'] == 'test':
            self.config['train_params']['num_epochs'] = 1

        if self.config['dataset']['sampling'] != 'no':
            self.config['train_params']['num_epochs'] += 30

    def sample_hypers(self):
        config = {'lr': 10 ** random.uniform(-8, -3),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'num_classes_iter': random.randint(3, 15),
                  'num_elements_class': random.randint(5, 15),
                  'temperature': random.randint(10, 80),
                  'num_epochs': 30}
        self.config['train_params'].update(config)

    def get_data(self, config, train_params, mode):

        if not mode == 'pretraining':
            # If distance sampling
            self.distance_sampling = config['sampling']
            if config['sampling'] != 'no':
                seed = random.randint(0, 100)
                # seed = 19
                self.dl_tr2, self.dl_ev2, self.query2, self.gallery2 = data_utility.create_loaders(
                    data_root=config['dataset_path'],
                    num_workers=config['nb_workers'],
                    num_classes_iter=train_params['num_classes_iter'],
                    num_elements_class=train_params['num_elements_class'],
                    size_batch=train_params['num_classes_iter'] * train_params[
                        'num_elements_class'],
                    mode=mode,
                    trans=config['trans'],
                    distance_sampler=config['sampling'],
                    val=config['val'],
                    seed=seed)
                self.dl_tr1, self.dl_ev1, self.query1, self.gallery1 = data_utility.create_loaders(
                    data_root=config['dataset_path'],
                    num_workers=config['nb_workers'],
                    num_classes_iter=train_params['num_classes_iter'],
                    num_elements_class=train_params['num_elements_class'],
                    size_batch=train_params['num_classes_iter'] * train_params[
                        'num_elements_class'],
                    mode=mode,
                    trans=config['trans'],
                    distance_sampler='no',
                    val=config['val'],
                    seed=seed)
            # If testing or normal training
            else:
                self.dl_tr, self.dl_ev, self.query, self.gallery = data_utility.create_loaders(
                    data_root=config['dataset_path'],
                    num_workers=config['nb_workers'],
                    num_classes_iter=train_params['num_classes_iter'],
                    num_elements_class=train_params['num_elements_class'],
                    size_batch=train_params['num_classes_iter'] * train_params[
                        'num_elements_class'],
                    mode=mode,
                    trans=config['trans'],
                    distance_sampler=config['sampling'],
                    val=config['val'])

        # Pretraining dataloader
        else:
            self.running_corrects = 0
            self.denom = 0
            self.dl_tr = data_utility.create_loaders(size_batch=64,
                                                     data_root=self.args.cub_root,
                                                     num_workers=self.args.nb_workers,
                                                     mode=self.args.mode,
                                                     trans=self.args.trans,
                                                     pretraining=self.args.pretraining)
