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

        self.best_recall = 0
        self.best_hypers = None
        self.num_iter = 30 if config['mode'] == 'hyper_search' or config['mode'] == 'gnn_hyper_search' or self.config['mode'] == 'pseudo_hyper_search'else 1

    def train(self):
        best_recall = 0
        best_loss = 100
        #self.num_iter = 26
        #num_classes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] #[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        #num_samples = [2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8]
        #num_layers = list(range(1, 14))
        for i in range(self.num_iter):
            #print("Iter {}/{}".format(i+1, self.num_iter))
            #self.config['models']['gnn_params']['gnn']['num_layers'] = num_layers[i]
            #self.nb_clusters = num_classes[i]
            self.nb_clusters = self.config['train_params']['num_classes_iter']
            #self.nb_clusters = 900
            #self.config['train_params']['num_elements_class'] = num_samples[i]
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
            self.encoder = encoder.cuda(self.device) #to(self.device) 
            if torch.cuda.device_count() > 1:
                self.gnn_dev = 1
            else:
                self.gnn_dev = 0
            print(self.device, self.gnn_dev)
            # 1 == dev
            self.gnn = net.GNNReID(self.gnn_dev, self.config['models']['gnn_params'], sz_embed).cuda(self.gnn_dev) #to(self.device)
            if self.config['models']['gnn_params']['pretrained_path'] != "no":
                
                load_dict = torch.load(self.config['models']['gnn_params']['pretrained_path'], map_location='cpu')
                '''
                load_dict_new = dict()
                for k, v in load_dict.items():
                    if k.split('.')[0] == 'bottleneck' or k.split('.')[0] == 'fc':
                        k = '.'.join(k.split('.')[:1] + ['0'] + k.split('.')[1:])
                        load_dict_new[k] = v
                    elif k.split('.')[0] != 'gnn_model':
                        load_dict_new[k] = v
                    else:
                        k = '.'.join(k.split('.')[:2] + ['layers'] + k.split('.')[2:])
                        load_dict_new[k] = v
                load_dict = load_dict_new
                '''
                self.gnn.load_state_dict(load_dict)

            self.graph_generator = net.GraphGenerator(self.device, **self.config['graph_params'])
             
            if self.config['application'] == 'DML':
                self.evaluator = Evaluator_DML(nb_clusters=self.nb_clusters, dev=self.device, gnn_dev=self.gnn_dev, **self.config['eval_params'])
            else:
                self.evaluator = Evaluator(**self.config['eval_params'])
            
            '''update_list = ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
                        "bottleneck.weight", "bottleneck.bias", "bottleneck.running_mean",
                        "bottleneck.running_var", "bottleneck.num_batches_tracked", "fc.weight"]
            update_list = []

            for name, param in self.encoder.named_parameters():
                if name not in update_list:
                    param.requires_grad=False

            for param in self.gnn.parameters():
                param.requires_gras=False'''
            
            old_params = list()
            new_params = list()

            for name, param in list(self.encoder.named_parameters()):
                if name.split('.')[0] == 'layer5' or name.split('.')[0] == 'layer6':
                    new_params.append(param)
                else:
                    old_params.append(param)
            
            params = list(self.gnn.parameters()) + old_params
            # list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
            param_groups = [{'params': params,
                             'lr': self.config['train_params']['lr']}, 
                             {'params': new_params,
                                 'lr': self.config['train_params']['lr'] * 10}]

            self.opt = RAdam(param_groups,
                             weight_decay=self.config['train_params'][
                                 'weight_decay'])

            self.get_loss_fn(self.config['train_params']['loss_fn'], self.config['dataset']['num_classes'])

            # Do training in mixed precision
            if self.config['train_params']['is_apex']:
                [self.encoder, self.gnn], self.opt = amp.initialize([self.encoder, self.gnn], self.opt,
                                                        opt_level="O1")
            #print(torch.cuda.device_count())
            #quit()
            if torch.cuda.device_count() > 1:
                gpus = list(range(torch.cuda.device_count()))
                gpus.remove(self.gnn_dev)
                self.encoder = nn.DataParallel(self.encoder, device_ids=gpus)
                #self.encoder = nn.DataParallel(self.encoder)
                #self.gnn = nn.DataParallel(self.gnn)

            self.get_data(self.config['dataset'], self.config['train_params'],
                          self.config['mode'])

            best_accuracy, model = self.execute(
                self.config['train_params'],
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Recall: {}'.format(best_accuracy))
            if best_accuracy > best_recall and not self.config['mode'] == 'test' and not self.config['mode'] == 'gnn_test' and not self.config['mode'] == 'traintest_test' and not self.config['mode'] == 'pseudo_test' and not self.config['mode'] == 'knn_test':
                os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
                          str(best_accuracy) + mode + self.net_type + '_' +
                          self.dataset_short + '.pth')
                os.rename(osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
                          str(best_accuracy) + 'gnn_' + mode + self.net_type + '_' +
                          self.dataset_short + '.pth')
                best_recall = best_accuracy
                best_hypers = ', '.join(
                        [str(k) + ': ' + str(v) for k, v in self.config.items()])
            elif self.config['mode'] == 'test' or self.config['mode'] == 'gnn_test' or self.config['mode'] == 'traintest_test' or self.config['mode'] == 'pseudo_test' or self.config['mode'] == 'knn_test':
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
        best_loss = 100
        scores = list()
        self.preds, self.feats, self.labs, self.preds_before, self.feats_before = dict(), dict(), dict(), dict(), dict()
        self.best_preds, self.best_labs, self.best_feats, self.best_preds_before, self.best_feats_before = dict(), dict(), dict(), dict(), dict()

        # feature dict for distance sampling
        if self.distance_sampling != 'no':
            self.feature_dict = dict()

        for e in range(1, train_params['num_epochs'] + 1):
            if self.config['mode'] == 'test' or self.config['mode'] == 'gnn_test' or self.config['mode'] == 'traintest_test' or self.config['mode'] == 'pseudo_test' or self.config['mode'] == 'knn_test':
                best_accuracy, best_loss = self.evaluate(eval_params, scores, 0, 0, 0, 10000)
            # If not testing
            else:
                logger.info(
                    'Epoch {}/{}'.format(e, train_params['num_epochs']))
                logger.info("No reduction of learning rate")
                if e == 31:
                    print("reduces Learning rate")
                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
                        'gnn_' + self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                if e == 41:
                    print("reduces Learning rate")
                    self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
                        'gnn_' + self.fn + '.pth')))

                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                self.check_sampling_strategy(e)

                # If distance_sampling == only, use first epoch to get features
                if (self.distance_sampling == 'only' and e == 1) \
                        or (self.distance_sampling == 'pre' and e == 1) \
                        or (self.distance_sampling == 'pre_soft' and e == 1) \
                        or (self.distance_sampling == '8closest' and e == 1) \
                        or (self.distance_sampling == '8closestClass' and e == 1) \
                        or (self.distance_sampling == 'kmeans' and e == 1) \
                        or (self.distance_sampling == 'kmeansClosest' and e == 1):
                    model_is_training = self.encoder.training
                    gnn_is_training = self.gnn.training
                    self.encoder.eval()
                    self.gnn.eval()
                    loss = 0
                    with torch.no_grad():
                        for x, Y, I, P in self.dl_tr:
                            Y = Y.cuda(self.device)
                            probs, fc7, student_feats = self.encoder(x.cuda(self.device), train_params['output_train_enc'])
                            if self.gnn_loss or self.of:
                                edge_attr, edge_index, fc7 = self.graph_generator.get_graph(fc7)
                                fc7 = fc7.cuda(self.gnn_dev)
                                edge_attr = edge_attr.cuda(self.gnn_dev)
                                edge_index = edge_index.cuda(self.gnn_dev)
                                pred, feats = self.gnn(fc7, edge_index, edge_attr, train_params['output_train_gnn'])
                                features = feats[0]
                            else: 
                                features = fc7
                            
                            for y, f, i in zip(Y, features, I):
                                if y.data.item() in self.feature_dict.keys():
                                    self.feature_dict[y.data.item()][i.item()] = f
                                else:
                                    self.feature_dict[y.data.item()] = {i.item(): f}
                    
                # Normal training with backpropagation
                else:
                    self.dl_tr.sampler.epoch = e
                    for x, Y, I, P in self.dl_tr:
                        loss = self.forward_pass(x, Y, I, P, train_params)
                        if self.gnn_loss:
                            for param in self.gnn.parameters():
                                if torch.isnan(param).any():
                                    logger.info("Parameters")
                                    logger.info(param)
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any():
                                        logger.info("Gradient")
                                        logger.info(param, param.grad)

                        # Check possible net divergence
                        if torch.isnan(loss):
                            logger.error("We have NaN numbers, closing\n\n\n")
                            #sys.exit(0)
                            return 0.0, self.encoder
                        #print("New batch")
                        #for param in self.gnn.parameters():
                        #    print(param)
                        #    print(param.grad)
                        #    break
                        # Backpropagation
                        if train_params['is_apex']:
                            with amp.scale_loss(loss, self.opt) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        #for param in self.gnn.parameters():
                        #    print(param)
                        #    print(param.grad)
                        #    break

                        self.opt.step()
                        #for param in self.gnn.parameters():
                        #    print(param)
                        #    break

                        if self.gnn_loss:
                            for param in self.gnn.parameters():
                                if torch.isnan(param).any():
                                    print(param)
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any():
                                        print(param, param.grad)

                        if self.center:
                            for param in self.center.parameters():
                                param.grad.data *= (
                                        1. / self.config['train_params']['loss_fn']['scaling_center'])
                            self.opt_center.step()

                # Set model to training mode again, if first epoch and only
                if (self.distance_sampling == 'only' and e == 1) \
                        or (self.distance_sampling == 'pre' and e == 1) \
                        or (self.distance_sampling == 'pre_soft' and e == 1) \
                        or (self.distance_sampling == '8closest' and e == 1) \
                        or (self.distance_sampling == 'kmeans' and e == 1) \
                        or (self.distance_sampling == 'kmeansClosest' and e == 1):
                    self.encoder.train(model_is_training)
                    self.gnn.train(gnn_is_training)
                best_accuracy, best_loss = self.evaluate(eval_params, scores, e, sum(self.losses['Total Loss'])/len(self.losses['Total Loss']) if len(self.losses['Total Loss'])>0 else 10,
                                          best_accuracy, best_loss)

            [self.losses_mean[k].append(sum(v) / len(v)) for k, v in
             self.losses.items() if len(v)>10]
            losses = defaultdict(list)
            logger.info('Loss Values: ')
            logger.info(', '.join([str(k) + ': ' + str(v[-1]) for k, v in self.losses_mean.items()]))
            # compute ranks and mAP at the end of each epoch

        end = time.time()

        self.save_results(train_params, since, end, best_accuracy, scores)

        return best_accuracy, self.encoder

    def forward_pass(self, x, Y, I, P, train_params):
        Y = Y.cuda(self.device)
        self.opt.zero_grad()
        if self.center:
            self.opt_center.zero_grad()
        probs, fc7, student_feats = self.encoder(x.cuda(self.device),
                                  output_option=train_params['output_train_enc'])

        #print(torch.max(torch.nn.functional.softmax(probs, dim=1), dim=1), torch.argmax(torch.nn.functional.softmax(probs, dim=1), dim=1), Y)
        #quit()
        # Add feature vectors to dict if distance sampling
        if self.distance_sampling != 'no':
            for y, f, i in zip(Y, fc7, I):
                i_new = y.data.item() - self.config['dataset']['num_classes']
                if y.data.item() in self.feature_dict.keys():
                    self.feature_dict[y.data.item()][i.item()] = f.detach()
                else:
                    self.feature_dict[y.data.item()] = {i.item(): f.detach()}
         
        # Compute CE Loss
        loss = 0
        if self.ce:
            probs = probs.cuda(self.gnn_dev)
            Y = Y.cuda(self.gnn_dev)
            #print(Y)
            loss0 = self.ce(probs/self.config['train_params']['temperatur'], Y)
            loss+= train_params['loss_fn']['scaling_ce'] * loss0
            self.losses['Cross Entropy'].append(loss.item())

        # Add other losses of not pretraining
        if not self.config['mode'] == 'pretraining':
            
            if self.gnn_loss or self.of:
                #print("Next Batch")
                #print(torch.argmax(probs, dim=1), Y)
                #print(fc7)
                edge_attr, edge_index, fc7 = self.graph_generator.get_graph(fc7)
                fc7 = fc7.cuda(self.gnn_dev)
                edge_attr = edge_attr.cuda(self.gnn_dev)
                edge_index = edge_index.cuda(self.gnn_dev)
                loss = loss.cuda(self.gnn_dev)
                pred, feats = self.gnn(fc7, edge_index, edge_attr, train_params['output_train_gnn'])

                for path,  pre, f, pre_b, f_b, lab in zip(P, pred[-1], feats[-1], fc7, probs, Y):
                    self.preds[path] = pre.detach()
                    self.feats[path] = f.detach()
                    self.preds_before[path] = pre_b.detach()
                    self.feats_before[path] = f_b.detach()
                    self.labs[path] = lab
 
                #pred, feats = self.gnn(fc7, train_params['output_train'])
                if self.gnn_loss:
                    if self.every:
                        loss1 = [gnn_loss(pr/self.config['train_params']['temperatur'], Y.cuda(self.gnn_dev)) for gnn_loss, pr in zip(self.gnn_loss, pred)]
                    else:
                        loss1 = [self.gnn_loss(pred[0]/self.config['train_params']['temperatur'], Y.cuda(self.gnn_dev))]
                    lo = [train_params['loss_fn']['scaling_gnn'] * l for l in loss1]
                    loss += sum(lo)
                    [self.losses['GNN'+ str(i)].append(l.item()) for i, l in enumerate(loss1)]
            
            if self.distance_sampling != 'no':
                if self.gnn_loss or self.of:
                    features = feats[0]
                else:
                    features = fc7
                for y, f, i in zip(Y, features, I):
                    if y.data.item() in self.feature_dict.keys():
                        self.feature_dict[y.data.item()][i.item()] = f.detach()
                    else:
                        self.feature_dict[y.data.item()] = {i.item(): f.detach()}


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
                p = feats[0].detach().cuda(self.gnn_dev)
                of_reg = self.of(fc7, p)
                loss += train_params['loss_fn']['scaling_of'] * of_reg
                self.losses['OF'].append(of_reg.item())

            # Compute CE loss with soft targets = predictions of gnn
            if self.distill:
                target = torch.stack([self.soft_targets[p] for p in P]).cuda(self.gnn_dev)
                distill = self.distill(probs/self.config['train_params']['loss_fn']['soft_temp'], target)
                loss += train_params['loss_fn']['scaling_distill'] * distill
                self.losses['Distillation'].append(distill.item())

            # compute MSE loss with feature vectors from gnn
            if self.of_pre:
                target = torch.stack([torch.tensor(self.feat_targets[p]) for p in P]).cuda(self.gnn_dev)
                of_pre = self.of_pre(fc7, target)

                loss += train_params['loss_fn']['scaling_of_pre'] * of_pre
                self.losses['OF Pretrained'].append(of_pre.item())

            if self.distance:
                teacher = torch.stack([torch.tensor(self.feat_targets[p]) for p in P]).cuda(self.gnn_dev)
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

    def evaluate(self, eval_params, scores, e, loss, best_accuracy, best_loss):
        if not self.config['mode'] == 'pretraining':
            with torch.no_grad():
                logger.info('EVALUATION')
                if self.config['mode'] != 'gnn' and self.config['mode'] != 'gnn_test' and self.config['mode'] != 'gnn_hyper_search' and self.config['mode'] != 'pseudo' and self.config['mode'] != 'pseudo_test' and self.config['mode'] != 'pseudo_hyper_search' and self.config['mode'] != 'traintest' and self.config['mode'] != 'traintest_test' and self.config['mode'] != 'knn_test' and self.config['mode'] != 'knn' and self.config['mode'] != 'knn_hyper_search':
                    if self.dataset_short != 'sop': 
                        logger.info('Train Dataset')
                        mAP_train, top_train = self.evaluator.evaluate(self.encoder, self.dl_tr,
                                    self.query, gallery=self.gallery, net_type=self.net_type,
                                    dataroot=self.config['dataset']['dataset_short'],
                                    nb_classes=self.config['dataset']['num_classes'])
                    logger.info('Test Dataset')
                    mAP, top = self.evaluator.evaluate(self.encoder, self.dl_ev,
                            self.query, gallery=self.gallery, net_type=self.net_type,
                            dataroot=self.config['dataset']['dataset_short'], 
                            nb_classes=self.config['dataset']['num_classes'])
                else:
                    if self.dataset_short != 'sop':
                        logger.info('Train Dataset')
                        mAP_train, top_train = self.evaluator.evaluate(self.encoder, self.dl_tr,
                                    self.query, self.gallery, self.gnn, self.graph_generator, 
                                    dl_ev_gnn=None, net_type=self.net_type,
                                    dataroot=self.config['dataset']['dataset_short'],
                                    nb_classes=self.config['dataset']['num_classes'])
                    
                    logger.info('Test Dataset')
                    mAP, top = self.evaluator.evaluate(self.encoder, self.dl_ev,
                            self.query, self.gallery, self.gnn, self.graph_generator, 
                            dl_ev_gnn=self.dl_ev_gnn, net_type=self.net_type,
                            dataroot=self.config['dataset']['dataset_short'],
                            nb_classes=self.config['dataset']['num_classes'])

                if self.config['application'] == 'reid':
                    logger.info('Mean AP Train: {:4.1%}'.format(mAP_train))

                    logger.info('Train CMC Scores{:>12}{:>12}{:>12}'
                                .format('allshots', 'cuhk03', 'Market'))
    
                    for k in (1, 5, 10):
                        logger.info('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                                    .format(k, top_train['allshots'][k - 1],
                                            top_train['cuhk03'][k - 1],
                                            top_train['Market'][k - 1]))

                    logger.info('Mean AP: {:4.1%}'.format(mAP))

                    logger.info('CMC Scores{:>12}{:>12}{:>12}'
                                .format('allshots', 'cuhk03', 'Market'))
    
                    for k in (1, 5, 10):
                        logger.info('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                                    .format(k, top['allshots'][k - 1],
                                            top['cuhk03'][k - 1],
                                            top['Market'][k - 1]))
                
                    if self.dataset_short == 'cuhk03-np':
                        eval_method = 'Market'
                    elif self.dataset_short == 'dukemtmc':
                        eval_method = 'Market'
                    else:
                        eval_method = self.dataset_short
                    scores.append((mAP,
                                   [top[eval_method][k - 1] for k
                                    in
                                    [1, 5, 10]]))
                    eval_met = top[eval_method][0]                 
                else:
                    scores.append((mAP, top))
                    eval_met = top[0]

                self.encoder.current_epoch = e
                if eval_met > best_accuracy:
                    best_accuracy = eval_met
                    #if loss < best_loss:
                    #logger.info("Loss {}, best Loss {}, Epoch {}".format(loss, best_loss, e))
                    best_loss = loss
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
            self.denom = 0
            self.running_corrects = 0
            if scores[-1] > best_accuracy:
                best_accuracy = scores[-1]
                torch.save(self.encoder.state_dict(),
                           osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))

        return best_accuracy, best_loss

    def check_sampling_strategy(self, e):
        # Different Distance Sampling Strategies
        if (self.distance_sampling == 'only' and e > 1) \
                or (self.distance_sampling == 'alternating'
                    and e % 2 == 0) or (self.distance_sampling
                                        == 'pre' and e > 1) or (
                self.distance_sampling == 'pre_soft' and e > 1) or (
                        self.distance_sampling == '8closest' and e > 1) or (
                                self.distance_sampling == '8closestClass' and e > 1) or (
                                        self.distance_sampling == 'kmeans' and e > 1) or (
                                                self.distance_sampling == 'kmeansClosest' and e > 1):
            self.dl_tr = self.dl_tr2
            self.dl_ev = self.dl_ev2
            self.gallery = self.gallery2
            self.query = self.query2
            self.dl_ev_gnn = self.dl_ev_gnn2
        elif (self.distance_sampling == 'only' and e == 1) \
                or (self.distance_sampling == 'alternating'
                    and e % 2 != 0) or (self.distance_sampling
                                        == 'pre' and e == 1) or (
                self.distance_sampling == 'pre_soft' and e == 1) or (
                        self.distance_sampling == '8closest' and e == 1) or (
                                self.distance_sampling == '8closestClass' and e == 1) or (
                                        self.distance_sampling == 'kmeans' and e == 1) or (
                                                self.distance_sampling == 'kmeansClosest' and e == 1):
            self.dl_tr = self.dl_tr1
            self.dl_ev = self.dl_ev1
            self.gallery = self.gallery1
            self.query = self.query1
            self.dl_ev_gnn = self.dl_ev_gnn1
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
        if self.config['mode'] == 'test' or self.config['mode'] == 'gnn_test' or self.config['mode'] == 'traintest_test' or self.config['mode'] == 'pseudo_test' or self.config['mode'] == 'knn_test':
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
        if not self.config['mode'] == 'test' and not self.config['mode'] == 'gnn_test' and not self.config['mode'] == 'traintest_test' and not self.config['mode'] == 'pseudo_test' and not self.config['mode'] == 'knn_test':
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
        # Loss for GroupLoss
        self.every = None
        if 'gnn' in params['fns'].split('_'):
            self.gnn_loss = nn.CrossEntropyLoss().cuda(self.gnn_dev)
        elif 'lsgnn' in params['fns'].split('_'):
            self.gnn_loss = losses.CrossEntropyLabelSmooth(
                num_classes=num_classes, dev=self.gnn_dev).cuda(self.gnn_dev)
        elif 'focalgnn' in params['fns'].split('_'):
            self.gnn_loss = losses.FocalLoss().cuda(self.gnn_dev)
        else:
            self.gnn_loss = None

        if 'gnnL' in params['fns'].split('_'):
            self.every = 1
            self.gnn_loss = [nn.CrossEntropyLoss().cuda(self.gnn_dev) for 
                    _ in range(self.config['models']['gnn_params']['gnn']['num_layers'])] 
        elif 'lsgnnL' in params['fns'].split('_'):
            self.every = 1
            self.gnn_loss = [losses.CrossEntropyLabelSmooth(
                num_classes=num_classes, dev=self.gnn_dev).cuda(self.gnn_dev) for 
                    _ in range(self.config['models']['gnn_params']['gnn']['num_layers'])]

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
            self.center = losses.CenterLoss(num_classes=num_classes).cuda(self.gnn_dev)
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
            self.soft_targets = {k: F.softmax(torch.tensor(v)/params['soft_temp']) for k, v in self.soft_targets.items()}
        elif 'distillKL' in params['fns'].split('_'):
            self.distill = losses.KLDivWithLogSM().cuda(self.gnn_dev)
            with open(params['preds'], 'r') as f:
                self.soft_targets = json.load(f)
            self.soft_targets = {k: F.softmax(torch.tensor(v)/params['soft_temp']) for k, v in self.soft_targets.items()}
        else:
            self.distill = None

        if 'ofpre' in params['fns'].split('_'):
            #self.of_pre = nn.MSELoss().to(self.device)
            #self.of_pre = nn.L1Loss().cuda(self.gnn_dev)
            print("Huber")
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
        self.sample_hypers() if self.config['mode'] == 'hyper_search' or self.config['mode'] == 'gnn_hyper_search' or self.config['mode'] == 'pseudo_hyper_search' or self.config['mode'] == 'knn_hyper_search' else None

        if self.config['dataset']['val']:
            self.config['dataset']['num_classes'] -= 100

        if self.config['mode'] == 'test' or self.config['mode'] == 'gnn_test' or self.config['mode'] == 'traintest_test' or self.config['mode'] == 'pseudo_test' or self.config['mode'] == 'knn_test':
            self.config['train_params']['num_epochs'] = 1

        if self.config['dataset']['sampling'] != 'no':
            self.config['train_params']['num_epochs'] += 30

    def sample_hypers(self):
        print("-------------------Hardcoded batchsize in line 769!-------------------")
        bs = 200
        num_classes_iter = random.randint(6, 40)
        num_elements_class = int(bs // num_classes_iter)
        config = {'lr': 10 ** random.uniform(-8, -2),
                'num_classes_iter': num_classes_iter,
                'num_elements_class': num_elements_class,
                'num_epochs': 50} #,
        #          'weight_decay': 10 ** random.uniform(-15, -6),
        #          'num_classes_iter': random.randint(6, 13), #100
        #          'num_elements_class': random.randint(3, 7),
        #          'temperatur': random.random(),
        #          'num_epochs': 5}
        #config['temperatur'] = 1
        self.config['train_params'].update(config)
        
        """
        self.config['train_params']['loss_fn']['soft_temp'] = config['temperatur']
        self.config['train_params']['loss_fn']['scaling_of_pre'] = random.random()*5
        self.config['train_params']['loss_fn']['scaling_distill'] = random.random()*5
        """
        #config = {'num_layers': random.randint(1, 8),
        #          'num_heads': random.choice([1, 2, 4, 8, 16])}
        #self.config['models']['gnn_params']['gnn'].update(config)
        
        '''logger.info("Additional augmentation hyper search")
        config = {'final_drop': random.random(),
                  'stoch_depth': random.random()}
        self.config['models']['encoder_params'].update(config)
        
        config = {'magnitude': random.randint(0, 30), 
                  'number_aug': random.randint(0, 14)}
        self.config['dataset'].update(config)'''

        logger.info("Updated Hyperparameters:")
        logger.info(self.config)

    def get_data(self, config, train_params, mode):

        if not mode == 'pretraining':
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
                    size_batch=train_params['num_classes_iter'] * train_params[
                        'num_elements_class'],
                    mode=mode,
                    trans=config['trans'],
                    number_aug=config['number_aug'],
                    magnitude=config['magnitude'],
                    distance_sampler=config['sampling'],
                    val=config['val'],
                    seed=seed,
                    num_classes=self.config['dataset']['num_classes'],
                    net_type=self.net_type,
                    nb_clusters=self.nb_clusters,
                    bssampling=self.config['dataset']['bssampling'])
                self.dl_tr1, self.dl_ev1, self.query1, self.gallery1, self.dl_ev_gnn1 = data_utility.create_loaders(
                    data_root=config['dataset_path'],
                    num_workers=config['nb_workers'],
                    num_classes_iter=train_params['num_classes_iter'],
                    num_elements_class=train_params['num_elements_class'],
                    size_batch=train_params['num_classes_iter'] * train_params[
                        'num_elements_class'],
                    mode=mode,
                    trans=config['trans'],
                    number_aug=config['number_aug'],
                    magnitude=config['magnitude'],
                    distance_sampler='no',
                    val=config['val'],
                    seed=seed,
                    num_classes=self.config['dataset']['num_classes'],
                    net_type=self.net_type,
                    nb_clusters=self.nb_clusters,
                    bssampling=self.config['dataset']['bssampling'])
            # If testing or normal training
            else:
                self.dl_tr, self.dl_ev, self.query, self.gallery, self.dl_ev_gnn = data_utility.create_loaders(
                    data_root=config['dataset_path'],
                    num_workers=config['nb_workers'],
                    num_classes_iter=train_params['num_classes_iter'],
                    num_elements_class=train_params['num_elements_class'],
                    size_batch=train_params['num_classes_iter'] * train_params[
                        'num_elements_class'],
                    mode=mode,
                    trans=config['trans'],
                    number_aug=config['number_aug'],
                    magnitude=config['magnitude'],
                    distance_sampler=config['sampling'],
                    val=config['val'],
                    num_classes=self.config['dataset']['num_classes'], 
                    net_type=self.net_type,
                    nb_clusters=self.nb_clusters,
                    bssampling=self.config['dataset']['bssampling'])

        # Pretraining dataloader
        else:
            self.running_corrects = 0
            self.denom = 0
            self.dl_tr = data_utility.create_loaders(size_batch=64,
                                                     data_root=self.args.cub_root,
                                                     num_workers=self.args.nb_workers,
                                                     mode=mode,
                                                     trans=self.args.trans,
                                                     pretraining=self.args.pretraining)
