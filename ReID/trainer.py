import os.path as osp
import logging
import random

import net
from RAdam import RAdam
from collections import defaultdict
import torch.nn as nn
from utils import losses 
import torch
from apex import amp
import random
import data_utility
from evaluation import Evaluator
import os
from torch import autograd
import numpy as np
#from evaluation.utils import visualize_att_map
from torch.utils.tensorboard import SummaryWriter
autograd.set_detect_anomaly(True)

logger = logging.getLogger('ReID.Training')

torch.manual_seed(0)


class Trainer():
    def __init__(self, config, save_folder_nets, save_folder_results,
                 device, timer, write=True):
        
        self.make_deterministic()
        self.update_configs = list()

        self.config = config

        self.device = device
        self.save_folder_results = save_folder_results
        self.save_folder_nets = save_folder_nets
        self.timer = timer
        # file name
        self.file_name = self.config['dataset'][
                      'dataset_short'] + '_intermediate_model_' + str(
            timer)
        self.net_type = self.config['models']['encoder_params']['net_type']
        self.dataset_short = self.config['dataset']['dataset_short']

        self.best_hypers = None
        self.num_iter = 30 if 'hyper' in config['mode'].split('_') else 1
        self.write = write
        logger.info("Writing and saving weights {}".format(write))

    def make_deterministic(self, seed=1):
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

        if self.write and 'hyper' not in self.config['mode'].split('_'):
            pretrained = str(
                self.model_params['encoder_params']['pretrained_path'] != 'no')
            # tensorboard path
            path = 'runs/' + "_".join([
                str(self.model_params['freeze_bb']),
                self.config['dataset']['corrupt'],
                self.train_params['loss_fn']['fns'],
                pretrained,
                self.train_params['output_train_enc'],
                self.train_params['output_train_gnn'],
                self.config['eval_params']['output_test_enc'],
                self.config['eval_params']['output_test_gnn'],
                self.model_params['encoder_params']['pool'],
                self.config['mode'],
                str(self.model_params['encoder_params']['red']),
                str(self.model_params['encoder_params']['neck'])])

            self.writer = SummaryWriter(path) # + str(i))
            logger.info("Writing to {}.".format(path))  

        logger.info('Search iteration {}'.format(i + 1))

        # sample parameter for hyper search
        self.update_params()

        if 'hyper' not in self.config['mode'].split('_'):
            self.make_deterministic(random.randint(0, 100))

        logger.info(self.config)
        logger.info(self.timer)
        self.device = 0
        
    def train(self):
        # initialize rank and hypers
        best_rank = 0
        best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])

        for i in range(self.num_iter):
            # init writer, make det det save name and update params
            self.init_iter(i)
            self.corrupt = self.config['dataset']['corrupt'] != 'no'

            # get models
            encoder, sz_embed = net.load_net(
                self.config['dataset']['dataset_short'],
                self.config['dataset']['num_classes'],
                self.config['mode'],
                self.model_params['attention'],
                add_distractors=self.config['dataset']['add_distractors'],
                **self.model_params['encoder_params'])
            self.encoder = encoder.cuda(self.device)  # to(self.device)
            param_groups = [{'params': list(set(self.encoder.parameters())),
                            'lr': self.train_params['lr']}]

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
                [self.encoder], self.opt = amp.initialize(
                    [self.encoder], self.opt,
                    opt_level="O1")

            # do paralel training in case there are more than one gpus
            if torch.cuda.device_count() > 1:
                self.encoder = nn.DataParallel(self.encoder)

            # execute training
            best_rank_iter, _ = self.execute(
                self.train_params,
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Rank-1: {}'.format(best_rank_iter))

            # save best model of iterations
            if best_rank_iter > best_rank and self.write:
                os.rename(osp.join(self.save_folder_nets, self.file_name + '.pth'),
                          str(best_rank_iter) + mode + self.net_type + '_' +
                          self.dataset_short + '.pth')

                best_rank = best_rank_iter
                
                best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])
            
            if self.write and 'hyper' not in self.config['mode'].split('_'):
                self.writer.close()

        logger.info("Best Hyperparameters found: " + best_hypers)
        logger.info(
            "Achieved {} with this hyperparameters".format(best_rank))
        logger.info("-----------------------------------------------------\n")

    def execute(self, train_params, eval_params):
        best_rank_iter = 0
        scores = list()
        
        if 'test' in self.config['mode'].split('_'):
            best_rank_iter = self.evaluate(scores, 0, 0)
        else:

            best_rank_iter = self.evaluate(
                scores, 0, best_rank_iter)

            for e in range(1, train_params['num_epochs'] + 1):
                logger.info(
                    'Epoch {}/{}'.format(e, train_params['num_epochs']))

                self.milestones(e, train_params, logger)
                
                for batch_num, (x, Y, I, P) in enumerate(self.dl_tr):
                    loss = self.forward_pass(x, Y, train_params, e)

                    if batch_num % 100 == 0:
                        logger.info("Iteration {}/{}: {}".format(
                            batch_num,
                            len(self.dl_tr),
                            {k: l[-1] for k, l in self.losses.items()}))

                    if train_params['is_apex']:
                        with amp.scale_loss(loss, self.opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    self.opt.step()
                
                best_rank_iter = self.evaluate(
                    scores, e, best_rank_iter)

                self.log()
                # compute ranks and mAP at the end of each epoch
                if e > 5 and best_rank_iter < 0.1 and self.num_iter > 1:
                    break

        return best_rank_iter, self.encoder

    def forward_pass(self, x, Y, train_params, e):
        Y = Y.cuda(self.device)
        self.opt.zero_grad()

        # bb for training with distractors (same bb but different output)
        if self.config['dataset']['add_distractors']:
            probs, fc7, distractor_bce = self.encoder(
                x.cuda(self.device),
                output_option=train_params['output_train_enc'])
        # normal training (same bb but different output)
        else:
            probs, fc7 = self.encoder(
                x.cuda(self.device),
                output_option=train_params['output_train_enc'])

        self.losses['Classification Accuracy'].append((torch.argmax(
            probs, dim=1) == Y).float().mean())

        # Compute CE Loss
        loss = 0
        if self.ce:
            if type(probs) != list:
                probs = [probs]

            for i, p in enumerate(probs):
                # -2 is the distractor label
                p = p[Y != -2]
                _Y = Y[Y != -2]
                loss0 = self.ce(p/self.train_params['temperatur'], _Y)
                loss += train_params['loss_fn']['scaling_ce'] * loss0
                self.losses['Cross Entropy ' + str(i)].append(loss0.item())

        if self.bce_distractor:
            # -2 again is distractor label
            bce_lab = torch.zeros(Y.shape).to(self.device)
            bce_lab[Y != -2] = 1
            distrloss = self.bce_distractor(
                distractor_bce.squeeze(), bce_lab.squeeze())
            loss += train_params['loss_fn']['scaling_bce'] * distrloss
            self.losses['BCE Loss Distractors ' + str(i)].append(
                distrloss.item())

        # Compute Triplet Loss
        if self.triplet:
            _fc7 = fc7[Y != -2]
            _Y = Y[Y != -2]
            triploss, _ = self.triplet(_fc7, _Y)
            loss += train_params['loss_fn']['scaling_triplet'] * triploss
            self.losses['Triplet'].append(triploss.item())

        if self.write and 'hyper' not in self.config['mode'].split('_'):
            for k, v in self.losses.items():
                self.writer.add_scalar('Loss/train/' + k, v[-1], e)

        return loss

    def evaluate(self, scores=None, e=None, best_rank_iter=None):
        with torch.no_grad():
            logger.info('EVALUATION ...')
            if self.config['mode'] in ['train', 'test', 'hyper_search']:
                mAP, top = self.evaluator.evaluate(
                    self.encoder,
                    self.dl_ev,
                    self.query,
                    gallery=self.gallery,
                    add_dist=self.config['dataset']['add_distractors'])

            logger.info('Mean AP: {:4.1%}'.format(mAP))
            logger.info('CMC Scores{:>12}'.format('Market'))

            for k in (1, 5, 10):
                logger.info('  top-{:<4}{:12.1%}'.format(k, top['Market'][k - 1]))

            if self.write and 'hyper' not in self.config['mode'].split('_'):
                logger.info("write evaluation")
                res = [
                    top['Market'][0],
                    top['Market'][4],
                    top['Market'][9],
                    mAP]
                names = ['rank-1', 'rank-5', 'rank-10', 'mAP']
                for t, rank in zip(res, names):
                    self.writer.add_scalar(
                        'Accuracy/test/' + str(rank), t, e)

            scores.append(
                (mAP, [top['Market'][k - 1] for k in [1, 5, 10]]))
            rank = top['Market'][0]
            
            self.encoder.current_epoch = e
            if self.train_params['store_every'] or rank > best_rank_iter:
                best_rank_iter = rank
                if self.write and 'hyper' not in self.config['mode'].split('_'):
                    torch.save(
                        self.encoder.state_dict(),
                        osp.join(
                            self.save_folder_nets, self.file_name + '.pth'))

        return best_rank_iter

    def get_loss_fn(self, params, num_classes, add_distractors=False):
        self.losses = defaultdict(list)
        self.losses_mean = defaultdict(list)

        if add_distractors:
            self.bce_distractor = nn.BCELoss()
        else:
            self.bce_distractor = None

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
        
        # Add triplet loss
        if 'triplet' in params['fns'].split('_'):
            self.triplet = losses.TripletLoss(margin=0.3).cuda(self.gnn_dev)
        else:
            self.triplet = None

    def update_params(self):
        self.sample_hypers() if 'hyper' in self.config['mode'].split('_') else None

        if self.config['dataset']['val']:
            self.config['dataset']['num_classes'] -= 100

        if 'test' in self.config['mode'].split('_'):
            self.train_params['num_epochs'] = 1

        if self.config['dataset']['sampling'] != 'no':
            self.train_params['num_epochs'] += 10

    def sample_hypers(self):
        # sample hypers for all iters before training bc of deterministic
        if not len(self.update_configs):
            for _ in range(self.num_iter):
                config = dict()
                config['train'] = {
                    'lr': 10 ** random.uniform(-8, -2),
                    'weight_decay': 10 ** random.uniform(-15, -6),
                    'num_classes_iter': random.randint(6, 9),  # 100
                    'num_elements_class': random.randint(3, 4),
                    'temperatur': random.random(),
                    'num_epochs': 30}
                self.update_configs.append(config)
            self.iter_i = 0

        self.train_params.update(self.update_configs[self.iter_i]['train'])

        logger.info("Updated Hyperparameters:")
        logger.info(self.config)
        self.iter_i += 1

    def get_data(self, config, train_params, mode):
        # If distance sampling
        self.dl_tr, self.dl_ev, self.query, self.gallery, self.dl_ev_gnn = data_utility.create_loaders(
                data_root=config['dataset_path'],
                num_workers=config['nb_workers'],
                num_classes_iter=train_params['num_classes_iter'],
                num_elements_class=train_params['num_elements_class'],
                trans=config['trans'],
                bssampling=self.config['dataset']['bssampling'],
                rand_scales=config['rand_scales'],
                add_distractors=config['add_distractors'],
                split=config['split'],
                sz_crop=config['sz_crop'])
            
        self.config['dataset']['num_classes'] = len(
            set(self.dl_tr.dataset.ys)) - 1

    def milestones(self, e, train_params, logger):
        if e in train_params['milestones']:
            logger.info("reduce learning rate")
            if self.write:
                self.encoder.load_state_dict(torch.load(
                    osp.join(self.save_folder_nets, self.file_name + '.pth')))
            else:
                logger.info("not loading weights as self.write = False")

            for g in self.opt.param_groups:
                g['lr'] = g['lr'] / 10.

    def log(self):
        [self.losses_mean[k].append(sum(v) / len(v)) for k, v in
            self.losses.items()]
        self.losses = defaultdict(list)
        logger.info('Loss Values: ')
        logger.info(', '.join([
            str(k) + ': ' + str(v[-1]) for k, v in self.losses_mean.items()]))
