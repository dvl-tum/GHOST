import os.path as osp
import logging
import random

import net
from utils.RAdam import RAdam
from collections import defaultdict
import torch.nn as nn
from utils import losses 
import torch
from apex import amp
import random
import data.data_utility as data_utility
from evaluation import Evaluator
import os
from torch import autograd
import numpy as np
from utils.utils import args
#from evaluation.utils import visualize_att_map
from torch.utils.tensorboard import SummaryWriter
autograd.set_detect_anomaly(True)

torch.manual_seed(0)


logger = logging.getLogger('ReID')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

os.makedirs('log_out', exist_ok=True)
fh = logging.FileHandler(os.path.join('log_out', 'train_reid.txt'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


class Trainer():
    def __init__(self, config, save_folder_nets, save_folder_results,
                 device, timer, write=True):
        
        # configs
        self.config = config
        self.train_config = args(self.config['train_params'])
        self.dataset_config = args(self.config['dataset'])
        self.model_config = args(self.config['encoder'])
        self.eval_config = args(self.config['eval_params'])

        self.device = device
        logger.info('Switching to device {}...'.format(device))
        self.save_folder_results = save_folder_results
        self.save_folder_nets = save_folder_nets

        # file name for intermediate results
        self.file_name = self.dataset_config.dataset_short \
            + '_intermediate_model_' + str(timer)
        
        # init variables
        self.update_configs = list()
        self.net_type = self.model_config.net_type
        self.dataset_short = self.dataset_config.dataset_short
        self.mode = self.config['mode']
        self.hyper = 'hyper' in self.mode
        self.num_iter = 30 if self.hyper else 1
        self.test = 'test' in self.mode
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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
    
    def init_iter(self, i):
        logger.info(f'Search iteration {i + 1}')
        # init tensorboard
        if self.write:
            pretrained = str(
                self.model_config.pretrained_path != 'no')
            # tensorboard path
            path = 'runs/' + "_".join([
                f"loss:{self.train_config.loss_fn['fns']}_",
                f"pre:{pretrained}_",
                f"outtrain{self.train_config.output_train_enc}_",
                f"outval:{self.eval_config.output_test_enc}_",
                f"pool:{self.model_config.pool}_",
                f"mode:{self.mode}_",
                f"red:{str(self.model_config.red)}_",
                f"neck:{str(self.model_config.neck)}_",
                f"iteration:{str(i)}"])

            self.writer = SummaryWriter(path) # + str(i))
            logger.info(f"Tensorboard pth {path}.....")

        # sample parameter for hyper search
        self.update_params()

        # make deterministic
        self.make_deterministic(random.randint(0, 100))

        # log updated config and time
        logger.info("")
        for ddict in [self.dataset_config, self.model_config, self.train_config, self.eval_config]:
            for k, v in ddict.__dict__.items():
                logger.info(f"{k}: {v}")
        logger.info("")
        
        self.device = 0
        
    def train(self):
        # initialize rank and hypers
        best_rank = 0

        for i in range(self.num_iter):
            # init writer, make det det save name and update params
            self.init_iter(i)

            # get models
            encoder, sz_embed, optimizer_state_dict = net.load_net(
                self.dataset_config.num_classes,
                **self.model_config.__dict__)
            self.encoder = encoder.cuda(self.device)  # to(self.device)
            param_groups = [{'params': list(set(self.encoder.parameters())),
                            'lr': self.train_config.lr}]
            logger.info(f"Dimension of Resnet output {sz_embed}")
            
            # get data
            self.get_data()
            
            # get evaluator
            self.evaluator = Evaluator(**self.eval_config.__dict__)
            
            self.opt = RAdam(param_groups,
                             weight_decay=self.train_config.weight_decay)
            
            if optimizer_state_dict is not None:
                self.opt.load_state_dict(optimizer_state_dict)
            
            # get loss functions
            self.get_loss_fn(self.train_config.loss_fn,
                             self.dataset_config.num_classes,
                             self.dataset_config.add_distractors)

            # Do training in mixed precision
            if self.train_config.is_apex:
                [self.encoder], self.opt = amp.initialize(
                    [self.encoder], self.opt,
                    opt_level="O1")

            # do paralel training in case there are more than one gpus
            if torch.cuda.device_count() > 1:
                self.encoder = nn.DataParallel(self.encoder)

            # execute training
            best_rank_iter, _ = self.execute()

            # log best reuslts and parameters of run
            hypers = self.combine_hypers()
            logger.info('Used Parameters: ' + hypers)
            logger.info('Best Rank-1: {}'.format(best_rank_iter))

            # save model if better than other models of previous iterations 
            # (hyper search) or just save model
            if best_rank_iter > best_rank:
                os.rename(osp.join(self.save_folder_nets, self.file_name + '.pth'),
                            str(best_rank_iter) + self.net_type + '_' +
                            self.dataset_short + '.pth')

                best_rank = best_rank_iter
                best_hypers = self.combine_hypers()
            
            if self.write:
                self.writer.close()

        if self.hyper:
            logger.info("Best Hyperparameters found: \n")
            for k, v in best_hypers:
                logger.info(f"{k}: {v}")
            logger.info(
                "Achieved {} with this hyperparameters...".format(best_rank))
    
    def combine_hypers(self):
        return ', '.join(
                [k + ': ' + str(v) for k, v in self.train_config.items()] + \
                    [k + ': ' + str(v) for k, v in self.dataset_config.items()] + \
                        [k + ': ' + str(v) for k, v in self.model_config.items()] + \
                            [k + ': ' + str(v) for k, v in self.eval_config.items()])

    def execute(self):
        best_rank_iter = 0
        scores = list()
        
        best_rank_iter = self.evaluate(scores, 0, best_rank_iter)

        if not self.test:
            for e in range(1, self.train_config.num_epochs + 1):
                logger.info(
                    'Epoch {}/{}'.format(e, self.train_config.num_epochs))

                self.milestones(e)
                
                for batch_num, (x, Y, I, P) in enumerate(self.dl_tr):
                    loss = self.forward_pass(x, Y, e)

                    '''if batch_num % 100 == 0:
                        last_loss = {k: l[-1] for k, l in self.losses.items()}
                        logger.info(f"Batch {batch_num}/{len(self.dl_tr)}: {last_loss}")'''

                    if self.train_config.is_apex:
                        with amp.scale_loss(loss, self.opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    self.opt.step()
                
                best_rank_iter = self.evaluate(
                    scores, e, best_rank_iter)

                self.log()

                # break training if hyperparameter search and low rank
                if e > 5 and best_rank_iter < 0.1 and self.hyper:
                    break

        return best_rank_iter, self.encoder

    def forward_pass(self, x, Y, e):
        Y = Y.cuda(self.device)
        self.opt.zero_grad()

        # bb for training with distractors (same bb but different output)
        if self.dataset_config.add_distractors:
            probs, fc7, distractor_bce = self.encoder(
                x.cuda(self.device),
                output_option=self.train_config.output_train_enc)

        # normal training (same bb but different output)
        else:
            probs, fc7 = self.encoder(
                x.cuda(self.device),
                output_option=self.train_config.output_train_enc)

        self.losses['Classification Accuracy'].append((torch.argmax(
            probs, dim=1) == Y).float().mean())

        # Compute CE Loss
        loss = 0
        if self.ce:
            # -2 is the distractor label
            probs = probs[Y != -2]
            _Y = Y[Y != -2]
            loss0 = self.ce(probs/self.train_config.temperatur, _Y)
            loss += self.train_config.loss_fn['scaling_ce'] * loss0
            self.losses['Cross Entropy'].append(loss0.item())

        if self.bce_distractor:
            # -2 again is distractor label
            bce_lab = torch.zeros(Y.shape).to(self.device)
            bce_lab[Y != -2] = 1
            distrloss = self.bce_distractor(
                distractor_bce.squeeze(), bce_lab.squeeze())
            loss += self.train_config.loss_fn['scaling_bce'] * distrloss
            self.losses['BCE Loss Distractors'].append(
                distrloss.item())

        # Compute Triplet Loss
        if self.triplet:
            # -2 again is distractor label
            _fc7 = fc7[Y != -2]
            _Y = Y[Y != -2]
            triploss, _ = self.triplet(_fc7, _Y)
            loss += self.train_config.loss_fn['scaling_triplet'] * triploss
            self.losses['Triplet'].append(triploss.item())

        if self.write:
            for k, v in self.losses.items():
                self.writer.add_scalar('Loss/train/' + k, v[-1], e)

        return loss

    def evaluate(self, scores=None, e=None, best_rank_iter=None):
        with torch.no_grad():
            logger.info('EVALUATION ...')
            mAP, top = self.evaluator.evaluate(
                self.encoder,
                self.dl_ev,
                self.query,
                gallery=self.gallery,
                add_dist=self.dataset_config.add_distractors)

            # log map and rank-k values
            logger.info('Mean AP: {:4.1%}'.format(mAP))
            logger.info('CMC Scores{:>12}'.format('Market'))

            for k in (1, 5, 10):
                logger.info('  top-{:<4}{:12.1%}'.format(k, top['Market'][k - 1]))

            if self.write:
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
            
            # store model if better than previous or every epoch if store every
            self.encoder.current_epoch = e
            if self.train_config.store_every or rank > best_rank_iter:
                best_rank_iter = rank
                torch.save({
                    'model_state_dict': self.encoder.state_dict(),
                    'epoch': self.encoder.current_epoch,
                    'optimizer_state_dict': self.opt.state_dict()},
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
                num_classes=num_classes, dev=self.device).cuda(self.device)
        elif 'focalce' in params['fns'].split('_'):
            self.ce = losses.FocalLoss().cuda(self.device)
        elif 'ce' in params['fns'].split('_'):
            self.ce = nn.CrossEntropyLoss().cuda(self.device)
        else:
            self.ce = None
        
        # Add triplet loss
        if 'triplet' in params['fns'].split('_'):
            self.triplet = losses.TripletLoss(margin=0.3).cuda(self.device)
        else:
            self.triplet = None

    def update_params(self):
        if self.hyper:
            self.sample_hypers()

        if self.test:
            self.train_config.num_epochs = 1

    def sample_hypers(self):
        # sample hypers for all iters before training bc of deterministic
        if not len(self.update_configs):
            self.iter_i = 0
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

        # take samples of current iteration
        self.train_config.__dict__.update(
            self.update_configs[self.iter_i]['train'])

        logger.info("Updated Hyperparameters:")
        self.iter_i += 1

    def get_data(self):
        # If distance sampling
        self.dl_tr, self.dl_ev, self.query, self.gallery = data_utility.create_loaders(
                dataset_config=self.dataset_config,
                num_classes_iter=self.train_config.num_classes_iter,
                num_elements_class=self.train_config.num_elements_class)
        
        if self.dataset_config.add_distractors:
            self.dataset_config.num_classes = len(
                set(self.dl_tr.dataset.ys)) - 1

    def milestones(self, e):
        if e in self.train_config.milestones:
            logger.info("Reduce learning rate...")

            # Load best weights so far
            self.encoder.load_state_dict(torch.load(
                osp.join(self.save_folder_nets, self.file_name + '.pth')))

            # divide learning rate by 10
            for g in self.opt.param_groups:
                g['lr'] = g['lr'] * self.train_config.lr_reduction

    def log(self):
        for k, v in self.losses.items():
            self.losses_mean[k].append(sum(v) / len(v))
        self.losses = defaultdict(list)
        logger.info('Loss Values: ')
        logger.info(', '.join([
            str(k) + ': ' + str(v[-1]) for k, v in self.losses_mean.items()]))
