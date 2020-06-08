from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import time
import os
import warnings
from apex import amp
import argparse
import random
import torch.nn.functional as F
import sys
import logging
from collections import defaultdict
from RAdam import RAdam
import gtg as gtg_module
import net
import data_utility
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

fh = logging.FileHandler('train_reid.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

warnings.filterwarnings("ignore")


class Hyperparameters():
    def __init__(self, dataset_name='cub'):
        self.dataset_name = dataset_name
        print(self.dataset_name)
        # print('Without GL')
        if dataset_name == 'Market':
            self.dataset_path = '../../datasets/Market-1501-v15.09.15'
            self.dataset_short = 'Market'
        elif dataset_name == 'cuhk03-detected':
            self.dataset_path = '../../datasets/cuhk03/detected'
            self.dataset_short = 'cuhk03'
        elif dataset_name == 'cuhk03-labeled':
            self.dataset_path = '../../datasets/cuhk03/labeled'
            self.dataset_short = 'cuhk03'
        elif dataset_name == 'cuhk03-np-detected':
            self.dataset_path = '../../datasets/cuhk03-np/labeled'
            self.dataset_short = 'cuhk03-np'
        elif dataset_name == 'cuhk03-np-labeled':
            self.dataset_path = '../../datasets/cuhk03-np/labeled'
            self.dataset_short = 'cuhk03-np'

        self.num_classes = {'Market': 751,
                            'cuhk03-detected': 1367,
                            'cuhk03-np-detected': 767,
                            'cuhk03-labeled': 1367,
                            'cuhk03-np-labeled': 767}
        self.num_classes_iteration = {'Market': 3,
                                      'cuhk03-detected': 5,
                                      'cuhk03-np-detected': 5,
                                      'cuhk03-labeled': 5,
                                      'cuhk03-np-labeled': 5}
        self.num_elemens_class = {'Market': 4,
                                  'cuhk03-detected': 5,
                                  'cuhk03-np-detected': 5,
                                  'cuhk03-labeled': 5,
                                  'cuhk03-np-labeled': 5}
        self.get_num_labeled_class = {'Market': 3,
                                      'cuhk03-detected': 2,
                                      'cuhk03-np-detected': 2,
                                      'cuhk03-labeled': 2,
                                      'cuhk03-np-labeled': 2}
        self.learning_rate = {'Market': 1.289377564403867e-05,
                              'cuhk03-detected': 4.4819286767613e-05,
                              'cuhk03-np-detected': 0.0002,
                              'cuhk03-labeled': 0.0002,
                              'cuhk03-np-labeled': 0.0002}
        self.weight_decay = {'Market': 1.9250447877921047e-14,
                             'cuhk03-detected': 1.5288509425482333e-13,
                             'cuhk03-np-detected': 4.863656728256105e-07,
                             'cuhk03-labeled': 4.863656728256105e-07,
                             'cuhk03-np-labeled': 4.863656728256105e-07}
        self.softmax_temperature = {'Market': 80,
                                    'cuhk03-detected': 80,
                                    'cuhk03-np-detected': 80,
                                    'cuhk03-labeled': 80,
                                    'cuhk03-np-labeled': 80}
        self.num_iter_gtg = {'Market': 2,
                             'cuhk03-detected': 1,
                             'cuhk03-np-detected': 1,
                             'cuhk03-labeled': 1,
                             'cuhk03-np-labeled': 1}

    def get_path(self):
        return self.dataset_path

    def get_number_classes(self):
        return self.num_classes[self.dataset_name]

    def get_number_classes_iteration(self):
        return self.num_classes_iteration[self.dataset_name]

    def get_number_elements_class(self):
        return self.num_elemens_class[self.dataset_name]

    def get_number_labeled_elements_class(self):
        return self.get_num_labeled_class[self.dataset_name]

    def get_learning_rate(self):
        return self.learning_rate[self.dataset_name]

    def get_weight_decay(self):
        return self.weight_decay[self.dataset_name]

    def get_epochs(self):
        return 70

    def get_num_gtg_iterations(self):
        return self.num_iter_gtg[self.dataset_name]

    def get_softmax_temperature(self):
        return self.softmax_temperature[self.dataset_name]


def init_args():
    dataset = 'cuhk03-detected'
    hyperparams = Hyperparameters(dataset)
    parser = argparse.ArgumentParser(
        description='Pretraining for Person Re-ID with Group Loss')
    parser.add_argument('--dataset_name', default=hyperparams.dataset_name,
                        type=str,
                        help='The name of the dataset')
    parser.add_argument('--dataset_short', default=hyperparams.dataset_short,
                        type=str,
                        help='without detected/labeled')
    parser.add_argument('--oversampling', default=1, type=int,
                        help='If oversampling shoulf be used')
    parser.add_argument('--nb_epochs', default=30, type=int)

    parser.add_argument('--cub-root', default=hyperparams.get_path(),
                        help='Path to dataset folder')
    parser.add_argument('--cub-is-extracted', action='store_true',
                        default=True,
                        help='If `images.tgz` was already extracted, do not extract it again.' +
                             ' Otherwise use extracted data.')
    parser.add_argument('--embedding-size', default=512, type=int,
                        dest='sz_embedding', help='The embedding size')
    parser.add_argument('--nb_classes',
                        default=hyperparams.get_number_classes(), type=int,
                        help='Number of first [0, N] classes used for training and ' +
                             'next [N, N * 2] classes used for evaluating with max(N) = 100.')
    parser.add_argument('--pretraining', default=0, type=int,
                        help='If pretraining or fine tuning is executed')
    parser.add_argument('--num_classes_iter',
                        default=hyperparams.get_number_classes_iteration(),
                        type=int,
                        help='Number of classes in the minibatch')
    parser.add_argument('--num_elements_class',
                        default=hyperparams.get_number_elements_class(),
                        type=int,
                        help='Number of samples per each class')
    parser.add_argument('--num_labeled_points_class',
                        default=hyperparams.get_number_labeled_elements_class(),
                        type=int,
                        help='Number of labeled samples per each class')
    parser.add_argument('--lr-net', default=hyperparams.get_learning_rate(),
                        type=float, help='The learning rate')
    parser.add_argument('--weight-decay',
                        default=hyperparams.get_weight_decay(), type=float,
                        help='The l2 regularization strength')
    parser.add_argument('--temperature',
                        default=hyperparams.get_softmax_temperature(),
                        help='Temperature parameter for the softmax')
    parser.add_argument('--nb_workers', default=4, type=int,
                        help='Number of workers for dataloader.')
    parser.add_argument('--net_type', default='resnet50', type=str,
                        choices=['bn_inception', 'densenet121', 'densenet161',
                                 'densenet169', 'densenet201',
                                 'resnet18', 'resnet34', 'resenet50',
                                 'resnet101', 'resnet152'],
                        help='The type of net we want to use')
    parser.add_argument('--sim_type', default='correlation', type=str,
                        help='type of similarity we want to use')
    parser.add_argument('--set_negative', default='hard', type=str,
                        help='type of threshold we want to do'
                             'hard - put all negative similarities to 0'
                             'soft - subtract minus (-) negative from each entry')
    parser.add_argument('--num_iter_gtg',
                        default=hyperparams.get_num_gtg_iterations(), type=int,
                        help='Number of iterations we want to do for GTG')
    parser.add_argument('--embed', default=0, type=int,
                        help='boolean controling if we want to do embedding or not')
    parser.add_argument('--scaling_loss', default=1, type=int,
                        dest='scaling_loss',
                        help='Scaling parameter for the loss')
    parser.add_argument('--decrease_learning_rate', default=10., type=float,
                        help='Number to divide the learnign rate with')
    parser.add_argument('--id', default=1, type=int,
                        help='id, in case you run multiple independent nets, for example if you want an ensemble of nets')
    parser.add_argument('--is_apex', default=1, type=int,
                        help='if 1 use apex to do mixed precision training')
    parser.add_argument('--both', default=0, type=int,
                        help='if labeled and detected of cuhk03 should be taken')
    parser.add_argument('--lab_smooth', default=1, type=int,
                        help='if label smoothing should be applied')
    parser.add_argument('--trans', default='bot', type=str,
                        help='wich augmentation shoulb be performed: '
                             'norm, bot, imgaug')
    parser.add_argument('--last_stride', default=1, type=int,
                        help='If last stride should be changed to 1')
    parser.add_argument('--neck', default=1, type=int,
                        help='if additional batchnorm layer should be added')
    parser.add_argument('--center', default=1, type=int,
                        help='if center loss should be added')
    parser.add_argument('--test_option', default='norm', type=str,
                        help='If features after BN or before should be taken for test if neck is enabled, '
                        'if plain normalization should be applied when neck is not enabeled: neck/norm, plain/norm .')
    parser.add_argument('--early_thresh', default=20, type=int,
                        help='threshold when to stop, i.e. after 7 epochs, '
                             'where best recall did not improve')
    parser.add_argument('--test', default=0, type=int, 
                        help='If net should only be tested, not trained')
    parser.add_argument('--use_pretrained', default=1, type=int,
                        help='If weights pretrained for 10 epochs, lr=0.0002 using cross entropy should be used')
    parser.add_argument('--bn_GL', default=0, type=int, 
                        help='if features after batchnorm layer should be used for group loss')
    parser.add_argument('--hyper_search', default=0, type=int,
                        help='If hyper parameter search is done or not')
    parser.add_argument('--distance_sampling', default=0, type=int,
                        help='if distance sampling should be applied')


    return parser.parse_args()


class PreTrainer():
    def __init__(self, args, data_dir, device, save_folder_results,
                 save_folder_nets):
        self.device = device
        self.data_dir = data_dir
        self.args = args
        self.save_folder_results = save_folder_results
        self.save_folder_nets = save_folder_nets
    
    def train_model(self, config, timer, load_path):
        print(self.args)
        early_thresh_counter = 0

        file_name = self.args.dataset_name + '_intermediate_model_' + str(
            timer)
        # add last stride and bottleneck
        model = net.load_net(dataset=self.args.dataset_short,
                             net_type=self.args.net_type,
                             nb_classes=self.args.nb_classes,
                             embed=self.args.embed,
                             sz_embedding=self.args.sz_embedding,
                             pretraining=self.args.pretraining,
                             last_stride=self.args.last_stride,
                             neck=self.args.neck, 
                             load_path=load_path,
                             use_pretrained=self.args.use_pretrained, 
                             bn_GL=self.args.bn_GL)
        model = model.to(self.device)

        gtg = gtg_module.GTG(self.args.nb_classes,
                             max_iter=config['num_iter_gtg'],
                             sim=self.args.sim_type,
                             set_negative=self.args.set_negative,
                             device=self.device).to(self.device)
        opt = RAdam(
            [{'params': list(set(model.parameters())), 'lr': config['lr']}],
            weight_decay=config['weight_decay'])
        criterion = nn.NLLLoss().to(self.device)

        # add label smoothing
        if self.args.lab_smooth:
            criterion2 = utils.CrossEntropyLabelSmooth(
                num_classes=self.args.nb_classes)
        else:
            criterion2 = nn.CrossEntropyLoss().to(self.device)

        # add center loss
        if self.args.center:
            criterion3 = utils.CenterLoss(num_classes=self.args.nb_classes)

        # do training in mixed precision
        if self.args.is_apex:
            model, opt = amp.initialize(model, opt, opt_level="O1")

        # add bag of trick transformation
        if not self.args.pretraining:
            if self.args.distance_sampling:
                dl_tr2, dl_ev2, query2, gallery2 = data_utility.create_loaders(
                    data_root=self.args.cub_root,
                    num_workers=self.args.nb_workers,
                    num_classes_iter=config['num_classes_iter'],
                    num_elements_class=config['num_elements_class'],
                    size_batch=config['num_classes_iter'] * config[
                        'num_elements_class'],
                    both=self.args.both,
                    trans=self.args.trans,
                    distance_sampler=self.args.distance_sampling)
                dl_tr1, dl_ev1, query1, gallery1 = data_utility.create_loaders(
                    data_root=self.args.cub_root,
                    num_workers=self.args.nb_workers,
                    num_classes_iter=config['num_classes_iter'],
                    num_elements_class=config['num_elements_class'],
                    size_batch=config['num_classes_iter'] * config[
                        'num_elements_class'],
                    both=self.args.both,
                    trans=self.args.trans,
                    distance_sampler=0)
            else:
                dl_tr, dl_ev, query, gallery = data_utility.create_loaders(
                    data_root=self.args.cub_root,
                    num_workers=self.args.nb_workers,
                    num_classes_iter=config['num_classes_iter'],
                    num_elements_class=config['num_elements_class'],
                    size_batch=config['num_classes_iter'] * config[
                        'num_elements_class'],
                    both=self.args.both,
                    trans=self.args.trans,
                    distance_sampler=0)
        else:
            running_corrects = 0
            denom = 0
            dl_tr = data_utility.create_loaders(size_batch=64,
                                                data_root=self.args.cub_root,
                                                num_workers=self.args.nb_workers,
                                                both=self.args.both,
                                                trans=self.args.trans,
                                                pretraining=self.args.pretraining)

        since = time.time()
        best_accuracy = 0
        if self.args.distance_sampling:
            feature_dict = defaultdict()
        scores = []
        for e in range(1, self.args.nb_epochs + 1):
            if not self.args.test:
                logger.info('Epoch {}/{}'.format(e, self.args.nb_epochs))
                if e == 31:
                    model.load_state_dict(torch.load(
                        os.path.join(self.save_folder_nets, file_name + '.pth')))
                    for g in opt.param_groups:
                        g['lr'] = config['lr'] / 10.

                if e == 51:
                    model.load_state_dict(torch.load(
                        os.path.join(self.save_folder_nets, file_name + '.pth')))
                    for g in opt.param_groups:
                        g['lr'] = config['lr'] / 10.

                i = 0
                if self.args.distance_sampling:
                    dl_tr.feature_dict = feature_dict
                    feature_dict = defaultdict()

                # after 30 epochs use distance
                if self.args.distance_sampling:
                    if e > 30:
                        dl_tr = dl_tr2
                        dl_ev = dl_ev2
                        gallery = gallery2
                        query = query2
                    else:
                        dl_tr = dl_tr1
                        dl_ev = dl_ev1
                        gallery = gallery1
                        query = query1

                for x, Y in dl_tr:
                    Y = Y.to(self.device)
                    opt.zero_grad()

                    probs, fc7 = model(x.to(self.device))
                    if self.args.distance_sampling:
                        for y, f in zip(Y, fc7):
                            feature_dict[y.data.item()].append(f)
                    loss = criterion2(probs, Y)

                    if not self.args.pretraining:
                        labs, L, U = data_utility.get_labeled_and_unlabeled_points(
                            labels=Y,
                            num_points_per_class=config[
                                'num_labeled_points_class'],
                            num_classes=self.args.nb_classes)
    
                        # compute the smoothed softmax
                        probs_for_gtg = F.softmax(probs / config['temperature'])
    
                        # do GTG (iterative process)
                        probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U,
                                               probs_for_gtg)
                        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

                        # compute the losses
                        loss1 = criterion(probs_for_gtg, Y)
                        loss = self.args.scaling_loss * loss1 + loss
                        # add center loss
                        if self.args.center:
                            loss += criterion3(fc7, Y)
    
                    else:
                        _, preds = torch.max(probs, 1)
                        denom += Y.shape[0]
                        running_corrects += torch.sum(
                            preds == Y.data).cpu().data.item()

                    i += 1
    
                    # check possible net divergence
                    if torch.isnan(loss):
                        logger.error("We have NaN numbers, closing\n\n\n")
                        sys.exit(0)

                    # backprop
                    if self.args.is_apex:
                        with amp.scale_loss(loss, opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    opt.step()

            # compute recall and NMI at the end of each epoch (for Stanford NMI takes forever so skip it)
            if not self.args.pretraining:
                with torch.no_grad():
                    logger.info('EVALUATION')
                    mAP, top = utils.evaluate_reid(model, dl_ev,
                                                   query=query,
                                                   gallery=gallery,
                                                   root=self.data_dir,
                                                   test_option=self.args.test_option)

                    logger.info('Mean AP: {:4.1%}'.format(mAP))

                    logger.info('CMC Scores{:>12}{:>12}{:>12}'
                                .format('allshots', 'cuhk03', 'Market'))
                    print('Mean AP: {:4.1%}'.format(mAP))

                    print('CMC Scores{:>12}{:>12}{:>12}'
                                .format('allshots', 'cuhk03', 'Market'))

                    for k in (1, 5, 10):
                        logger.info('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                                    .format(k, top['allshots'][k - 1],
                                            top['cuhk03'][k - 1],
                                            top['Market'][k - 1]))
                        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                                    .format(k, top['allshots'][k - 1],
                                            top['cuhk03'][k - 1],
                                            top['Market'][k - 1]))


                    scores.append((mAP,
                                   [top[self.args.dataset_short][k - 1] for k
                                    in
                                    [1, 5, 10]]))
                    model.current_epoch = e
                    if top[self.args.dataset_short][0] > best_accuracy:
                        best_accuracy = top[self.args.dataset_short][0]
                        torch.save(model.state_dict(),
                                   os.path.join(self.save_folder_nets,
                                                file_name + '.pth'))
                        early_thresh_counter = 0
                    elif early_thresh_counter >= self.args.early_thresh:
                        logger.info(
                            'Early stopping at epoch {} after no improvement for {} epochs'.format(
                                e, self.args.early_thresh))
                        break
                    else:
                        early_thresh_counter += 1

            else:
                logger.info(
                    'Loss {}, Accuracy {}'.format(torch.mean(loss.cpu()),
                                                  running_corrects / denom))
                print('Loss {}, Accuracy {}'.format(torch.mean(loss.cpu()),
                                                  running_corrects / denom))
                scores.append(running_corrects / denom)
                denom = 0
                running_corrects = 0
                if scores[-1] > best_accuracy:
                    best_accuracy = scores[-1]
                    torch.save(model.state_dict(),
                               os.path.join(self.save_folder_nets,
                                            file_name + '.pth'))

        end = time.time()
        logger.info(
            'Completed {} epochs in {}s on {}'.format(self.args.nb_epochs,
                                                      end - since,
                                                      self.args.dataset_name))

        file_name = str(
            best_accuracy) + '_' + self.args.dataset_name + '_' + str(
            self.args.id) + '_' + self.args.net_type + '_' + str(
            config['lr']) + '_' + str(config['weight_decay']) + '_' + str(
            config['num_classes_iter']) + '_' + str(
            config['num_elements_class']) + '_' + str(
            config['num_labeled_points_class'])
        if self.args.test:
            file_name = 'test_' + file_name 
        if not self.args.pretraining:
            with open(
                    os.path.join(self.save_folder_results, file_name + '.txt'),
                    'w') as fp:
                fp.write(file_name + "\n")
                fp.write(str(self.args))
                fp.write('\n')
                fp.write(str(config))
                fp.write('\n')
                fp.write('\n'.join('%s %s' % x for x in scores))
                fp.write("\n\n\n")

        return best_accuracy, model


def main():
    args = init_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timer = time.time()
    logger.info('Switching to device {}'.format(device))

    save_folder_results = 'search_results'
    save_folder_nets = 'search_results_net'
    if not os.path.isdir(save_folder_results):
        os.makedirs(save_folder_results)
    if not os.path.isdir(save_folder_nets):
        os.makedirs(save_folder_nets)
    
    trainer = PreTrainer(args, args.cub_root, device,
                         save_folder_results, save_folder_nets)

    best_recall = 0
    best_hypers = None

    if args.hyper_search:
        num_iter = 30
    else:
        num_iter = 1
    
    # NORM VS PLAIN
    lab_smooth = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    trans = ['bot', 'bot', 'bot', 'bot', 'norm', 'norm', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'norm', 'norm', 'bot', 'bot']
    neck = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    last_stride = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_option = ['norm', 'norm', 'neck', 'plain', 'plain', 'norm', 'plain', 'norm', 'norm', 'norm', 'neck', 'plain', 'plain', 'norm', 'plain', 'norm']
    bn_GL = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    center = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    use_pretrained = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
    
    # BOT EXPERIMENT
    lab_smooth = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    trans = ['norm', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot']
    neck = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    last_stride = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
    test_option = ['norm', 'norm', 'norm', 'norm', 'neck', 'neck', 'neck', 'neck', 'norm', 'norm']
    center = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]

    #appearance test
    lab_smooth = [1, 1, 1, 1, 1, 1]
    trans = ['appearance', 'appearance', 'appearance', 'appearance', 'appearance', 'appearance']
    neck = [1, 1, 0, 0, 1, 1]
    last_stride = [0, 0, 0, 0, 0, 0]
    test_option = ['norm', 'neck', 'norm', 'plain', 'norm', 'neck']
    center = [0, 0, 0, 0, 0, 0]
    bn_GL = [0, 0, 0, 0, 1, 1]

    #distance sampler
    lab_smooth = [0]
    trans = ['appearance']
    neck = [0]
    last_stride = [0]
    test_option = ['norm']
    center = [0]
    bn_GL = [0]
    distance_sampling = [1]
    
    print("EXPERIMENT: AFFINE BACK TO ONE DATA AUGMENTATION")

    # Random search
    for i in range(num_iter):
        trainer.args.lab_smooth = lab_smooth[i]
        trainer.args.trans = trans[i]
        trainer.args.neck = neck[i]
        trainer.args.last_stride = last_stride[i]
        trainer.args.test_option = test_option[i] #neck_test[i]
        trainer.args.bn_GL = bn_GL[i]
        trainer.args.center = center[i]
        trainer.args.use_pretrained = 0 #use_pretrained[i]
        trainer.args.distance_sampling = 1 #distance_sampling[i]
        print(trainer.args)
        if args.pretraining:
            mode = 'finetuned_'
        else:
            mode = ''
        if trainer.args.neck:
            mode = mode + 'neck_'

        if args.test:
            trainer.args.nb_epochs = 1
            load_path = os.path.join('save_trained_nets', mode + trainer.args.net_type + '_' + trainer.args.dataset_name + '.pth')
        else:
            load_path = os.path.join('net', 'finetuned_' + mode + trainer.args.dataset_short + '_' + trainer.args.net_type + '.pth')
        print(load_path)
        if args.use_pretrained:
            logger.info('Load model from {}'.format(load_path))
        else:
            logger.info('Using model only pretrained on ImageNet')

        logger.info('Search iteration {}'.format(i + 1))

        if args.hyper_search:
            config = {'lr': 10 ** random.uniform(-8, -3),
                      'weight_decay': 10 ** random.uniform(-15, -6),
                      'num_classes_iter': random.randint(2, 5),
                      'num_elements_class': random.randint(4, 9),
                      'num_labeled_points_class': random.randint(1, 3),
                      'num_iter_gtg': random.randint(1, 3),
                      'temperature': random.randint(10, 80)}
            trainer.args.nb_epochs = 30
        elif args.pretraining:
            config = {'lr': 0.0002,
                      'weight_decay': 0, # rest does not matter
                      'num_classes_iter': 0,
                      'num_elements_class': 0,
                      'num_labeled_points_class': 0,
                      'num_iter_gtg': 0,
                      'temperature': 0}
            trainer.args.nb_epochs = 10
        else:
            config = {'lr': args.lr_net,
                      'weight_decay': args.weight_decay,
                      'num_classes_iter': args.num_classes_iter,
                      'num_elements_class': args.num_elements_class,
                      'num_labeled_points_class': args.num_labeled_points_class,
                      'num_iter_gtg': args.num_iter_gtg,
                      'temperature': args.temperature}

        best_accuracy, model = trainer.train_model(config, timer, load_path)

        hypers = ', '.join([k + ': ' + str(v) for k, v in config.items()])
        logger.info('Used Parameters: ' + hypers)
        print('Used Parameters: ' + hypers)

        logger.info('Best Recall: {}'.format(best_accuracy))
        print('Best Recall: {}'.format(best_accuracy))
        if best_accuracy > best_recall and not args.test:
            os.rename(os.path.join(save_folder_nets,
                args.dataset_name + '_intermediate_model_' + str(
                                       timer) + '.pth'),
                      mode + args.net_type + '_' + args.dataset_name + '.pth')
            best_recall = best_accuracy
            best_hypers = '_'.join(
                [str(k) + '_' + str(v) for k, v in config.items()])
        elif args.test:
            best_recall = best_accuracy
            best_hypers = '_'.join(
                [str(k) + '_' + str(v) for k, v in config.items()])

    logger.info("Best Hyperparameters found: " + best_hypers)
    logger.info("Achieved {} with this hyperparameters".format(best_recall))
    logger.info("-----------------------------------------------------\n")


if __name__ == '__main__':
    main()
