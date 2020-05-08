from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import dataset
import PIL
from apex import amp
import argparse
import copy
import random
from RAdam import RAdam
import gtg
import net
import data_utility
import utils


class Hyperparameters():
    def __init__(self, dataset_name='cub'):
        self.dataset_name = dataset_name
        if dataset_name == 'cub':
            self.dataset_path = '../../datasets/CUB_200_2011'
        elif dataset_name == 'cars':
            self.dataset_path = '../../datasets/CARS'
        elif dataset_name == 'Market':
            self.dataset_path = '../../datasets/Market'
        elif dataset_name == 'cuhk03':
            self.dataset_path = '../../datasets/cuhk03'
        else:
            self.dataset_path = '../../datasets/Stanford'

        self.num_classes = {'cub': 100, 'cars': 98, 'Stanford': 11318, 'Market': 651, 'cuhk03': 1267}
        self.num_classes_iteration = {'cub': 6, 'cars': 5, 'Stanford': 10, 'Market': 5, 'cuhk03': 5}
        self.num_elemens_class = {'cub': 9, 'cars': 7, 'Stanford': 6, 'Market': 7, 'cuhk03': 7}
        self.get_num_labeled_class = {'cub': 2, 'cars': 3, 'Stanford': 2, 'Market': 2, 'cuhk03': 2}
        # self.learning_rate = 0.0002
        self.learning_rate = {'cub': 0.0001563663718906821, 'cars': 0.0002, 'Stanford': 0.0006077651100709081, 'Market': 0.0002, 'cuhk03': 0.00002}
        self.weight_decay = {'cub': 6.059722614369727e-06, 'cars': 4.863656728256105e-07, 'Stanford': 5.2724883734490575e-12, 'Market': 4.863656728256105e-07, 'cuhk03': 4.863656728256105e-07}
        self.softmax_temperature = {'cub': 24, 'cars': 79, 'Stanford': 54, 'Market': 79, 'cuhk03': 79}

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
        return 1

    def get_softmax_temperature(self):
        return self.softmax_temperature[self.dataset_name]


def init_args():
    dataset = 'Market'
    hyperparams = Hyperparameters(dataset)
    parser = argparse.ArgumentParser(
        description='Pretraining for Person Re-ID with Group Loss')
    parser.add_argument('--dataset_name', default=dataset, type=str,
                        help='The name of the dataset')
    parser.add_argument('--oversampling', default=1, type=int,
                        help='If oversampling shoulf be used')
    parser.add_argument('--num_epochs', default=100, type=int)

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

    '''parser.add_argument('--num_classes_iter',
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
    parser.add_argument('--nb_epochs', default=hyperparams.get_epochs(),
                        type=int, help='Number of training epochs.')
    parser.add_argument('--temperature',
                        default=hyperparams.get_softmax_temperature(),
                        help='Temperature parameter for the softmax')'''

    parser.add_argument('--nb_workers', default=4, type=int,
                        help='Number of workers for dataloader.')
    parser.add_argument('--net_type', default='bn_inception', type=str,
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
    parser.add_argument('--is_apex', default=0, type=int,
                        help='if 1 use apex to do mixed precision training')


    return parser.parse_args()


class PreTrainer():
    def __init__(self, args, data_dir, save_name, device):
        self.device = device
        self.data_dir = data_dir
        self.save_name = save_name
        self.args = args

    def train_model(self, config):

        file_name = args.dataset_name + str(
            args.id) + '_' + args.net_type + '_' + str(
            config['lr_net']) + '_' + str(config['weight_decay']) + '_' + str(
            config['num_classes_iter']) + '_' + str(
            config['num_elements_class']) + '_' + str(
            config['num_labeled_points_class'])

        model = net.load_net(dataset=self.args.dataset_name, net_type=self.args.net_type,
                             nb_classes=self.args.nb_classes, embed=config['embed'],
                             sz_embedding=config['sz_embedding'])
        model = model.to(self.device)

        gtg = gtg.GTG(config['nb_classes'], max_iter=config['num_iter_gtg'],
                      sim=args.sim_type,
                      set_negative=args.set_negative, device=self.device).to(self.device)
        opt = RAdam(
            [{'params': list(set(model.parameters())), 'lr': config['lr']}],
            weight_decay=config['weight_decay'])
        criterion = nn.NLLLoss().to(self.device)
        criterion2 = nn.CrossEntropyLoss().to(self.device)

        # do training in mixed precision
        if args.is_apex:
            model, opt = amp.initialize(model, opt, opt_level="O1")

        # create loaders
        batch_size = config['num_classes_iter'] * config['num_elements_class']
        dl_tr, dl_ev, _, _ = data_utility.create_loaders(args.cub_root,
                                                         args.nb_classes,
                                                         args.cub_is_extracted,
                                                         args.nb_workers,
                                                         config['num_classes_iter'],
                                                         config['num_elements_class'],
                                                         batch_size)


        since = time.time()
        best_accuracy = 0
        scores = []
        for e in range(1, args.nb_epochs + 1):
            if e == 31:
                model.load_state_dict(torch.load(
                    os.path.join(save_folder_nets, file_name + '.pth')))
                for g in opt.param_groups:
                    g['lr'] = args.lr_net / 10.

            if e == 51:
                model.load_state_dict(torch.load(
                    os.path.join(save_folder_nets, file_name + '.pth')))
                for g in opt.param_groups:
                    g['lr'] = args.lr_net / 10.

            i = 0
            for x, Y in dl_tr:
                Y = Y.to(device)
                opt.zero_grad()

                probs, fc7 = model(x.to(device))
                labs, L, U = data_utility.get_labeled_and_unlabeled_points(
                    labels=Y,
                    num_points_per_class=config['num_labeled_points_class'],
                    num_classes=args.nb_classes)

                # compute the smoothed softmax
                probs_for_gtg = F.softmax(probs / config['temperature'])

                # do GTG (iterative process)
                probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U,
                                       probs_for_gtg)
                probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

                # compute the losses
                loss1 = criterion(probs_for_gtg, Y)
                loss2 = criterion2(probs, Y)
                loss = args.scaling_loss * loss1 + loss2
                i += 1

                # check possible net divergence
                if torch.isnan(loss):
                    print("We have NaN numbers, closing")
                    print("\n\n\n")
                    sys.exit(0)

                # backprop
                if args.is_apex:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                opt.step()

            # compute recall and NMI at the end of each epoch (for Stanford NMI takes forever so skip it)
            with torch.no_grad():
                logging.info("**Evaluating...**")
                nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes,
                                             args.net_type,
                                             dataroot=args.dataset_name)
                print('Recall {}, NMI {}'.format(recall, nmi))
                scores.append((nmi, recall))
                model.current_epoch = e
                if recall[0] > best_accuracy:
                    best_accuracy = recall[0]
                    torch.save(model.state_dict(),
                               os.path.join(save_folder_nets,
                                            file_name + '.pth'))

        return scores

def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp

def main():
    args = init_args()
    save_name = args.model_name + '_' + args.dataset_name + '_pretrained.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir('search_results'):
        os.makedirs('search_results')

    trainer = PreTrainer(args, args.cub_root, save_name, device)

    best_acc = 0
    num_iter = 100
    # Random search
    for i in range(num_iter):

        # random search for hyperparameters
        lr = 10**random.uniform(-8, -3)
        batch_size = random.choice([8, 16, 32, 64])
        weight_decay = 10 ** random.uniform(-15, -6)
        num_classes_iter = random.randint(2, 10)
        num_elements_classes = random.randint(4, 10)
        num_labeled_class = random.randint(1, 3)
        decrease_lr = random.randint(0, 15)
        set_negative = random.choice(0, 1)
        #sim_type = random.choice(0, 1)
        num_iter_gtg = random.choice(1, 3)


        config = {'lr': lr,
                  'weight_decay': weight_decay,
                  'batch_size': batch_size,
                  'num_classes_iter': num_classes_iter,
                  'num_elements_classes': num_elements_classes,
                  'num_labeled_points_class': num_labeled_class,
                  'decrease_lr': decrease_lr,
                  'set_negative': set_negative,
                  'num_iter_gtg': num_iter_gtg}

        model, val_acc_history = trainer.train_model(config)

        with open(os.path.join(save_folder_results, file_name + '.txt'),
                  'a+') as fp:
            fp.write(file_name + "\n")
            fp.write(str(args))
            fp.write('\n')
            fp.write(str(config))
            fp.write('\n')
            fp.write('\n'.join('%s %s' % x for x in scores))
            fp.write("\n\n\n")

        hypers = '_'.join([str(k) + '_' + str(v) for k, v in config.items()])

        acc = max(val_acc_history)
        if acc > best_acc:
            best_model = model
            best_acc = acc
            best_hypers = hypers

        save_name = os.path.join('search_results', str(acc) + hypers + '.txt')
        with open(save_name, 'w') as file:
            for e in val_acc_history:
                file.write(str(e.data))
                file.write('\n')

    torch.save(best_model, 'fine_tuned' + args.model_name + args.dataset_name + '.pth')

    print("Best Hyperparameters found: " + best_hypers)



if __name__ == '__main__':
    main()
