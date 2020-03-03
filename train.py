import logging, imp
import random
import os
import sys
import warnings
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import gtg
import net
import data_utility
import utils
from RAdam import RAdam

import argparse
import random


def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp


warnings.filterwarnings("ignore")


class Hyperparameters():
    def __init__(self, dataset_name='cub'):
        self.dataset_name = dataset_name
        if dataset_name == 'cub':
            self.dataset_path = '../gtg_embedding_old_reliable_dirty/CUB_200_2011'
        elif dataset_name == 'cars':
            self.dataset_path = '../gtg_embedding_old_reliable_dirty/CARS'
        else:
            self.dataset_path = '../gtg_embedding_old_reliable_dirty/Stanford'

    def get_path(self):
        return self.dataset_path

    def get_number_classes(self):
        if self.dataset_name == 'cub':
            return 100
        elif self.dataset_name == 'cars':
            return 98
        else:
            return 11318

    def get_number_classes_iteration(self):
        # return random.randint(5, 10)
        # return 5
        # return 6
        return 10

    def get_number_elements_class(self):
        # return random.randint(5, 10)
        # return 7
        # return 9
        return 6
    
    def get_number_labeled_elements_class(self):
        # return random.randint(1, 4)
        # return 3
        # return 2
        return 2

    def get_learning_rate(self):
        # return rnd(2, 4)
        # return 0.0001968001756542917
        # return 0.0001563663718906821
        return 0.0027904232570937727

    def get_weight_decay(self):
        # return rnd(5, 15)
        # return 4.863656728256105e-07
        # return 6.059722614369727e-06
        return 5.2724883734490575e-12

    def get_epochs(self):
        # return 60
        return 40

    def get_num_gtg_iterations(self):
        # return random.randint(1, 3)
        return 1

    def get_softmax_temperature(self):
        # return random.randint(1, 100)
        # return 79
        # return 24
        return 54

parser = argparse.ArgumentParser(description='Training inception V2' +
                                             ' (BNInception) on CUB200 with Proxy-NCA loss as described in ' +
                                             '`No Fuss Distance Metric Learning using Proxies.`')
dataset_name = 'Stanford'  # cub, cars or stanford
parser.add_argument('--dataset_name', default=dataset_name, type=str, help='The name of the dataset')
hyperparams = Hyperparameters(dataset_name)
parser.add_argument('--cub-root', default=hyperparams.get_path(), help='Path to dataset folder')
parser.add_argument('--cub-is-extracted', action='store_true',
                    default=True, help='If `images.tgz` was already extracted, do not extract it again.' +
                                       ' Otherwise use extracted data.')
parser.add_argument('--embedding-size', default=512, type=int, dest='sz_embedding', help='The embedding size')
parser.add_argument('--nb_classes', default=hyperparams.get_number_classes(), type=int,
                    help='Number of first [0, N] classes used for training and ' +
                         'next [N, N * 2] classes used for evaluating with max(N) = 100.')
parser.add_argument('--num_classes_iter', default=hyperparams.get_number_classes_iteration(), type=int,
                    help='Number of classes in the minibatch')
parser.add_argument('--num_elements_class', default=hyperparams.get_number_elements_class(), type=int,
                    help='Number of samples per each class')
parser.add_argument('--num_labeled_points_class', default=hyperparams.get_number_labeled_elements_class(), type=int,
                    help='Number of labeled samples per each class')
parser.add_argument('--lr-net', default=hyperparams.get_learning_rate(), type=float, help='The learning rate')
parser.add_argument('--weight-decay', default=hyperparams.get_weight_decay(), type=float, help='The l2 regularization strength')
parser.add_argument('--nb_epochs', default=hyperparams.get_epochs(), type=int, help='Number of training epochs.')
parser.add_argument('--nb_workers', default=1, type=int, help='Number of workers for dataloader.')
parser.add_argument('--net_type', default='resnet50', type=str, help='The type of net we want to use')
parser.add_argument('--sim_type', default='correlation', type=str, help='type of similarity we want to use')
parser.add_argument('--set_negative', default='soft', type=str,
                    help='type of threshold we want to do'
                         'hard - put all negative similarities to 0'
                         'soft - subtract minus (-) negative from each entry')
parser.add_argument('--num_iter_gtg', default=hyperparams.get_num_gtg_iterations(), type=int, help='Number of iterations we want to do for GTG')
parser.add_argument('--embed', default=0, type=int, help='boolean controling if we want to do embedding or not')
parser.add_argument('--use_prior', default=True, type=bool, help='if True use prior otherwise not')
parser.add_argument('--use_double_loss', default=True, type=bool, dest='use_double_loss', help='if True use also classification loss')
parser.add_argument('--scaling_loss', default=1, type=int, dest='scaling_loss', help='Scaling parameter for the loss')
parser.add_argument('--temperature', default=hyperparams.get_softmax_temperature(), help='Temperature parameter for the softmax')
parser.add_argument('--decrease_learning_rate', default=10., type=float,
                    help='Number of iterations we want to do for GTG')
parser.add_argument('--evaluate_beginning', default=False, type=bool, help='if True evaluate at the beginning')
parser.add_argument('--revert_best_accuracy', default=0, type=int,
                    help='if 1, at the end of each epoch revert the net to the checkpoint with the best accuracy')

args = parser.parse_args()

file_name = args.dataset_name + '_paramRes_' + args.net_type + '_' + str(args.lr_net) + '_' + str(args.weight_decay) + '_' + str(
    args.num_classes_iter) + '_' + str(args.num_elements_class) + '_' + str(args.num_labeled_points_class)
batch_size = args.num_classes_iter * args.num_elements_class
device = 'cuda:0'

print(args.net_type + " with inception hyperparams")

if args.embed == 0:
    save_folder = 'sop_resnet'
    save_folder_results = 'sop_resnet_results'
else:
    if args.revert_best_accuracy == 0:
        save_folder = 'cub_embedding'
        save_folder_results = 'cub_results_embedding'
    else:
        save_folder = 'cub_embedding_revert_best'
        save_folder_results = 'cub_results_revert_best'

if args.net_type == 'bn_inception':
    model = net.bn_inception(pretrained=True, nb_classes=args.nb_classes)
    if args.embed:
        model = net.Inception_embed(model, 1024, args.sz_embedding, num_classes=args.nb_classes)
        model.load_state_dict(torch.load(os.path.join('net', 'finetuned_cub_embedded_512_10_.pth')))
elif args.net_type == 'resnet18':
    model = net.resnet18(pretrained=True)
    model.fc = nn.Linear(512, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_resnet18.pth'))
elif args.net_type == 'resnet34':
    model = net.resnet34(pretrained=True)
    model.fc = nn.Linear(512, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_resnet34.pth'))
elif args.net_type == 'resnet50':
    model = net.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_resnet50.pth'))
elif args.net_type == 'resnet101':
    model = net.resnet101(pretrained=True)
    model.fc = nn.Linear(2048, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_resnet101.pth'))
elif args.net_type == 'resnet152':
    model = net.resnet152(pretrained=True)
    model.fc = nn.Linear(2048, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_resnet152.pth'))
elif args.net_type == 'densenet121':
    model = net.densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_densenet121.pth'))
elif args.net_type == 'densenet161':
    model = net.densenet161(pretrained=True)
    model.classifier = nn.Linear(2208, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_densenet161.pth'))
elif args.net_type == 'densenet169':
    model = net.densenet169(pretrained=True)
    model.classifier = nn.Linear(1664, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_densenet169.pth'))
elif args.net_type == 'densenet201':
    model = net.densenet201(pretrained=True)
    model.classifier = nn.Linear(1920, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_sop_densenet201.pth'))
elif args.net_type == 'shufflenet_v2_x0_5':
    model = net.shufflenet_v2_x0_5(pretrained=True)
    model.fc = nn.Linear(1024, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_cub_shuffle05.pth'))
elif args.net_type == 'shufflenet_v2_x1_0':
    model = net.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(1024, args.nb_classes)
    model.load_state_dict(torch.load('net/finetuned_cub_shuffle10.pth'))

# put the net, gtg and criterion to cuda
model = model.to(device)
gtg = gtg.GTG(args.nb_classes, max_iter=args.num_iter_gtg, sim=args.sim_type,
              set_negative=args.set_negative, device=device).to(device)
opt = RAdam([{'params': list(set(model.parameters())), 'lr': args.lr_net}], weight_decay=args.weight_decay)
criterion = nn.NLLLoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)


def softmax_calibrated(x, scale=1.):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = x.max(dim=1)[0].reshape((-1, 1))
    exp_x = torch.exp((x - max_x) / scale)
    return exp_x / exp_x.sum(dim=1).reshape((-1, 1))

losses, scores, scores_tr = [], [], []

# create loaders
dl_tr, dl_ev, _, dl_train_evaluate = data_utility.create_loaders(args.cub_root, args.nb_classes, args.cub_is_extracted,
                                                                 args.nb_workers,
                                                                 args.num_classes_iter, args.num_elements_class,
                                                                 batch_size)

if args.evaluate_beginning:
    logging.info("**Evaluating initial model...**")
    with torch.no_grad():
        nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes, args.net_type, dataroot=args.cub_root)
    print(nmi, recall)


best_accuracy = 0
for e in range(1, args.nb_epochs + 1):
    if e == 31:
        model.load_state_dict(torch.load(os.path.join(save_folder, file_name + '.pth')))
        for g in opt.param_groups:
            g['lr'] = args.lr_net / 10.
        # scheduler.step()


    losses_per_epoch = []

    i = 0
    for x, Y in dl_tr:
        Y = Y.to(device)
        opt.zero_grad()

        probs, fc7 = model(x.to(device))
        labs, L, U = data_utility.get_labeled_and_unlabeled_points(labels=Y,
                                                                   num_points_per_class=args.num_labeled_points_class,
                                                                   num_classes=args.nb_classes)
        probs_for_gtg = F.softmax(probs / args.temperature)
        if args.use_prior:
            probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U, probs_for_gtg)

        else:
            probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U)

        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

        if args.use_double_loss:
            loss1 = criterion(probs_for_gtg, Y)
            loss2 = criterion2(probs, Y)
            loss = args.scaling_loss * loss1 + loss2
            i += 1
        else:
            loss = criterion(probs_for_gtg, Y)
        if torch.isnan(loss):
            print("We have NaN numbers, closing")
            print("\n\n\n")
            sys.exit(0)


        loss.backward()
        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

    losses.append(np.mean(losses_per_epoch[-20:]))
    logging.info(
        "Epoch: {}, loss: {:.3f}".format(e, losses[-1]))
    with torch.no_grad():
        logging.info("**Evaluating...**")
        nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes, args.net_type, dataroot='Stanford')
        # nmi_tr, recall_tr = utils.evaluate(model, dl_train_evaluate, args.nb_classes, args.net_type, dataroot=args.cub_root)
        print(recall)
        scores.append((nmi, recall))
        # scores_tr.append((nmi_tr, recall_tr))
        model.losses = losses
        model.current_epoch = e
        if recall[0] > best_accuracy:
            best_accuracy = recall[0]
            torch.save(model.state_dict(), os.path.join(save_folder, file_name + '.pth'))
        if args.revert_best_accuracy:
            model.load_state_dict(torch.load(os.path.join(save_folder, file_name + '.pth')))


with open(os.path.join(save_folder_results, file_name + '.txt'), 'a+') as fp:
    fp.write(file_name + "\n")
    fp.write(str(args))
    fp.write('\n')
    fp.write('\n'.join('%s %s' % x for x in scores))
    fp.write("\n\n\n")
