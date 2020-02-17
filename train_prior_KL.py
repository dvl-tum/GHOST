import argparse
import logging, imp
import time
import random
import pickle
import os
import sys

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

import gtg
import net
import data_utility
import utils

import warnings
warnings.filterwarnings("ignore")

def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp


parser = argparse.ArgumentParser(description='Training inception V2' +
                                             ' (BNInception) on CUB200 with Proxy-NCA loss as described in ' +
                                             '`No Fuss Distance Metric Learning using Proxies.`')

# export directory, training and val datasets, test datasets
parser.add_argument('--cub-root',
                    default='CARS', help='Path to root CUB folder, containing the images folder.')
parser.add_argument('--cub-is-extracted', action='store_true',
                    default=True, help='If `images.tgz` was already extracted, do not extract it again.' +
                                        ' Otherwise use extracted data.')
parser.add_argument('--embedding-size', default=64, type=int,
                    dest='sz_embedding', help='Size of embedding that is appended to InceptionV2.')
parser.add_argument('--number-classes', default=98, type=int,
                    dest='nb_classes', help='Number of first [0, N] classes used for training and ' +
                                            'next [N, N * 2] classes used for evaluating with max(N) = 100.')
parser.add_argument('--num_classes_iter', default=random.randint(3, 10), type=int,
                    help='Number of classes we want to use for every iteration')
parser.add_argument('--num_elements_class', default=random.randint(5, 10), type=int,
                    help='Number of elements per class we want in every iteration')
parser.add_argument('--lr-embedding', default=rnd(1, 10), type=float,
                    help='Learning rate for embedding.')
parser.add_argument('--lr-net', default=rnd(1, 4), type=float,
                    help='Learning rate for Inception, excluding embedding layer.')
parser.add_argument('--weight-decay', default=rnd(8, 15), type=float,
                    dest='weight_decay', help='Weight decay for Inception, embedding layer and Proxy NCA.')
parser.add_argument('--epsilon', default=1e-2, type=float,
                    help='Epsilon (optimizer) for Inception, embedding layer and Proxy NCA.')
parser.add_argument('--gamma', default=1e-1, type=float,
                    help='Gamma for multi-step learning-rate-scheduler.')
parser.add_argument('--epochs', default=30, type=int,
                    dest='nb_epochs', help='Number of training epochs.')
parser.add_argument('--log-filename', default='example',
                    help='Name of log file.')
parser.add_argument('--gpu-id', default=0, type=int,
                    help='ID of GPU that is used for training.')
parser.add_argument('--workers', default=8, type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.')
parser.add_argument('--num_labeled_points_class', default=random.randint(1, 5), type=int,
                    help='Number of labeled points for each class')
parser.add_argument('--net_type', default='bn_inception', type=str,
                    help='type of net we want to use')
parser.add_argument('--sim_type', default='correlation', type=str,
                    help='type of similarity we want to use')
parser.add_argument('--set_negative', default='hard', type=str,
                    help='type of threshold we want to do'
                         'hard - put all negative similarities to 0'
                         'hard - put all negative similarities to 0'
                         'soft - subtract minus (-) negative from each entry')
parser.add_argument('--num_iter_gtg', default=random.randint(1, 6), type=int,
                    help='Number of iterations we want to do for GTG')
parser.add_argument('--embed', default=False, type=bool,
                    help='boolean controling if we want to do embedding or not')
parser.add_argument('--scaling_loss_parameter', default=random.randint(1, 100), type=int,
                    help='Scaling parameter for the KL div loss')

args = parser.parse_args()

# create a salt and name files with it
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
file_name = ''.join(random.choice(ALPHABET) for i in range(16))
file_name_pickle = file_name + '.pickle'

with open(os.path.join('pickle_folder_inception_Cars_prior', file_name_pickle), 'wb') as handle:
    pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

batch_size = args.num_classes_iter * args.num_elements_class
torch.cuda.set_device(args.gpu_id)

if args.net_type == 'bn_inception':
    model = net.bn_inception(pretrained=True, nb_classes=98)
    if args.embed:
        model = net.Inception_embed(model, 1024, args.sz_embedding)

else:
    model = torchvision.models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])

# put the net, gtg and criterion to cuda
model = model.cuda()
gtg = gtg.GTG(args.num_classes_iter, max_iter=args.num_iter_gtg, sim=args.sim_type,
              set_negative=args.set_negative).cuda()
criterion = nn.NLLLoss().cuda()
criterion2 = nn.KLDivLoss().cuda()

opt = torch.optim.Adam(
    [
        {  # proxy nca parameters
            'params': gtg.parameters(),
            'lr': 1e-3
        },
        {  # net parameters, excluding embedding layer
            'params': list(
                set(
                    model.parameters()
                )
            ),
            'lr': args.lr_net
        }
    ],
    eps=args.epsilon,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [3, 10, 16],
                                                 gamma=args.gamma)

imp.reload(logging)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
losses_1 = []
losses_2 = []
scores = []
scores_tr = []

# create loaders
dl_tr, dl_ev, _ = data_utility.create_loaders(args.cub_root, args.nb_classes, args.cub_is_extracted, args.nb_workers,
                                           args.num_classes_iter, args.num_elements_class, batch_size)

t1 = time.time()
# logging.info("**Evaluating initial model...**")
# with torch.no_grad():
#     utils.evaluate(model, dl_ev, args.nb_classes, args.net_type)

def get_predict_and_target(probs, U, y):
    sliced_probs = probs[U]
    sliced_y = y[U]
    len_indices = len(sliced_y)
    valid_indices_1 = []
    valid_indices_2 = []
    for i in range(len_indices):
        for j in range(len_indices):
            if sliced_y[i] == sliced_y[j] and i != j:
                valid_indices_1.append(i)
                valid_indices_2.append(j)
    t1 = sliced_probs[valid_indices_1]
    t2 = sliced_probs[valid_indices_2]
    return t1, t2

for e in range(1, args.nb_epochs+1):
    scheduler.step()
    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    losses_per_epoch_1 = []
    losses_per_epoch_2 = []

    print_gradients = False

    for x, Y in dl_tr:

        y = Y.detach().numpy()
        classes_to_use = np.unique(y)

        if len(classes_to_use) != args.num_classes_iter:
            continue

        # map classes indices to numbers [0, n) where n is the number of classes used for each iteration
        # y is just a renaming of Y
        
        dict_map = {}
        for i in range(len(classes_to_use)):
            dict_map[classes_to_use[i]] = i
        for i in range(Y.shape[0]):
            y[i] = dict_map[y[i]]

        if not print_gradients:
            opt.zero_grad()

        if args.net_type == 'bn_inception':
            if args.embed:
                probs, fc7 = model(x.cuda())
            else:
                probs, fc7 = model(x.cuda())

        # at the moment it is supposed to work with resnet
        else:
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            fc7 = avgpool(fc7)
            fc7 = fc7.view(x.size(0), -1)

        labs, L, U = data_utility.get_labeled_and_unlabeled_points(labels=y,
                                                                   num_points_per_class=args.num_labeled_points_class)
        pred, targ = get_predict_and_target(probs, U, Y)
        pred = F.log_softmax(pred)
        targ = F.softmax(targ)
        probs = F.softmax(probs)
        probs, W = gtg(fc7, fc7.shape[0], labs, L, U, probs, classes_to_use)
        probs = torch.log(probs + 1e-12)
        loss1 = criterion(probs.cuda(), Y.cuda())
        loss2 = criterion2(pred.cuda(), targ.cuda())
        loss = loss1 + args.scaling_loss_parameter * loss2
        if torch.isnan(loss):
            print("We have NaN numbers, closing")
            print("\n\n\n")
            sys.exit(0)

        if print_gradients:
            print(data_utility.debug_info(gtg, model))

        loss.backward()
        losses_per_epoch.append(loss.data.cpu().numpy())
        losses_per_epoch_1.append(loss1.data.cpu().numpy())
        losses_per_epoch_2.append(args.scaling_loss_parameter * loss2.data.cpu().numpy())
        opt.step()

    time_per_epoch_2 = time.time()

    losses.append(np.mean(losses_per_epoch[-20:]))
    losses_1.append(np.mean(losses_per_epoch_1[-20:]))
    losses_2.append(np.mean(losses_per_epoch_2[-20:]))

    # logging.info(
    #     "Epoch: {}, loss: {:.3f}, loss1: {:.3f}, loss2: {:.3f), time (seconds): {:.2f}.".format(
    #         e,
    #         losses[-1],
    #         losses_1[-1],
    #         losses_2[-1],
    #         time_per_epoch_2 - time_per_epoch_1
    #     )
    # )
    with torch.no_grad():
        logging.info("**Evaluating...**")
        nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes, args.net_type)
        scores.append((nmi, recall))
        # scores.append(utils.evaluate(model, dl_ev, args.nb_classes, args.net_type))
    model.losses = losses
    model.current_epoch = e

    if recall[0] < 0.40:
        print("NMI is less than 40, closing")
        print(nmi)
        print("\n\n\n")
        sys.exit(0)
    if e > 5 and recall[0] < 0.40:
        print("NMI is less than 40, closing")
        print("\n\n\n")
        print(nmi)
        sys.exit(0)
    if e > 5 and recall[0] < 0.50:
        print("NMI is less than 50, closing")
        print("\n\n\n")
        print(nmi)
        sys.exit(0)

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))

with open('inception_results_Cars_prior_KL.txt', 'a+') as fp:
    fp.write(file_name + "\n")
    fp.write('\n'.join('%s %s' % x for x in scores))
    fp.write("\n\n\n")

