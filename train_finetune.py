import argparse
import logging, imp
import time
import random
import pickle
import os
import sys
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

import gtg
import net
import data_utility
import utils as utils

from RAdam import RAdam

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
                    default=False, help='If `images.tgz` was already extracted, do not extract it again.' +
                                        ' Otherwise use extracted data.')
parser.add_argument('--embedding-size', default=512, type=int,
                    dest='sz_embedding', help='Size of embedding that is appended to InceptionV2.')
parser.add_argument('--number-classes', default=98, type=int,
                    dest='nb_classes', help='Number of first [0, N] classes used for training and ' +
                                            'next [N, N * 2] classes used for evaluating with max(N) = 100.')
parser.add_argument('--lr-net', default=3e-4, type=float,
                    help='Learning rate for Inception, excluding embedding layer.')
parser.add_argument('--weight-decay', default=0, type=float,
                    dest='weight_decay', help='Weight decay for Inception, embedding layer and Proxy NCA.')
parser.add_argument('--epochs', default=10, type=int,
                    dest='nb_epochs', help='Number of training epochs.')
parser.add_argument('--log-filename', default='example',
                    help='Name of log file.')
parser.add_argument('--epsilon', default=1e-2, type=float,
                    help='Epsilon (optimizer) for Inception, embedding layer and Proxy NCA.')
parser.add_argument('--gpu-id', default=0, type=int,
                    help='ID of GPU that is used for training.')
parser.add_argument('--workers', default=2, type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='type of net we want to use')
parser.add_argument('--embed', default=False, type=bool,
                    help='Number of iterations we want to do for GTG')

args = parser.parse_args()

batch_size = 64
# torch.cuda.set_device(args.gpu_id)

if args.net_type == 'bn_inception':
    model = net.bn_inception(pretrained=True, nb_classes=args.nb_classes)
    model.last_linear = nn.Linear(1024, args.nb_classes)
    if args.embed:
        model = net.Inception_embed(model, 1024, args.sz_embedding, args.nb_classes)

else:
    model = net.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.nb_classes)

# put the net, gtg and criterion to cuda
model = model.cuda()
# gtg = gtg_new.GTG(args.num_classes_iter, max_iter=args.num_iter_gtg, sim=args.sim_type,
#                   set_negative=args.set_negative).cuda()
criterion = nn.CrossEntropyLoss().cuda()


opt = RAdam(
    [
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
scores = []
scores_tr = []

# create loaders
dl_ev, dl_finetune = data_utility.create_loaders_finetune(args.cub_root, args.nb_classes, args.cub_is_extracted, args.nb_workers, batch_size)

t1 = time.time()
logging.info("**Evaluating initial model...**")
# with torch.no_grad():
#    nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes, args.net_type)
correct = 0
total = 0
with torch.no_grad():
    for data in dl_finetune:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs, fc7 = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network : %d %%' % (
        100 * correct / total))

for e in range(1, args.nb_epochs+1):
    # scheduler.step()
    time_per_epoch_1 = time.time()
    losses_per_epoch = []

    print_gradients = False

    for x, Y in dl_finetune:
        if not print_gradients:
            opt.zero_grad()

        probs_h, fc7_h = model(x.cuda())

        loss_h = criterion(probs_h.cuda(), Y.cuda())
        loss = loss_h

        loss.backward()

        # if print_gradients:
        #     print(data_utility.debug_info(gtg, model))

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

    time_per_epoch_2 = time.time()

    if True:
        losses.append(np.mean(losses_per_epoch[-20:]))
        logging.info(
            "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
                e,
                losses[-1],
                time_per_epoch_2 - time_per_epoch_1
            )
        )
    if True:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dl_finetune:
                images = images.cuda()
                labels = labels.cuda()
                outputs, fc7 = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network: %d %%' % (
                100 * correct / total))
        torch.save(model.state_dict(), 'net/cars_resnet.pth')
        with torch.no_grad():
            logging.info("**Evaluating...**")
            _, recall = utils.evaluate(model, dl_ev, args.nb_classes, args.net_type)
        model.losses = losses
        model.current_epoch = e



t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))

