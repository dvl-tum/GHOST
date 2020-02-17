import logging, imp
import random
import os
import sys
import warnings

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import gtg
import net
import data_utility
import utils
from config import parser
from RAdam import RAdam

warnings.filterwarnings("ignore")


args = parser.parse_args()

file_name = args.dataset_name + '_' + str(args.lr_net) + '_' + str(args.weight_decay) + '_' + str(
    args.num_classes_iter) + '_' + str(args.num_elements_class) + '_' + str(args.num_labeled_points_class)
batch_size = args.num_classes_iter * args.num_elements_class
device = 'cuda:0'
save_folder = 'cub_embedding'


if args.net_type == 'bn_inception':
    model = net.bn_inception(pretrained=True, nb_classes=args.nb_classes)
    if args.embed:
        model = net.Inception_embed(model, 1024, args.sz_embedding, num_classes=args.nb_classes)
        model.load_state_dict(torch.load(os.path.join('net', 'finetuned_cub_embedded_512_10_.pth')))
else:
    model = net.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.nb_classes)
    model.load_state_dict(torch.load('net/cars_resnet.pth'))

# put the net, gtg and criterion to cuda
model = model.to(device)
gtg = gtg.GTG(args.nb_classes, max_iter=args.num_iter_gtg, sim=args.sim_type,
              set_negative=args.set_negative, device=device).to(device)
opt = RAdam([{'params': list(set(model.parameters())), 'lr': args.lr_net}], weight_decay=args.weight_decay)
criterion = nn.NLLLoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)


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
        nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes, args.net_type, dataroot=args.cub_root)
        # nmi_tr, recall_tr = utils.evaluate(model, dl_train_evaluate, args.nb_classes, args.net_type, dataroot=args.cub_root)
        print(recall)
        scores.append((nmi, recall))
        # scores_tr.append((nmi_tr, recall_tr))
        model.losses = losses
        model.current_epoch = e
        if recall[0] > best_accuracy:
            best_accuracy = recall[0]
            torch.save(model.state_dict(), os.path.join(save_folder, file_name + '.pth'))
        elif args.revert_best_accuracy:
            model.load_state_dict(torch.load(os.path.join(save_folder, file_name + '.pth')))


with open(os.path.join('new_results_embedding/', file_name + '.txt'), 'a+') as fp:
    fp.write(file_name + "\n")
    fp.write(str(args))
    fp.write('\n')
    fp.write('\n'.join('%s %s' % x for x in scores))
    fp.write("\n\n\n")
