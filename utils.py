import torch.nn as nn
import evaluation
import torch
import logging
import sys
import numpy as np
import net
import data_utility
from collections import OrderedDict
import os


# just looking at this gives me AIDS, fix it fool!
def predict_batchwise(model, dataloader, net_type, dataroot):
    fc7s, L = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            _, fc7 = model(X.cuda())
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    return torch.squeeze(fc7), torch.squeeze(Y)

def predict_batchwise_reid(model, dataloader):
    fc7s, L = [], []
    features = dict()
    labels = dict()
    with torch.no_grad():
        for X, Y, P in dataloader:
            _, fc7 = model(X.cuda())
            for path, out, y in zip(P, fc7, Y):
                features[path] = out
                labels[path] = y
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    return torch.squeeze(fc7), torch.squeeze(Y), features, labels


def evaluate_reid(model, dataloader, nb_classes, net_type='bn_inception',
             dataroot='CARS', query=None, gallery=None, root=None):
    model_is_training = model.training
    model.eval()
    X, T, features, labels = predict_batchwise_reid(model, dataloader)
    mAP, cmc = evaluation.calc_mean_average_precision(features, labels, query, gallery,
                                                 root)
    print(mAP, cmc)
    model.train(model_is_training)
    return mAP, cmc


def evaluate(model, dataloader, nb_classes, net_type='bn_inception',
             dataroot='CARS', query=None, gallery=None, root=None):
    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels (non-batch-wise)

    X, T = predict_batchwise(model, dataloader, net_type, dataroot)

    if dataroot != 'Stanford':
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(T, evaluation.cluster_by_kmeans(X, nb_classes))
        logging.info("NMI: {:.3f}".format(nmi * 100))
    else:
        nmi = -1

    recall = []
    if dataroot != 'Stanford':
        Y = evaluation.assign_by_euclidian_at_k(X, T, 8)
        which_nearest_neighbors = [1, 2, 4, 8]
    else:
        Y = evaluation.assign_by_euclidian_at_k(X, T, 1000)
        which_nearest_neighbors = [1, 10, 100, 1000]
    
    for k in which_nearest_neighbors:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    model.train(model_is_training) # revert to previous training state
    return nmi, recall


if __name__ == '__main__':

    model = net.load_net(dataset='cuhk03', net_type='resnet50',
                         nb_classes=1367, embed=0,
                         sz_embedding=512,
                         pretraining=True)

    dl_tr, dl_ev, _, _, query, gallery = data_utility.create_loaders(
        '../../datasets/cuhk03', 1367, True, 1, 2, 2, 4)

    evaluate(model=model, dataloader=dl_ev, nb_classes=1367, net_type='resnet50',
             dataroot='cuhk03', query=query, gallery=gallery, root='../../datasets/cuhk03')