import torch.nn as nn
import evaluation
import torch
import logging
import sys
import numpy as np


# just looking at this gives me AIDS, fix it fool!
def predict_batchwise(model, dataloader, net_type):
    fc7s, L = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            _, fc7 = model(X.cuda())
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    return torch.squeeze(fc7), torch.squeeze(Y)


def evaluate(model, dataloader, nb_classes, net_type='bn_inception', dataroot='CARS'):
    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels (non-batch-wise)
    X, T = predict_batchwise(model, dataloader, net_type)


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
