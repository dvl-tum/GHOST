import torch
import numpy as np
import sklearn.metrics.pairwise
import logging


logger = logging.getLogger('GNNReID.Recall')


def assign_by_euclidian_at_k(X, T, k, P=None, query=None, gallery=None):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
     
    if query is not None: # inshop
        x = torch.cat([X[i].unsqueeze(0) for i in range(len(P)) if P[i] in query], 0)
        TQ = [T[i].item() for i in range(len(P)) if P[i] in query]
        q = [P[i] for i in range(len(P)) if P[i] in query]
        y = torch.cat([X[i].unsqueeze(0) for i in range(len(P)) if P[i] in gallery], 0)
        TG = [T[i].item() for i in range(len(P)) if P[i] in gallery]
        g = [P[i] for i in range(len(P)) if P[i] in gallery]
        
        distances = sklearn.metrics.pairwise.pairwise_distances(x, y)
        indices = np.argsort(distances, axis = 1)[:,:k]
        
        return np.array([[TG[i] for i in ii] for ii in indices]), TQ

    else: # normal
        print("HERE")
        distances = sklearn.metrics.pairwise.pairwise_distances(X)
        # get nearest points
        #k = 8
        indices = np.argsort(distances, axis = 1)[:, 1 : k + 1]
        #for i in range(indices.shape[0]):
        #    logger.info("Sample {}".format(P[i]))
        #    for j in range(indices.shape[1]):
        #        logger.info(P[indices[i, j]])
        
        return np.array([[T[i] for i in ii] for ii in indices]), T


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    Y = torch.from_numpy(Y)
    
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))

