import torch
import numpy as np
import sklearn.metrics.pairwise

def assign_by_euclidian_at_k(X, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    if type(X) != dict:
        print("HERE")
        distances = sklearn.metrics.pairwise.pairwise_distances(X)
    else:
        samples = list(X.keys())
        T = [T[k][k] for k in X.keys()]
        distances = list()
        for k1 in samples:
            dist = list()
            for k2 in samples:
                if k2 in X[k1].keys():
                    dist.append(X[k1][k2])
                else:
                    dist.append(1000)
            distances.append(dist)
        distances = np.array(distances)
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1]
    
    return np.array([[T[i] for i in ii] for ii in indices]), T


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    Y = torch.from_numpy(Y)
    #for t, y in zip(T, Y):
    #    print(t, y[:k])
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


