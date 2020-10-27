import torch
import numpy as np
import sklearn.metrics.pairwise

def assign_by_euclidian_at_k(X, T, k, P=None, query=None, gallery=None):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    if P is not None:
        x = torch.cat([X[i] for i in range(len(P)) if P[i] in query], 0)
        TQ = [T[i] for i in range(len(P)) if P[i] in query]
        y = torch.cat([X[i] for i in range(len(P)) if P[i] in gallery], 0)
        TG = [T[i] for i in range(len(P)) if P[i] in gallery]
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        distances = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        distances.addmm_(1, -2, x, y.t())

    elif type(X) != dict:
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
    indices = np.argsort(distances, axis = 1)[:, 1 : k + 1]

    if P is None:
        return np.array([[T[i] for i in ii] for ii in indices]), T
    else:
        return np.array([[TG[i] for i in ii] for ii in indices]), TQ


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


