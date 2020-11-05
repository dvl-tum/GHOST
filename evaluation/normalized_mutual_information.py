import sklearn.cluster
import sklearn.metrics.cluster
import torch
import numpy as np

def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    if X.__class__.__name__ == 'defaultdict':
        x_new = list()
        for k, v in X.items():
            x_new.append(v["feats"].unsqueeze(0))
        X = torch.cat(x_new, dim=0).cpu()
        print(X.shape)
    return sklearn.cluster.KMeans(nb_clusters).fit(X).labels_

def calc_normalized_mutual_information(ys, xs_clustered):
    if ys.__class__.__name__ == 'defaultdict':
        y_new = list()
        for k, v in ys.items():
            y_new.append(v[k])
        ys = torch.tensor(y_new).cpu()
    #for y, x in zip(xs_clustered, ys):
    #    print(y, x)
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)
