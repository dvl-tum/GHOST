import sklearn.cluster
import sklearn.metrics.cluster

def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    if type(X) == dict:
        x_new = list()
        for k, v in X:
            x_new.append(v[k])
        X = np.vstack(x_new)
    return sklearn.cluster.KMeans(nb_clusters).fit(X).labels_

def calc_normalized_mutual_information(ys, xs_clustered):
    if type(ys) == dict:
        y_new = list()
        for k, v in X:
            y_new.append(v[k])
        ys = np.hstack(y_new)
    #for y, x in zip(xs_clustered, ys):
    #    print(y, x)
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)
