from torch.utils.data.sampler import Sampler
import random
import copy
import torch
import sklearn.metrics.pairwise
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger('GNNReID.CombineSampler')

class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl, batch_sampler=None):
        logger.info("Combine Sampler")
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.feature_dict = None
        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(cl_b, n_cl)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()

        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        # add elements till every class has the same num of obs
        # np.random.choice(idxs, size=self.num_instances, replace=True)
        #for inds in l_inds:
        #    choose = copy.deepcopy(inds)
        #    while len(inds) < self.max:
        #        inds += [random.choice(choose)]
        #print("NEW MAX")

        #for inds in l_inds:
        #    n_els = self.max - len(inds) + 1  # take out 1?
        #    inds.extend(inds[:n_els])  # max + 1
        #print("OLD MAX")

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:] 
            assert len(inds) == 0
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerNoise(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl, batch_sampler):
        logger.info("Combine Sampler Noise")
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.feature_dict = None
        self.epoch = 0
        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(cl_b, n_cl)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()

        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        # add elements till every class has the same num of obs
        # np.random.choice(idxs, size=self.num_instances, replace=True)
        #for inds in l_inds:
        #    choose = copy.deepcopy(inds)
        #    while len(inds) < self.max:
        #        inds += [random.choice(choose)]
        #print("NEW MAX")

        #for inds in l_inds:
        #    n_els = self.max - len(inds) + 1  # take out 1?
        #    inds.extend(inds[:n_els])  # max + 1
        #print("OLD MAX")

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]
            assert len(inds) == 0
        
        for _ in range(int((self.epoch/50)*self.n_cl)): #range(1)
            rands = [lst.pop() for lst in split_list_of_indices]
            random.shuffle(rands)
            [lst.extend([rands.pop()]) for lst in split_list_of_indices]
        
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]

        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class TrainTestCombi(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl, l_inds_train=None, l_inds_gallery=None, backbone=False):
        self.l_inds = l_inds
        self.l_inds_train = l_inds_train
        self.l_inds_gallery = l_inds_gallery if l_inds_gallery is not None else l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.backbone = backbone
        self.k = 50

        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)
        self.get_train_inds()

    def get_train_inds(self):
        random.seed(0)
        np.random.seed(0)
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds_train))
        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(
                inds) // self.n_cl + 1) * self.n_cl - len(inds),
                                           replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]
            assert len(inds) == 0

        random.shuffle(split_list_of_indices)
        self.train_samples = [item for sublist in split_list_of_indices[:self.cl_b] for item in
                          sublist]
        [self.train_samples.pop(ind) for ind in random.sample(list(range(len(self.train_samples))), 2)]
     
    def __iter__(self):
        if self.backbone:
            self.flat_list = [samp for cl in self.l_inds for samp in cl]
            gallery_list = [samp for cl in self.l_inds_gallery for samp in cl]
            self.flat_list = self.flat_list + gallery_list + self.train_samples
        else:
            # get knn
            self.flat_list = list()
            x = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
            y = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
            m, n = x.size(0), y.size(0)
            x = x.view(m, -1)
            y = y.view(n, -1)
            dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

            dist.addmm_(1, -2, x, y.t())
            sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
            indices = list(self.feature_dict.keys())
            batches = list()
            for samp in sorted_dist:
                knn = [indices[i] for i in samp[:self.k]]

                #for query_class in self.l_inds:
                query_idx = indices[samp[0]]
                for n in knn[1:]:
                    gallery_idx = indices[n]
                    batch = self.train_samples + [gallery_idx, query_idx]
                    self.flat_list.append(batch)

            logger.info("{} batches".format(len(self.flat_list)))
            self.flat_list = [item for batch in self.flat_list for item in
                              batch]

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class NumberSampler():
    def __init__(self, num_classes, num_samples):
        self.bs = num_classes * num_samples
        self.possible_denominators = [i for i in range(2, int(self.bs/2+1)) if self.bs%i == 0]

    def sample(self):
        num_classes = random.choice(self.possible_denominators)
        num_samples = int(self.bs/num_classes)
        logger.info("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples


class BatchSizeSampler():
    def sample(self):
        num_classes = random.choice(range(2, 20))
        num_samples = random.choice(range(2, 20))
        logger.info("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples


class ClusterQuality():
    def __init__(self, only_anch=False, num_samps=9):
        if only_anch:
            self.range = 1
        else:
            self.range = num_samps
        self.acc = list()
        self.label_acc = defaultdict(list)
        self.variety = list()

    def check(self, samps, batch=None):
        #print(samps)
        for i in range(self.range):
            acc = (sum([1 for j in samps if j == samps[i]]) - 1)/len(samps)
            self.acc.append(acc)
            self.label_acc[samps[i]].append(acc)
        if batch:
            self.variety.append(len(set(batch))/len(batch))


class PseudoSamplerVIII(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        logger.info("Pseudo sampler - kNN reciprocal")
        self.feature_dict = None
        self.bs = 7 #num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.k1 = 30
        self.k2 = self.bs
        print(self.bs, self.k1, self.k2)
        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.num_samples)

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.bs = self.num_classes * self.num_samples
            quality_checker.num_samples = self.bs
        
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            y =  torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            indices = [k for k in self.feature_dict.keys()]

        # generate distance mat for all classes as in Hierachrical Triplet Loss
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        dist = dist.cpu().numpy()
        sorted_dist = np.argsort(dist, axis=1)
        batches = list()
        exp = 0
        no = 0
        for i in range(sorted_dist.shape[0]):
            e = 0
            #print()
            #print(i, indices[i])
            # k-reciprocal neighbors
            forward = sorted_dist[i, :self.k1 + 1]
            #print(forward)
            #print([self.labels[indices[k]] for k in forward])
            backward = sorted_dist[forward, :self.k1 + 1]
            #print(backward)
            rr = np.where(backward == i)[0]
            #print(rr)
            reciprocal = forward[rr]
            reciprocal_expansion = reciprocal
            for cand in reciprocal:
                cand_forward = sorted_dist[cand, :int(np.around(self.k1 / 2)) + 1]
                cand_backward = sorted_dist[cand_forward, :int(np.around(self.k1 / 2)) + 1]
                fi_cand = np.where(cand_backward == cand)[0]
                cand_reciprocal = cand_forward[fi_cand]
                if len(np.intersect1d(cand_reciprocal, reciprocal)) > 2 / 3 * len(
                        cand_reciprocal):
                    #print("expansion")
                    #print(reciprocal_expansion, [self.labels[indices[k]] for k in reciprocal_expansion])
                    reciprocal_expansion = np.append(reciprocal_expansion, cand_reciprocal)
                    #print(reciprocal_expansion, [self.labels[indices[k]] for k in reciprocal_expansion])
                    #print()
                    e =1
            if e == 1:
                exp +=1
            else: 
                no +=1
            reciprocal_expansion = np.unique(reciprocal_expansion)
            #print(reciprocal_expansion)
            batch = reciprocal_expansion[np.argsort(dist[i, reciprocal_expansion])[:self.bs]].tolist()
            #print(batch)
            k = 0
            while len(batch) < self.bs:
                #print(len(batch), self.bs, sorted_dist[i, k], batch, sorted_dist[i, k] not in batch)
                if sorted_dist[i, k] not in batch:
                    batch.append(sorted_dist[i, k])
                k += 1
            #[batch.append(i) for i in sorted_dist[i, :self.bs] if (i not in batch) and (len(batch) <= self.bs)]
            #print(len(batch), self.bs)
            #print(batch)
            batch = [indices[k] for k in batch]
            #print(batch)
            assert len(batch) == self.bs
            batches.append(batch)
            #print(batch)
            #print([self.labels[k] for k in batch])
            self.quality_checker.check([self.labels[i] for i in batch])
        logger.info("Number batches {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))

        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        logger.info("{} Number of samples to process".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSampler(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        logger.info("Pseudo sampler - kNN")
        self.feature_dict = None
        self.bs = 9 #num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.num_samples)

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.bs = self.num_classes * self.num_samples
            quality_checker.num_samples = self.bs
        
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            y =  torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            indices = [k for k in self.feature_dict.keys()]

        # generate distance mat for all classes as in Hierachrical Triplet Loss
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
        #indices = list(self.feature_dict.keys())
        batches = list()
        for samp in sorted_dist:
            batch = [indices[i] for i in samp[:self.bs]]
            self.quality_checker.check([self.labels[i] for i in batch])
            batches.append(batch)
        logger.info("Number batches {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))

        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        logger.info("{} Number of samples to process".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerII(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # rows as classes
        logger.info("Pseudo sampler II - rows as classes")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.labels = None

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        self.quality_checker = ClusterQuality(only_anch=False, num_samps=self.num_samples)
        
    def get_dist(self):
        x = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
        y = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        self.sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
        self.indices_orig = list(self.feature_dict.keys())

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            quality_checker.num_samples = self.num_samples
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if self.epoch % 5 == 2:
            self.get_dist()
        
        indices = copy.deepcopy(self.indices_orig)
        batches = list()
        while indices:
            batch = list()
            anchor = random.choice(indices)
            ind = self.indices_orig.index(anchor)
            j = 0
            while len(batch) < self.num_samples:
                if not indices:
                    batch.append(self.indices_orig[self.sorted_dist[ind][j]])
                elif self.indices_orig[self.sorted_dist[ind][j]] in indices:
                    #logger.info('{}, {}'.format(sorted_dist[ind][j], indices_orig[sorted_dist[ind][j]]))
                    batch.append(self.indices_orig[self.sorted_dist[ind][j]])
                    indices.remove(self.indices_orig[self.sorted_dist[ind][j]])
                j += 1
            self.quality_checker.check([self.labels[i] for i in batch])
            batches.append(batch)
        logger.info("Number anchors {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        if len(batches)%self.num_classes != 0:
            n = (len(batches)//self.num_classes)*self.num_classes + self.num_classes - len(batches)
            logger.info("Add {} anochors".format(n))
            batches += random.choices(batches, k=n)
        assert len(batches)%self.num_classes == 0
        
        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        logger.info("{} Number of samples to process".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerIII(Sampler):
    def __init__(self, num_classes, num_samples, nb_clusters=None, batch_sampler=None):
        # kmeans
        logger.info("Pseudo sampler III - kmeans")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.cl_b = num_classes
        self.n_cl = num_samples
        self.epoch = 0
        self.nb_clusters = nb_clusters

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.n_cl)
    def get_clusters(self):
        logger.info(self.nb_clusters)
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            self.indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            self.indices = [k for k in self.feature_dict.keys()]
        #logger.info("Kmeans")
        #self.cluster = sklearn.cluster.KMeans(self.nb_clusters).fit(x).labels_        
        #logger.info('spectral')
        #self.cluster = sklearn.cluster.SpectralClustering(self.nb_clusters, assign_labels="discretize", random_state=0).fit(x).labels_
        #self.nb_clusters = 600
        #logger.info('ward')
        #self.cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=self.nb_clusters).fit(x).labels_
        #logger.info('DBSCAN')
        #eps = 0.9
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x).labels_
        logger.info("Optics")
        eps = 0.7
        min_samples = 5
        logger.info("Eps {}, min samples {}".format(eps, min_samples))
        self.cluster = sklearn.cluster.OPTICS(min_samples=min_samples, eps=eps).fit(x).labels_
        #logger.info("Birch")
        #self.cluster = sklearn.cluster.Birch(n_clusters=self.nb_clusters).fit(x).labels_

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()
            quality_checker.num_samps=self.n_cl
        if self.epoch % 5 == 2:
            self.get_clusters()
        
        ddict = defaultdict(list)
        for idx, label in zip(self.indices, self.cluster):
            ddict[label].append(idx)

        l_inds = []
        for key in ddict:
            l_inds.append(ddict[key]) 
        
        l_inds = list(map(lambda a: random.sample(a, len(a)), l_inds))
        
        # add elements till every class has the same num of obs
        # np.random.choice(idxs, size=self.num_instances, replace=True)
        #for inds in l_inds:
        #    choose = copy.deepcopy(inds)
        #    while len(inds) < self.max:
        #        inds += [random.choice(choose)]
        #print("NEW MAX")

        #for inds in l_inds:
        #    n_els = self.max - len(inds) + 1  # take out 1?
        #    inds.extend(inds[:n_els])  # max + 1
        #print("OLD MAX")

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]
        print("Num samples")

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                self.quality_checker.check([self.labels[i] for i in inds[:self.n_cl]], inds[:self.n_cl])
                inds = inds[self.n_cl:] 
            assert len(inds) == 0
        logger.info("{} blocks".format(len(split_list_of_indices)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        #print(self.quality_checker.label_acc)
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))
        logger.info("Cluster Variety {}".format(sum(self.quality_checker.variety)/len(self.quality_checker.variety)))
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            logger.info("Add {} blocks".format(self.cl_b - len(split_list_of_indices) % self.cl_b))
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        assert len(split_list_of_indices) % self.cl_b == 0

        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        logger.info("{} samples to process".format(len(set(self.flat_list)))) 
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerIV(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        self.epoch = 0
        logger.info("Pseudo sampler VI - 8 closest within class")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.num_samples)
        self.labels = None
    
    def get_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
        return sorted_dist

    def get_classes(self):        
        feats = list()
        indices = list()
        dists = list()
        for k in self.feature_dict.keys():
            #print(self.feature_dict[k])
            feats.append([f.cpu() for f in self.feature_dict[k].values()])
            indices.append(list(self.feature_dict[k].keys()))
            x = torch.cat([f.unsqueeze(0) for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0) for f in self.feature_dict[k].values()], 0)
            dists.append(self.get_dist(x, y))

        self.feats = feats
        self.indices = indices
        self.dists = dists

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.quality_checker.num_samples = self.num_samples
        if self.epoch % 5 == 2:
            logger.info('recompute dist for classes')
            self.get_classes()
        
        batches = list()
        for ind, dist in zip(self.indices, self.dists):
            for i in range(len(ind)):
                #closest
                batches.append([ind[dist[i][j]] for j in range(self.num_samples)])
                self.quality_checker.check([self.labels[ind[dist[i][j]]] for j in range(self.num_samples)])
                #farthest away
                #batch = [ind[dist[i][0]]]
                #[batch.append(ind[dist[i][j]]) for j in range(len(ind) - 1, len(ind) - self.num_samples, -1)]
                #batches.append(batch)
        
        logger.info("Number of anchors {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))
        if len(batches)%self.num_classes != 0:
            n = (len(batches)//self.num_classes)*self.num_classes + self.num_classes - len(batches)
            logger.info("Add {} anchors".format(n))
            batches += random.choices(batches, k=n)
        assert len(batches)%self.num_classes == 0

        random.shuffle(batches)
        self.flat_list = [item for anchor in batches for item in anchor]
        logger.info("Samples to be processed".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerV(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        logger.info("Pseudo sampler V - num_classes anchors, num_samples closest")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.num_samples)
        self.labels = None

    def get_distances(self):
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0) for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0) for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
        else:
            x = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
            y = torch.cat([f.unsqueeze(0) for f in self.feature_dict.values()], 0)
            self.indices = list(self.feature_dict.keys())
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        self.sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)

    
    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.quality_checker.num_samples = self.num_samples

        #if self.epoch % 5 == 2:
        if self.epoch % 1 == 0 or self.epoch % 5 == 2:
            logger.info("recompute dist")
            self.get_distances()

        batches = list()
        for i in range(len(self.indices)):
            batches.append([self.indices[self.sorted_dist[i][j]] for j in range(self.num_samples)])
            self.quality_checker.check([self.labels[self.indices[self.sorted_dist[i][j]]] for j in range(self.num_samples)])
        logger.info("Number anchors {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))
        if len(batches)%self.num_classes != 0:
            n = (len(batches)//self.num_classes)*self.num_classes + self.num_classes - len(batches)
            logger.info("Add {} anchors".format(n))
            batches += random.choices(batches, k=n)
        assert len(batches)%self.num_classes == 0

        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        logger.info("Samples to be processed {}".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerVI(Sampler):
    def __init__(self, num_classes=6, num_samples=9, nb_clusters=None, batch_sampler=None):
        logger.info("Pseudo sampler VI - closest kmeans")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = 6
        self.num_samples = 9
        self.epoch = 0
        self.nb_clusters = nb_clusters

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.num_samples)
        self.labels = None

    def get_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        sorted_dist = np.argsort(dist.cpu().numpy(), axis=1)
        return sorted_dist

    def get_classes(self):
        feats = list()
        indices = list()
        dists = list()
        self.feature_dict = defaultdict(dict)
        for i in range(len(self.indices)):
            self.feature_dict[self.cluster[i]][self.indices[i]] = self.x[i]
        
        for k in self.feature_dict.keys():
            #print(self.feature_dict[k])
            feats.append([f.cpu() for f in self.feature_dict[k].values()])
            indices.append(list(self.feature_dict[k].keys()))
            x = torch.cat([f.unsqueeze(0) for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0) for f in self.feature_dict[k].values()], 0)
            dists.append(self.get_dist(x, y))

        self.feats = feats
        self.indices = indices
        self.dists = dists


    def get_clusters(self):
        logger.info(self.nb_clusters)
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            self.x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            self.x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            self.indices = [k for k in self.feature_dict.keys()]
        logger.info("Kmeans")
        self.cluster = sklearn.cluster.KMeans(self.nb_clusters).fit(self.x).labels_
        #logger.info('spectral')
        #self.cluster = sklearn.cluster.SpectralClustering(self.nb_clusters, assign_labels="discretize", random_state=0).fit(x).labels_
        #self.nb_clusters = 900
        #logger.info('ward')
        #self.cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=self.nb_clusters).fit(x).labels_
        #logger.info('DBSCAN')
        #eps = 0.9
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x).labels_
        #logger.info("Optics")
        #eps = 0.7
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.OPTICS(min_samples=min_samples, eps=eps).fit(x).labels_
        #logger.info("Birch")
        #self.cluster = sklearn.cluster.Birch(n_clusters=self.nb_clusters).fit(x).labels_

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.quality_checker.num_samples = self.num_samples

        if self.epoch % 5 == 2:
            logger.info('recompute clusters and dist in classes')
            self.get_clusters()
            self.get_classes()

        batches = list()
        print(sum([len(shit) for shit in self.indices]))
        for ind, dist in zip(self.indices, self.dists):
            for i in range(len(ind)):
                #closest
                if len(ind) >= self.num_samples:
                    batch = [ind[dist[i][j]] for j in range(self.num_samples)]
                    batches.append(batch)
                    self.quality_checker.check([self.labels[k] for k in batch])
                else:
                    batch = [ind[dist[i][j]] for j in range(len(ind))] + random.choices(ind, k=self.num_samples-len(ind))
                    batches.append(batch)
                    self.quality_checker.check([self.labels[k] for k in batch])
                #farthest away
                #batch = [ind[dist[i][0]]]
                #[batch.append(ind[dist[i][j]]) for j in range(len(ind) - 1, len(ind) - self.num_samples, -1)]
                #batches.append(batch)
        print(len(set([shit[0] for shit in batches])))
        print(len(batches))
        logger.info("Number of anchors {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))
        if len(batches)%self.num_classes != 0:
            n = (len(batches)//self.num_classes)*self.num_classes + self.num_classes - len(batches)
            logger.info("Add {} anchors".format(n))
            batches += random.choices(batches, k=n)
        assert len(batches)%self.num_classes == 0
        print(len(set([s for shit in batches for s in shit])))
        random.shuffle(batches)
        self.flat_list = [item for anchor in batches for item in anchor]
        print(len(set(self.flat_list)))
        print("Lets evaluate")
        logger.info("Samples to be processed".format(len(self.flat_list)))
        return (iter(self.flat_list))
    
    def __len__(self):
        return len(self.flat_list)


class PseudoSamplerVII(Sampler):
    def __init__(self, num_classes=6, num_samples=9, nb_clusters=None, batch_sampler=None):
        logger.info("Pseudo sampler VII - closest to centroids kmeans")
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = 6
        self.num_samples = 9
        self.epoch = 0
        self.nb_clusters = nb_clusters

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        self.quality_checker = ClusterQuality(only_anch=True, num_samps=self.num_samples)
        self.labels = None

    def get_farthest(self, points, k, cluster_num):
        #points = [torch.from_numpy(p) for p in points]
        k = min(k, len(points))
        remaining_points = points[:]
        solution_set = []
        solution_set.append(remaining_points.pop(cluster_num))
        for _ in range(k-1):
            distances = torch.min(self.get_dist(torch.stack(solution_set, dim=0), torch.stack(remaining_points, dim=0)), dim=0)[0]
            solution_set.append(remaining_points.pop(np.argmax(distances)))
        return solution_set #[i for i in range(len(points)) for j in solution_set if (points[i] == j).all()]


    def get_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        return dist


    def get_classes(self):
        #self.dists = list()
        self.feature_dict = defaultdict(dict)
        for i in range(len(self.indices)):
            self.feature_dict[self.cluster[i]][self.indices[i]] = self.x[i]
        
        self.indices = list()
        self.feats = list()
        self.centr = list()
        for k in self.feature_dict.keys():
            #print(self.feature_dict[k])
            self.feats.append([f.cpu() for f in self.feature_dict[k].values()])
            self.indices.append(list(self.feature_dict[k].keys()))
            #y = torch.cat([f.unsqueeze(0) for f in self.feature_dict[k].values()], 0)
            #y = torch.cat([f.unsqueeze(0) for f in self.feature_dict[k].values()], 0)
            #self.dists.append(self.get_dist(x, y))
            self.centr.append(self.centroids[k])

        #self.centroid_dist = self.get_dict(torch.tensor(centr), torch.tensor(centr))

    def get_clusters(self):
        logger.info(self.nb_clusters)
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            self.x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            self.x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            self.indices = [k for k in self.feature_dict.keys()]
        logger.info("Kmeans")
        kmeans = sklearn.cluster.KMeans(self.nb_clusters).fit(self.x)
        self.cluster = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

        #logger.info('spectral')
        #self.cluster = sklearn.cluster.SpectralClustering(self.nb_clusters, assign_labels="discretize", random_state=0).fit(x).labels_
        #self.nb_clusters = 900
        #logger.info('ward')
        #self.cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=self.nb_clusters).fit(x).labels_
        #logger.info('DBSCAN')
        #eps = 0.9
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x).labels_
        #logger.info("Optics")
        #eps = 0.7
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.OPTICS(min_samples=min_samples, eps=eps).fit(x).labels_
        #logger.info("Birch")
        #self.cluster = sklearn.cluster.Birch(n_clusters=self.nb_clusters).fit(x).labels_

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.quality_checker.num_samples = self.num_samples

        if self.epoch % 5 == 2:
            logger.info('recompute clusters and dist in classes')
            self.get_clusters()
            self.get_classes()

        batches = list()
        remaining_clusters = list(range(len(self.indices)))
        centroids = {i: torch.tensor(self.centr[i]) for i in range(len(self.indices))}
        print(sum([len(shit) for shit in self.indices]))
        initial_indices = {i: copy.deepcopy(self.indices[i]) for i in range(len(self.indices))}
        while centroids:
            clusters = self.get_farthest(list(centroids.values()), self.num_classes, random.choice(remaining_clusters))
            clusters = [k for k, v in centroids.items() for c in clusters if (c==v).all()]
            for cl in clusters:
                if len(self.indices[cl]) >= self.num_samples:
                    batch = random.choices(self.indices[cl], k=self.num_samples)
                    batches.append(batch)
                    self.quality_checker.check([self.labels[k] for k in batch], batch)
                else:
                    batch = self.indices[cl] + random.choices(initial_indices[cl], k=self.num_samples-len(self.indices[cl]))
                    batches.append(batch)
                    self.quality_checker.check([self.labels[k] for k in batch], batch)
                [self.indices[cl].remove(ind) for ind in self.indices[cl] if ind in batch]
                if len(self.indices[cl]) == 0:
                    centroids.pop(cl)
                remaining_clusters = list(range(len(centroids)))
        print(len(set([s for shit in batches for s in shit])))
        logger.info("Number of blocks {}".format(len(batches)))
        logger.info("Cluster Quality {}".format(sum(self.quality_checker.acc)/len(self.quality_checker.acc)))
        logger.info("Cluster Variety {}".format(sum(self.quality_checker.variety)/len(self.quality_checker.variety)))
        logger.info("Class Cluster Qualiy {}".format({k: sum(v)/len(v) for k, v in self.quality_checker.label_acc.items()}))
        if len(batches)%self.num_classes != 0:
            n = (len(batches)//self.num_classes)*self.num_classes + self.num_classes - len(batches)
            logger.info("Add {} anchors".format(n))
            batches += random.choices(batches, k=n)
        assert len(batches)%self.num_classes == 0

        random.shuffle(batches)
        self.flat_list = [item for anchor in batches for item in anchor]
        logger.info("Samples to be processed".format(len(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSamplerOrig(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.strategy = strategy
        self.max = -1
        self.feature_dict = dict()
        self.epoch = 0
        for inds, samp in samples.items():
            if len(samp) > self.max:
                self.max = len(samp)
        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        self.feature_dict = {k: self.feature_dict[k] for k in sorted(self.feature_dict.keys())}
        dist_mat = np.zeros([len(self.feature_dict), len(self.feature_dict)])
        i = 0
        for ind1, feat_vect1 in self.feature_dict.items():
            j = 0
            for ind2, feat_vect2 in self.feature_dict.items():
                if i > j:
                    dist_mat[i, j] = dist_mat[j, i]
                    j += 1
                    continue
                x = torch.stack(feat_vect1, 0)
                y = torch.stack(feat_vect2, 0)
                m, n = x.size(0), y.size(0)
                x = x.view(m, -1)
                y = y.view(n, -1)
                # use x^TX + y^Ty - 2x^Ty
                dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,
                                                                       m).t()
                dist.addmm_(1, -2, x, y.t())
                dist_mat[i, j] = torch.sum(dist).data.item() / (m*n)
                j += 1
            i += 1
        self.inter_class_dist = dist_mat

    def __iter__(self):
        if (self.strategy == 'alternating' and self.epoch % 5 == 2) or (
                self.strategy == 'only' and self.epoch % 5 == 2) or (
                self.strategy == 'pre' and self.epoch % 5 == 2) or (
                self.strategy == 'pre_soft' and self.epoch % 5 == 2):
            print('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()
        # shuffle elements inside each class
        l_inds = {ind: random.sample(sam, len(sam)) for ind, sam in self.samples.items()}
        if self.strategy == 'pre_soft':
            self.epoch += 30

        for c, inds in l_inds.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.max:
                l_inds[c] += [random.choice(choose)]
        # get clostest classes for each class
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()
        for cl in range(indices.shape[0]):
            possible_classes = indices[cl, :].tolist()
            possible_classes.remove(possible_classes.index(cl))
            if self.strategy != 'pre':
                sample_margin = int(len(possible_classes) * (1-(self.epoch/100)))
                classes = np.random.randint(sample_margin, size=self.num_classes-1).tolist()
                cls = [possible_classes[i] for i in classes]
            else:
                cls = possible_classes[:self.num_classes -1]

            cls.append(cl)
            batch = [s for c in cls for s in random.sample(l_inds[c], self.num_samples)]
            batches.append(batch)

        random.shuffle(batches)
        if len(batches) % self.num_classes != 0:
            b = np.random.choice(np.arange(len(batches)), size=len(batches) % self.num_classes, replace=False).tolist()
            [batches.append(batches[m]) for m in b]


        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSampler(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy, m, batch_sampler=None):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.strategy = strategy
        self.m = m
        self.max = -1
        self.feature_dict = dict()
        self.index_dict = dict()
        self.epoch = 0
        print("In distance constructor")
        print(len(samples))
        # add until num_samples
        for c, inds in self.samples.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.num_samples:
                self.samples[c] += [random.choice(choose)]

        self.inter_class_dist = np.ones([len(samples), len(samples)])
        
        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # sort dicts and generate indicator tensor
        self.feats_sorted = defaultdict(list)
        self.index_sorted = defaultdict(list)
        self.indicator = list()
        feats = list()
        for k in sorted(self.feature_dict.keys()):
            # print(len(self.feature_dict[k]), self.feature_dict[k].keys())
            for i, f in self.feature_dict[k].items():
                self.feats_sorted[k].append(f)
                self.index_sorted[k].append(i)
            while len(self.feats_sorted[k]) < self.num_samples:
                ind = random.sample(range(len(self.feats_sorted[k])), 1)
                self.feats_sorted[k].append(self.feats_sorted[k][ind[0]])
                self.index_sorted[k].append(self.index_sorted[k][ind[0]])

            assert len(self.index_sorted[k]) == len(self.feats_sorted[k])
            self.indicator.append([k] * len(self.index_sorted[k]))
            feats.append(self.feats_sorted[k])
        self.indicator = [i for l in self.indicator for i in l]
        feats = [f for c in feats for f in c]

        # self.feature_dict = {k: self.feature_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.index_dict = {k: self.index_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.indicator = torch.tensor([k for k, v in self.feature_dict.items() for i in range(len(v))])

        self.indicator = torch.tensor(self.indicator)

        # stack features and generate pairwise sample dist
        # for f in feats:
        #    print(f, type(f))
        feats = torch.stack(feats)
        m, n = feats.size(0), feats.size(0)
        x = feats.view(m, -1)
        y = feats.view(n, -1)
        # use x^TX + y^Ty - 2x^Ty
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        self.sample_dist = dist

        # generate pairwise class distance
        dist_mat = np.zeros([len(self.feats_sorted), len(self.feats_sorted)])
        for i in self.feats_sorted.keys():
            for j in self.feats_sorted.keys():
                if i > j:
                    dist_mat[i, j] = dist_mat[j, i]
                    j += 1
                    continue
                row = self.indicator == i
                col = self.indicator == j
                r_max, r_min = torch.max(row.nonzero()).item(), \
                               torch.min(row.nonzero()).item()
                c_max, c_min = torch.max(col.nonzero()).item(), \
                               torch.min(col.nonzero()).item()
                mat_ij = dist[r_min:r_max + 1, c_min:c_max + 1]
                dist_ij = torch.sum(mat_ij).data.item() / (
                            mat_ij.size(0) * mat_ij.size(0))
                dist_mat[i, j] = dist_ij
                dist_mat[j, i] = dist_ij
        self.inter_class_dist = dist_mat

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
        self.get_inter_class_distances()
        
        #if (self.strategy == 'alternating' and self.epoch % 5 == 2) or (
        #        self.strategy == 'only' and self.epoch % 5 == 2) or (
        #        self.strategy == 'pre' and self.epoch % 5 == 2) or (
        #        self.strategy == 'pre_soft' and self.epoch % 5 == 2):
        #    logger.info('recompute dist at epoch {}'.format(self.epoch))
        #    self.get_inter_class_distances()
        # sort class distances
        
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()

        # for each anchor class sample hardest classes and hardest features
        for cl in range(indices.shape[0]):
            possible_classes = indices[cl, :].tolist()
            possible_classes.remove(possible_classes.index(cl))
            #print(possible_classes[:self.num_classes], self.inter_class_dist[cl, possible_classes[0]], self.inter_class_dist[cl, possible_classes[1]], self.inter_class_dist[cl, possible_classes[2]], self.inter_class_dist[cl, possible_classes[3]])
            # get classes
            if self.strategy != 'pre':
                sample_margin = max(
                    int(len(possible_classes) * (1 - (self.epoch / self.m))),
                    self.num_classes - 1)
                classes = np.random.choice(sample_margin,
                                            size=self.num_classes - 1, replace=False).tolist()
                cls = [possible_classes[i] for i in classes]
            else:
                cls = possible_classes[:self.num_classes - 1]
            # randomly sample anchor class samples
            ind_cl = [s for s in
                      random.sample(range(len(self.index_sorted[cl])),
                                    self.num_samples)]
            batch = [[self.index_sorted[cl][i] for i in ind_cl]]
            for c in cls:
                # threshold for samples to be sampled
                thresh = self.inter_class_dist[cl, c]

                # extract pairwise sample distance of anchor and sampled class
                row = self.indicator == cl
                col = self.indicator == c
                r_max, r_min = torch.max(row.nonzero()).item(), \
                               torch.min(row.nonzero()).item()
                c_max, c_min = torch.max(col.nonzero()).item(), \
                               torch.min(col.nonzero()).item()
                mat_clc = self.sample_dist[r_min:r_max + 1, c_min:c_max + 1]
                #print(mat_clc)
                # take only the rows corresponding to sampled anchor samples
                mat_clc = [mat_clc[i, :] for i in range(mat_clc.shape[0]) if
                           i in ind_cl]
                if len(mat_clc) == 0:
                    print(mat_clc, ind_cl, r_min, r_max, c_min, c_max)
                    print("Error: sub matrix can not have zero entries")
                    quit()

                mat_clc = torch.stack(mat_clc)
                # samples from sampled class only if distance > than thresh
                possible_samples = torch.unique(
                    (mat_clc > thresh).nonzero()[:, 1])
                if possible_samples.shape[0] >= self.num_samples:
                    ind = random.sample(range(possible_samples.shape[0]),
                                        self.num_samples)
                elif possible_samples.shape[0] == 0:
                    ind = random.sample(range(c_max - c_min + 1),
                                        self.num_samples)
                else:
                    # if not enough samples with distance > tresh --> random
                    other_samps = [i for i in range(c_max - c_min + 1) if
                                   i not in possible_samples]
                    ind = random.sample(other_samps, self.num_samples -
                                        possible_samples.shape[0])
                    [ind.append(possible_samples[i]) for i in
                    range(possible_samples.shape[0])]
                
                samps = [self.index_sorted[c][i] for i in ind]
                batch.append(samps)
            batch = [s for c in batch for s in c]
            batches.append(batch)
        random.shuffle(batches)
        if len(batches) % self.num_classes != 0:
            b = np.random.choice(np.arange(len(batches)), size=len(batches) % self.num_classes, replace=False).tolist()
            [batches.append(batches[m]) for m in b]

        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSamplerEnsure(Sampler):
    def __init__(self, num_classes, num_samples, samples, strategy, m):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.strategy = strategy
        self.m = m
        self.max = -1
        self.feature_dict = dict()
        self.index_dict = dict()
        self.epoch = 0

        # add until num_samples
        for c, inds in self.samples.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.num_samples:
                self.samples[c] += [random.choice(choose)]

        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # sort dicts and generate indicator tensor
        self.feats_sorted = defaultdict(list)
        self.index_sorted = defaultdict(list)
        self.indicator = list()
        feats = list()
        for k in sorted(self.feature_dict.keys()):
            #print(len(self.feature_dict[k]), self.feature_dict[k].keys())
            for i, f in self.feature_dict[k].items():
                self.feats_sorted[k].append(f)
                self.index_sorted[k].append(i)
            while len(self.feats_sorted[k]) < self.num_samples:
                ind = random.sample(range(len(self.feats_sorted[k])), 1)
                self.feats_sorted[k].append(self.feats_sorted[k][ind[0]])
                self.index_sorted[k].append(self.index_sorted[k][ind[0]])

            assert len(self.index_sorted[k]) == len(self.feats_sorted[k])
            self.indicator.append([k]*len(self.index_sorted[k]))
            feats.append(self.feats_sorted[k])
        
        del self.feature_dict

        self.indicator = [i for l in self.indicator for i in l]
        feats = [f for c in feats for f in c]
        # self.feature_dict = {k: self.feature_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.index_dict = {k: self.index_dict[k] for k in sorted(self.feature_dict.keys())}
        # self.indicator = torch.tensor([k for k, v in self.feature_dict.items() for i in range(len(v))])

        self.indicator = torch.tensor(self.indicator)

        # stack features and generate pairwise sample dist
        #for f in feats:
        #    print(f, type(f))
        feats = torch.stack(feats)
        m, n = feats.size(0), feats.size(0)
        x = feats.view(m, -1)
        y = feats.view(n, -1)
        # use x^TX + y^Ty - 2x^Ty
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        self.sample_dist = dist

        # generate pairwise class distance
        dist_mat = np.zeros([len(self.feats_sorted), len(self.feats_sorted)])
        for i in self.feats_sorted.keys():
            for j in self.feats_sorted.keys():
                if i > j:
                    dist_mat[i, j] = dist_mat[j, i]
                    j += 1
                    continue
                row = self.indicator == i
                col = self.indicator == j
                r_max, r_min = torch.max(row.nonzero()).item(), \
                               torch.min(row.nonzero()).item()
                c_max, c_min = torch.max(col.nonzero()).item(), \
                               torch.min(col.nonzero()).item()
                mat_ij = dist[r_min:r_max+1, c_min:c_max+1]
                dist_ij = torch.sum(mat_ij).data.item() / (mat_ij.size(0)*mat_ij.size(0))
                dist_mat[i, j] = dist_ij
                dist_mat[j, i] = dist_ij
        self.inter_class_dist = dist_mat

    def __iter__(self):

        if (self.strategy == 'alternating' and self.epoch % 5 == 2) or (
                self.strategy == 'only' and self.epoch % 5 == 2) or (
                        self.strategy == 'pre' and self.epoch % 5 == 2) or (
                                self.strategy == 'pre_soft' and self.epoch % 5 == 2):
            print('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()

        if self.strategy == 'pre_soft':
            self.epoch = self.epoch + 30
        # sort class distances
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()

        # for each anchor class sample hardest classes and hardest features
        anchor_samples = copy.deepcopy(self.index_sorted)
        for cl in range(indices.shape[0]):
            while len(anchor_samples[cl]) > 0:
                possible_classes = indices[cl, :].tolist()
                possible_classes.remove(possible_classes.index(cl))

                # get classes
                if self.strategy != 'pre':
                    sample_margin = max(int(len(possible_classes) * (1-(self.epoch/self.m))), self.num_classes - 1)
                    classes = np.random.randint(sample_margin, size=self.num_classes-1).tolist()
                    cls = [possible_classes[i] for i in classes]
                else:
                    cls = possible_classes[:self.num_classes -1]

                # randomly sample anchor class samples
                if len(anchor_samples[cl]) < self.num_samples:
                    ind_cl_1 = [self.index_sorted[cl].index(i) for i in anchor_samples[cl]]
                    ind_cl = random.sample(range(len(self.index_sorted[cl])),
                                           self.num_samples - len(anchor_samples[cl]))
                    ind_cl += ind_cl_1
                    anchor_samples[cl] = list()
                else:
                    samps = random.sample(anchor_samples[cl], self.num_samples)
                    ind_cl = [self.index_sorted[cl].index(i) for i in samps]
                    [anchor_samples[cl].remove(sam) for sam in samps]

                #ind_cl = [s for s in random.sample(range(len(anchor_samples[cl])), self.num_samples)]
                batch = [[self.index_sorted[cl][i] for i in ind_cl]]

                for c in cls:
                    # threshold for samples to be sampled
                    thresh = self.inter_class_dist[cl, c]

                    # extract pairwise sample distance of anchor and sampled class
                    row = self.indicator == cl
                    col = self.indicator == c
                    r_max, r_min = torch.max(row.nonzero()).item(), \
                                   torch.min(row.nonzero()).item()
                    c_max, c_min = torch.max(col.nonzero()).item(), \
                                   torch.min(col.nonzero()).item()
                    mat_clc = self.sample_dist[r_min:r_max + 1, c_min:c_max + 1]

                    # take only the rows corresponding to sampled anchor samples
                    mat_clc = [mat_clc[i, :] for i in range(mat_clc.shape[0]) if i in ind_cl]
                    if len(mat_clc) == 0:
                        print(mat_clc, ind_cl, r_min, r_max, c_min, c_max)
                        print("Error: sub matrix can not have zero entries")
                        quit()

                    mat_clc = torch.stack(mat_clc)

                    # samples from sampled class only if distance > than thresh
                    possible_samples = torch.unique((mat_clc > thresh).nonzero()[:, 1])
                    #print(possible_samples, ind_cl, r_min, r_max, c_min, c_max, mat_clc.shape)
                    if possible_samples.shape[0] >= self.num_samples:
                        ind = random.sample(range(possible_samples.shape[0]), self.num_samples)
                    elif possible_samples.shape[0] == 0:
                        ind = random.sample(range(c_max-c_min+1), self.num_samples)
                    else:
                        # if not enough samples with distance > tresh --> random
                        other_samps = [i for i in range(c_max-c_min+1) if i not in possible_samples]
                        ind = random.sample(other_samps, self.num_samples-possible_samples.shape[0])
                        [ind.append(possible_samples[i]) for i in range(possible_samples.shape[0])]

                    samps = [self.index_sorted[c][i] for i in ind]
                    batch.append(samps)
                batch = [s for c in batch for s in c]
                batches.append(batch)
        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        print('SAMPLES')
        print(len(set(self.flat_list)))
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class DistanceSamplerMean(Sampler):
    def __init__(self, num_classes, num_samples, samples):
        print("USING DIST MEAN")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.max = -1
        self.feature_dict = dict()
        self.epoch = 0

        for inds, samp in samples.items():
            if len(samp) > self.max:
                self.max = len(samp)
        self.inter_class_dist = np.ones([len(samples), len(samples)])

    def get_inter_class_distances(self):
        if len(self.feature_dict) == 0:
            return
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        features_mean = [torch.mean(torch.stack(self.feature_dict[k], 0), 0) for
                             k in sorted(self.feature_dict.keys())]
        x = y = torch.stack(features_mean, 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        # use x^TX + y^Ty - 2x^Ty
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,
                                                               m).t()
        dist.addmm_(1, -2, x, y.t())

        self.inter_class_dist = dist.cpu().data.numpy()

    def __iter__(self):
        if self.epoch % 5 == 2:
            print('recompute dist at epoch {}'.format(self.epoch))
            self.get_inter_class_distances()

        # shuffle elements inside each class
        l_inds = {ind: random.sample(sam, len(sam)) for ind, sam in self.samples.items()}

        for c, inds in l_inds.items():
            choose = copy.deepcopy(inds)
            while len(inds) < self.max:
                l_inds[c] += [random.choice(choose)]
        # get clostest classes for each class
        indices = np.argsort(self.inter_class_dist, axis=1)
        batches = list()
        for cl in range(indices.shape[0]):
            possible_classes = indices[cl, :].tolist()
            possible_classes.remove(possible_classes.index(cl))
            cls = possible_classes[:self.num_classes - 1]
            cls.append(cl)
            batch = [s for c in cls for s in random.sample(l_inds[c], self.num_samples)]
            batches.append(batch)

        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class PretraingSampler(Sampler):
    def __init__(self, samples):
        self.samples = samples
        self.flat_list = list()
        self.max = 0

        for inds in samples:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        samples = list(map(lambda a: random.sample(a, len(a)), self.samples))

        for samp in samples:
            choose = copy.deepcopy(samp)
            while len(samp) < self.max:
                samp += [random.choice(choose)]

        self.flat_list = [item for sublist in samples for item in sublist]
        random.shuffle(self.flat_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerAdvanced(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """
    def __init__(self, l_inds, num_classes, num_elements_class, dict_class_distances, iterations_for_epoch):
        self.l_inds = l_inds
        self.num_classes = num_classes
        self.num_elements_class = num_elements_class
        self.batch_size = self.num_classes * self.num_elements_class
        self.flat_list = []
        self.iterations_for_epoch = iterations_for_epoch
        self.dict_class_distances = dict_class_distances

    def __iter__(self):
        self.flat_list = []
        for ii in range(int(self.iterations_for_epoch)):
            temp_list = []

            # get the class
            pivot_class_index = random.randint(0, self.num_classes - 1)
            pivot_class = self.l_inds[pivot_class_index]

            # put the elements of the class in a temp list
            pivot_elements = random.sample(pivot_class, self.num_elements_class)
            temp_list.extend(pivot_elements)

            # get the k nearest neighbors of the class
            other_class_indices = self.dict_class_distances[pivot_class_index][:self.num_classes - 1]

            # for each neighbor, put the elements of it in a temp list
            for class_index in other_class_indices:
                other_class = self.l_inds[class_index]
                # toDO - try/except error if class has less than k elements, in which case get all of them
                other_elements = random.sample(other_class, self.num_elements_class)
                temp_list.extend(other_elements)

            # shuffle the temp list
            random.shuffle(temp_list)
            self.flat_list.extend(temp_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerSuperclass(Sampler):
   """
   l_inds (list of lists)
   cl_b (int): classes in a batch
   n_cl (int): num of obs per class inside the batch
   """
   def __init__(self, l_inds, num_classes, num_elements_class, dict_superclass, iterations_for_epoch):
       self.l_inds = l_inds
       self.num_classes = num_classes
       self.num_elements_class = num_elements_class
       self.flat_list = []
       self.iterations_for_epoch = iterations_for_epoch
       self.dict_superclass = dict_superclass
   def __iter__(self):
       self.flat_list = []
       for ii in range(int(self.iterations_for_epoch)):
           temp_list = []
           # randomly sample the superclass
           superclass = random.choice(list(self.dict_superclass.keys()))
           list_of_potential_classes = self.dict_superclass[superclass]
           # randomly sample k classes for the superclass
           classes = random.sample(list_of_potential_classes, self.num_classes)
           # get the n objects for each class
           for class_index in classes:
               # classes are '141742158611' etc instead of 1, 2, 3, ..., this should be fixed by finding a mapping between two types of names
               class_ = self.l_inds[class_index]
               # check if the number of elements is >= self.num_elements_class
               if len(class_) >= self.num_elements_class:
                   elements = random.sample(class_, self.num_elements_class)
               else:
                   elements = random.choices(class_, k=self.num_elements_class)
               temp_list.extend(elements)
           # shuffle the temp list
           random.shuffle(temp_list)
           self.flat_list.extend(temp_list)
       return iter(self.flat_list)
   def __len__(self):
       return len(self.flat_list)


class CombineSamplerSuperclass2(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, num_classes, num_elements_class, dict_superclass, iterations_for_epoch):
        self.l_inds = l_inds
        self.num_classes = num_classes
        self.num_elements_class = num_elements_class
        self.flat_list = []
        self.iterations_for_epoch = iterations_for_epoch
        self.dict_superclass = dict_superclass

    def __iter__(self):
        self.flat_list = []
        for ii in range(int(self.iterations_for_epoch)):
            temp_list = []

            # randomly sample the superclass
            superclass_1 = random.choice(list(self.dict_superclass.keys()))
            list_of_potential_classes_1 = self.dict_superclass[superclass_1]

            superclass_2 = superclass_1
            while superclass_2 == superclass_1:
                superclass_2 = random.choice(list(self.dict_superclass.keys()))
            list_of_potential_classes_2 = self.dict_superclass[superclass_2]

            # randomly sample k classes for the superclass
            classes = random.sample(list_of_potential_classes_1, self.num_classes // 2)
            classes_2 = random.sample(list_of_potential_classes_2, self.num_classes // 2)
            classes.extend(classes_2)

            # get the n objects for each class
            for class_index in classes:
                # classes are '141742158611' etc instead of 1, 2, 3, ..., this should be fixed by finding a mapping between two types of names
                class_ = self.l_inds[class_index]
                # check if the number of elements is >= self.num_elements_class
                if len(class_) >= self.num_elements_class:
                    elements = random.sample(class_, self.num_elements_class)
                else:
                    elements = random.choices(class_, k=self.num_elements_class)
                temp_list.extend(elements)

            # shuffle the temp list
            random.shuffle(temp_list)
            self.flat_list.extend(temp_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


if __name__ == '__main__':

    feature_dict = {0: [torch.tensor([0.3376, 0.4358, 1.0508, 0.5083, 0.4756, 0.2441]),
                       torch.tensor([0.3079, 0.9692, 0.4834, 0.3521, 0.7603, 0.5337])],
                1: [torch.tensor([0.8047, 0.4968, 0.3809, 0.8354, 0.5464, 0.1128]),
                      torch.tensor([0.4136, 0.4592, 0.8535, 0.4343, 0.8022, 0.5010])],
                2: [torch.tensor([0.4578, 1.0303, 0.5601, 0.3042, 0.0160, 0.1713])]}

    samples = {0: [0, 1], 1: [2, 4], 2: [3]}

    samp = DistanceSampler(num_classes=3, num_samples=5, samples=samples, strategy='alternating', m=100)
    samp.feature_dict = feature_dict
    samp.index_dict = samples
    samp.get_inter_class_distances()
