from torch.utils.data.sampler import Sampler
import random
import copy
import torch
import sklearn.metrics.pairwise
from collections import defaultdict
import numpy as np


class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl):
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []

        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        # add elements till every class has the same num of obs
        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.max:
                inds += [random.choice(choose)]

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]

        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class DistanceSampler(Sampler):
    def __init__(self, num_classes, num_samples, samples):
        print("USING DIST")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.max = -1
        self.feature_dict = dict()

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


class DistanceSamplerMean(Sampler):
    def __init__(self, num_classes, num_samples, samples):
        print("USING DIST MEAN")
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.samples = samples
        self.max = -1
        self.feature_dict = dict()

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

    feature_dict = {1008: [torch.tensor([0.3376, 0.4358, 1.0508, 0.5083, 0.4756, 0.2441]),
                       torch.tensor([0.3079, 0.9692, 0.4834, 0.3521, 0.7603, 0.5337])],
                654: [torch.tensor([0.8047, 0.4968, 0.3809, 0.8354, 0.5464, 0.1128]),
                      torch.tensor([0.4136, 0.4592, 0.8535, 0.4343, 0.8022, 0.5010])],
                890: [torch.tensor([0.4578, 1.0303, 0.5601, 0.3042, 0.0160, 0.1713])]}
