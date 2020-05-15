from torch.utils.data.sampler import Sampler
import random
import copy


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
            n_els = self.max - len(inds) + 1  # take out 1?
            inds.extend(inds[:n_els])  # max + 1

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
