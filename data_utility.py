import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, CombineSamplerAdvanced, CombineSamplerSuperclass, CombineSamplerSuperclass2
import numpy as np
import pickle


def create_loaders(data_root, num_classes, is_extracted, num_workers, num_classes_iter, num_elements_class, size_batch):
    Dataset = dataset.Birds(
        root=data_root,
        labels=list(range(0, num_classes)),
        is_extracted=is_extracted,
        transform=dataset.utils.make_transform())

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])

    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=size_batch,
        shuffle=False,
        sampler=CombineSampler(list_of_indices_for_each_class, num_classes_iter, num_elements_class),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True
    )

    dl_finetune = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dl_train_evaluate = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=150,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return dl_tr, dl_ev, dl_finetune, dl_train_evaluate


def create_loaders_finetune(data_root, num_classes, is_extracted, num_workers, size_batch):

    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
            batch_size=150,
            shuffle=False,
            num_workers=1,
            pin_memory=True
    )

    dl_finetune = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


    return dl_ev, dl_finetune




def create_loaders_new(data_root, num_classes, is_extracted, dict_class_distances, num_classes_iter=0, num_elements_class=0, iterations_for_epoch=200):
    Dataset = dataset.Birds(
       root=data_root,
       labels=list(range(0, num_classes)),
       is_extracted=is_extracted,
       transform=dataset.utils.make_transform())
    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
       ddict[label].append(idx)
    list_of_indices_for_each_class = []
    for key in ddict:
       list_of_indices_for_each_class.append(ddict[key])
    # with open('list_of_indices_for_each_class_stanford.pickle', 'wb') as handle:
    #     pickle.dump(list_of_indices_for_each_class, handle, protocol=pickle.HIGHEST_PROTOCOL)
    list_of_indices_for_each_class = pickle.load(open('list_of_indices_for_each_class_stanford.pickle', 'rb'))
    dl_tr = torch.utils.data.DataLoader(
       Dataset,
       batch_size=num_classes_iter * num_elements_class,
       shuffle=False,
       sampler=CombineSamplerSuperclass2(list_of_indices_for_each_class, num_classes_iter, num_elements_class, dict_class_distances, iterations_for_epoch=iterations_for_epoch),
       num_workers=8,
       drop_last=True,
       pin_memory=True
    )
    return dl_tr


def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):
    labs, L, U = [], [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    for i in range(num_points):
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs.append(labels[i])
            labs_buffer[labels[i]] += 1
    return labs, L, U


def debug_info(gtg, model):
    for name, param in gtg.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # print(name, param.grad.data.norm(2))
                print(name, torch.mean(param.grad.data))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # print(name, param.grad.data.norm(2))
                print(name, torch.mean(param.grad.data))
    print("\n\n\n")

