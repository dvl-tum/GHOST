import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, CombineSamplerAdvanced, \
    CombineSamplerSuperclass, CombineSamplerSuperclass2, PretraingSampler
import numpy as np
import pickle
import dataset.extract_market as extract_market
import dataset.extract_cuhk03 as extract_cuhk03
import os
import json
import random
import copy


def create_loaders(data_root, is_extracted, num_workers, num_classes_iter,
                   num_elements_class, size_batch):
    if data_root.split('/')[-1] == 'Market':
        market = dataset.Market1501(root=data_root)
        labels_train = market.split['trainval']
        query = market.split['query']
        gallery = market.split['gallery']
        labels_val = list(set(query + gallery))

    elif data_root.split('/')[-1] == 'cuhk03':
        cuhk03 = dataset.CUHK03(root=data_root)
        labels_train = cuhk03.split['trainval']
        query = cuhk03.split['query']
        gallery = cuhk03.split['gallery']
        labels_val = list(set(query + gallery))

    Dataset = dataset.Birds(
        root=data_root,
        labels=labels_train,
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
        sampler=CombineSampler(list_of_indices_for_each_class,
                               num_classes_iter, num_elements_class),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=labels_val,
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False),
            eval_reid=True
        ),
        batch_size=50,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return dl_tr, dl_ev, query, gallery


def create_dataloaders_pretraining(input_size=224, data_root=None,
                                   batch_size=64,
                                   oversampling=True, train_percentage=1):
    # TODO combine_sampler for pre-training
    if data_root.split('/')[-1] == 'Market':
        market = dataset.Market1501(root=data_root)
        labels_train = market.split['trainval']

    elif data_root.split('/')[-1] == 'cuhk03':
        cuhk03 = dataset.CUHK03(root=data_root)
        labels_train = cuhk03.split['trainval']
    else:
        print("unknown dataset name - End.")
        quit()

    Dataset = dataset.DataSetPretraining(
        root=os.path.join(data_root, 'images'),
        labels=labels_train,
        transform=dataset.utils.make_transform(sz_crop=input_size))

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])


    train = torch.utils.data.DataLoader(Dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        sampler=PretraingSampler(list_of_indices_for_each_class),
                                        num_workers=4,
                                        pin_memory=True)

    return train



def get_labeled_and_unlabeled_points(labels, num_points_per_class,
                                     num_classes=100):
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
