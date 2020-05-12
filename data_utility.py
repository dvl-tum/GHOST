import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, CombineSamplerAdvanced, CombineSamplerSuperclass, CombineSamplerSuperclass2
import numpy as np
import pickle
import dataset.extract_market as extract_market
import dataset.extract_cuhk03 as extract_cuhk03
import os
import json
import random
import copy

def create_loaders(data_root, is_extracted, num_workers, num_classes_iter, num_elements_class, size_batch):
    if data_root.split('/')[-1] == 'Market':
        with open(os.path.join(data_root, 'splits.json'), 'r') as f:
            obj = json.load(f)[0]

        labels_train = obj['trainval']
        labels_val = list(set(obj['query'] + obj['gallery']))
        query = obj['query']
        gallery = obj['gallery']

    elif data_root.split('/')[-1] == 'cuhk03':
        with open(os.path.join(data_root, 'splits.json'), 'r') as f:
            obj = json.load(f)[0]

        labels_train = obj['trainval']
        labels_val = list(set(obj['query'] + obj['gallery']))
        query = obj['query']
        gallery = obj['gallery']

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
        sampler=CombineSampler(list_of_indices_for_each_class, num_classes_iter, num_elements_class),
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


def create_dataloaders_pretraining(input_size=224, data_dir=None, batch_size=64,
                                   oversampling=True, train_percentage=1):
    # TODO combine_sampler for pre-training
    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        obj = json.load(f)[0]
    train_indices = obj['trainval']

    image_datasets = get_datasets(input_size, train_indices, data_dir, oversampling, train_percentage)

    train = torch.utils.data.DataLoader(image_datasets['train'],
                                        batch_size=batch_size, shuffle=True,
                                        num_workers=4)
    if len(image_datasets['val']) != 0:
        val = torch.utils.data.DataLoader(image_datasets['val'],
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=4)
    else: val = None

    return train, val


def get_datasets(train_indices, data_dir, oversampling, train_percentage, sz_crop=224):

    # get samples
    train, val, labels_train, labels_val, map = get_samples(train_indices, data_dir, oversampling, train_percentage)
    import dataset.utils as utils
    # TODO: input size and mean make_transform()
    data_transforms = utils.make_transform(sz_crop=sz_crop)
    data_transforms = {'train': data_transforms, 'val': data_transforms}

    print("Initializing Datasets...")
    train_dataset = dataset.DataSetPretraining(root=os.path.join(data_dir, 'images'),
                            labels=labels_train,
                            file_names=train,
                            transform=data_transforms['train'])
    val_dataset = dataset.DataSetPretraining(root=os.path.join(data_dir, 'images'),
                          labels=labels_val, file_names=val,
                          transform=data_transforms['val'])

    return {'train': train_dataset, 'val': val_dataset}


def get_samples(train_indices, data_dir, oversampling, train_percentage):
    # get train and val samples and split
    train = list()
    labels_train = list()
    val = list()
    labels_val = list()

    samps = list()
    for ind in train_indices:
        samples = os.listdir(
            os.path.join(data_dir, 'images', "{:05d}".format(ind)))
        samples = [os.path.join(os.path.join(data_dir, 'images', "{:05d}".format(ind)), samp) for samp in samples]
        samps.append(samples)

    max_num = max([len(c) for c in samps])

    random.seed(40)
    for i, samples in enumerate(samps):
        num_train = int(train_percentage * len(samples))
        train_samps = samples[:num_train]
        val_samps = samples[num_train:]

        if oversampling:
            choose_train = copy.deepcopy(train_samps)
            while len(train_samps) < int(max_num * train_percentage):
                train_samps += [random.choice(choose_train)]

            choose_val = copy.deepcopy(val_samps)
            while len(val_samps) < int(max_num * (1 - train_percentage)):
                val_samps += [random.choice(choose_val)]

            for v in val_samps:
                if v in train_samps:
                    print.error(
                        "Sample of validation set in training set - End.")
                    quit()

        train.append(train_samps)
        val.append(val_samps)

        labels_train.append([train_indices[i]] * len(train_samps))
        labels_val.append([train_indices[i]] * len(val_samps))
    train = [t for classes in train for t in classes]
    val = [t for classes in val for t in classes]
    # mapping of labels from 0 to num_classes
    map = {class_ind: i for i, class_ind in enumerate(train_indices)}
    labels_train = [map[t] for classes in labels_train for t in classes]
    labels_val = [map[t] for classes in labels_val for t in classes]

    return train, val, labels_train, labels_val, map


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

