import random
import dataset
import torch
from collections import defaultdict
from combine_sampler import BatchSizeSampler, CombineSampler, QueryGuidedSampler
from dataset.MOTdata import get_sequence_class
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import copy

logger = logging.getLogger('GNNReID.DataUtility')


def create_loaders(
        data_root, num_workers, num_classes_iter=None, num_elements_class=None,
        trans='norm', bssampling=None, rand_scales=False,
        add_distractors=False, split='split_3', sz_crop=[384, 128]):

    config = {'bss': bssampling, 'num_workers': num_workers,
              'nci': num_classes_iter, 'nec': num_elements_class,
              'trans': trans}

    if data_root != 'MOT17':
        # get dataset
        labels, paths = dataset.load_data(
            root=data_root, add_distractors=add_distractors)

        # get validation images
        labels, paths = labels[0], paths[0]
        data, data_root = get_single(labels, paths, data_root)
        query, gallery = data[2], data[3]
    else:
        labels = paths = data = query = None

    dl_tr = get_train_dataloader(
        config, labels, paths, data_root, rand_scales, split, sz_crop)
    dl_ev, dl_ev_gnn = get_val_dataloader(
        config, data, data_root, rand_scales=False, split=split, sz_crop=sz_crop)

    if query is None:
        query = dl_ev.dataset.query_paths
        gallery = dl_ev.dataset.gallery_paths

    return dl_tr, dl_ev, query, gallery, dl_ev_gnn


def get_train_dataloader(config, labels, paths, data_root, rand_scales,
        split='split_3', sz_crop=[384, 128]):
    # get dataset
    if data_root == 'MOT17':
        Dataset = get_sequence_class(split=split)
        Dataset = Dataset(mode='train')
    else:
        Dataset = dataset.Birds(root=data_root,
                                labels=labels['bounding_box_train'],
                                paths=paths['bounding_box_train'],
                                trans=config['trans'],
                                rand_scales=rand_scales,
                                sz_crop=sz_crop)

    # get sampler
    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        if key != -2:
            list_of_indices_for_each_class.append(ddict[key])

    sampler = CombineSampler(
        list_of_indices_for_each_class,
        config['nci'],
        config['nec'],
        batch_sampler=config['bss'],
        distractor_idx=Dataset.distractor_idx)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # get dataloader
    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=config['nci']*config['nec'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    return dl_tr


def get_val_dataloader(config, data, data_root, rand_scales=False,
        split='split_3', sz_crop=[384, 128]):
    if data is not None:
        labels_ev, paths_ev, _, _ = data

    # get dataset
    if data_root == 'MOT17':
        dataset_ev = get_sequence_class(split=split)
        dataset_ev = dataset_ev(mode='test')
    else:
        dataset_ev = dataset.Birds(
            root=data_root,
            labels=labels_ev,
            paths=paths_ev,
            trans=config['trans'],
            eval_reid=True,
            rand_scales=rand_scales,
            sz_crop=sz_crop)

    dl_ev = torch.utils.data.DataLoader(
        dataset_ev,
        batch_size=50,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    return dl_ev


def get_single(labels, paths, data_root):
    labels_ev = labels['bounding_box_test'] + labels['query']
    paths_ev = paths['bounding_box_test'] + paths['query']

    query = [
        os.path.join(
            data_root,
            'images',
            '{:05d}'.format(int(q.split('_')[0])), q) 
        for q in paths['query']]
    gallery = [
        os.path.join(
            data_root,
            'images',
            '{:05d}'.format(int(g.split('_')[0])), g)
        for g in paths['bounding_box_test']]

    data = (labels_ev, paths_ev, query, gallery)

    return data, data_root




