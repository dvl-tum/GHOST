import random
import data
import torch
from collections import defaultdict
from .combine_sampler import CombineSampler
# from .MOTdata import MOTReIDDataset
import numpy as np
import os
from data.MOTdata import MOTReIDDataset
from data.ReIDdata import ReIDDataset 

def create_loaders(
        dataset_config, num_classes_iter, num_elements_class):
    
    if dataset_config.dataset_short != 'MOT17':
        # get dataset
        labels, paths = data.load_data(
            root=dataset_config.dataset_path,
            add_distractors=dataset_config.add_distractors)

        # combine paths and labels of query and gallery
        labels_ev, paths_ev, query, gallery = \
            combine_query_gallery(labels, paths, dataset_config.dataset_path)
        # train labels and paths
        labels_tr = labels['bounding_box_train']
        paths_tr = paths['bounding_box_train']

    else:
        labels_tr = paths_tr = labels_ev = paths_ev = query = gallery = None

    # get train loader
    dl_tr = get_train_dataloader(
        dataset_config,
        labels_tr,
        paths_tr,
        num_classes_iter,
        num_elements_class)

    # get val Loader
    dl_ev = get_val_dataloader(
        dataset_config, labels_ev, paths_ev)

    if query is None:
        query = dl_ev.dataset.query_paths
        gallery = dl_ev.dataset.gallery_paths

    return dl_tr, dl_ev, query, gallery


def get_train_dataloader(dataset_config, labels, paths, nci, nec):
    # get dataset
    if dataset_config.dataset_short == 'MOT17':
        # Dataset = get_sequence_class(split=dataset_config.split)
        Dataset = MOTReIDDataset(
            root=dataset_config.dataset_path,
            split=dataset_config.split,
            mode='train',
            trans=dataset_config.trans,
            rand_scales=dataset_config.rand_scales,
            sz_crop=dataset_config.sz_crop)
    else:
        Dataset = ReIDDataset(
            root=dataset_config.dataset_path,
            labels=labels,
            paths=paths,
            trans=dataset_config.trans,
            rand_scales=dataset_config.rand_scales,
            sz_crop=dataset_config.sz_crop)

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
        nci,
        nec,
        batch_sampler=dataset_config.bssampling,
        distractor_idx=Dataset.distractor_idx)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # get dataloader
    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=nci*nec,
        shuffle=False,
        sampler=sampler,
        num_workers=dataset_config.nb_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    return dl_tr


def get_val_dataloader(dataset_config, labels_ev, paths_ev):
    # get dataset
    if dataset_config.dataset_short == 'MOT17':
        # dataset_ev = get_sequence_class(split=dataset_config.split)
        dataset_ev = MOTReIDDataset(
            root=dataset_config.dataset_path,
            split=dataset_config.split,
            mode='test',
            trans=dataset_config.trans,
            rand_scales=dataset_config.rand_scales,
            sz_crop=dataset_config.sz_crop,
            eval_reid=True)
    else:
        dataset_ev = ReIDDataset(
            root=dataset_config.dataset_path,
            labels=labels_ev,
            paths=paths_ev,
            trans=dataset_config.trans,
            eval_reid=True,
            rand_scales=dataset_config.rand_scales,
            sz_crop=dataset_config.sz_crop)

    dl_ev = torch.utils.data.DataLoader(
        dataset_ev,
        batch_size=50,
        shuffle=False,
        num_workers=dataset_config.nb_workers,
        pin_memory=True,
        drop_last=False
    )

    return dl_ev


def combine_query_gallery(labels, paths, dataset_path):
    labels_ev = labels['bounding_box_test'] + labels['query']
    paths_ev = paths['bounding_box_test'] + paths['query']

    query = [
        os.path.join(
            dataset_path,
            'images',
            '{:05d}'.format(int(q.split('_')[0])), q) 
        for q in paths['query']]
    gallery = [
        os.path.join(
            dataset_path,
            'images',
            '{:05d}'.format(int(g.split('_')[0])), g)
        for g in paths['bounding_box_test']]

    return labels_ev, paths_ev, query, gallery




