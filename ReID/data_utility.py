import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, DistanceSampler, PretraingSampler,\
    KReciprocalSampler, ClusteringSampler
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import copy

logger = logging.getLogger('GNNReID.DataUtility')


def create_loaders(data_root, num_workers, num_classes_iter=None, 
        num_elements_class=None, mode='single', trans='norm', 
        distance_sampler='only', val=0, seed=0, bssampling=None):

    config = {'bss': bssampling,'num_workers': num_workers,
              'nci': num_classes_iter, 'nec': num_elements_class,
              'trans': trans, 'ds': distance_sampler, 'mode': mode}

    labels, paths = dataset.load_data(root=data_root, mode=mode, val=val,
                                          seed=seed)
    labels, paths = labels[0], paths[0]
    
    data, data_root = get_validation_images(mode, labels, paths, data_root)
    query, gallery = data[2], data[3]

    dl_tr = get_train_dataloader(config, labels, paths, data_root)
    dl_ev, dl_ev_gnn = get_val_dataloader(config, data, data_root)

    return dl_tr, dl_ev, query, gallery, dl_ev_gnn


def get_train_dataloader(config, labels, paths, data_root):
    # get dataset
    if config['mode'] != 'all':
        Dataset = dataset.Birds(root=data_root,
                                labels=labels['bounding_box_train'],
                                paths=paths['bounding_box_train'],
                                trans=config['trans'])
    else:
        Dataset = dataset.All(root=data_root,
                              labels=labels['bounding_box_train'],
                              paths=paths['bounding_box_train'],
                              trans=config['trans'])
    
    # get sampler
    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])

    if distance_sampler != 'no':
        sampler = DistanceSampler(config['nci'], config['nce'], ddict,
                                  config['ds'], batch_sampler=config['bss'])
    else:
        sampler = CombineSampler(list_of_indices_for_each_class,
                                 config['nci'], config['nce'],
                                 batch_sampler=config['bss'])
    
    # get dataloader
    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=config['nci']*config['nec'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True
    )
    
    return dl_tr


def get_val_dataloader(config, data, data_root, trans):
    labels_ev, paths_ev, query, gallery = data
    
    # get dataset
    if config['mode'] != 'all':
        dataset_ev = dataset.Birds(
            root=data_root,
            labels=labels_ev,
            paths=paths_ev,
            trans=config['trans'],
            eval_reid=True)

    else:
        dataset_ev = dataset.All(
            root - data_root,
            labels=labels_ev,
            paths=paths_ev,
            trans=config['trans'],
            eval_reid=True
        )

    # dataloader
    if 'gnn' in config['mode'].split('_') or 'pseudo' in config['mode'].split('_'):
        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])

        if 'gnn' in mode.split('_'):
            sampler = CombineSampler(list_of_indices_for_each_class,
                                 config['nci'], config['nec'])
            dl_ev = torch.utils.data.DataLoader(
                dataset_ev,
                batch_size=config['nci']*config['nec'],
                shuffle=False,
                sampler=sampler,
                num_workers=config['num_workers'],
                drop_last=True,
                pin_memory=True
            )
            dl_ev_gnn = None

        elif 'pseudo' in mode.split('_'):
            sampler = PseudoSamplerV(config['nci'], config['nec'])
            dl_ev_gnn = torch.utils.data.DataLoader(
                dataset_ev,
                batch_size=config['nci']*config['nec'],
                shuffle=False,
                sampler=sampler,
                num_workers=1,
                drop_last=True,
                pin_memory=True)
            dl_ev = torch.utils.data.DataLoader(
                copy.deepcopy(dataset_ev),
                batch_size=64,
                shuffle=True,
                num_workers=1,
                pin_memory=True)

    else:
        dl_ev = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        dl_ev_gnn = None
    
    return dl_ev, dl_ev_gnn


def get_validation_images(mode, labels, paths, data_root):
    if mode == 'both':
        data, data_root = get_labeled_and_detected(labels, paths, data_root)
    elif mode == 'all':
        data, data_root = get_market_and_cuhk03(labels, paths, data_root)
    else:
        data, data_root = get_single(labels, paths, data_root)

    return data, data_root


def get_labeled_and_detected(labels, paths, data_root):
    data_root = os.path.dirname(data_root)
    labels_ev = {'detected': labels['bounding_box_test']['detected'] +
                            labels['query']['detected'],
                'labeled': labels['bounding_box_test']['labeled'] +
                            labels['query']['labeled']}

    paths_ev = {'detected': paths['bounding_box_test']['detected'] +
                            paths['query']['detected'],
                'labeled': paths['bounding_box_test']['labeled'] +
                            paths['query']['labeled']}

    query = [os.path.join(data_root, 'detected', 'images',
                '{:05d}'.format(int(q.split('_')[0])), q) for q
                in paths['query']['detected']] + \
            [os.path.join(data_root, 'labeled', 'images',
                '{:05d}'.format(int(q.split('_')[0])), q) for q 
                in paths['query']['labeled']]

    gallery = [os.path.join(data_root, 'detected', 'images',
                '{:05d}'.format(int(g.split('_')[0])), g) for g
                in paths['bounding_box_test']['detected']] + \
            [os.path.join(data_root, 'labeled', 'images',
                '{:05d}'.format(int(g.split('_')[0])), g) for g 
                in paths['bounding_box_test']['labeled']]

    data = (labels_ev, paths_ev, query, gallery)

    return data, data_root


def get_market_and_cuhk03(labels, paths, data_root):
    data_root = os.path.dirname(data_root)
    labels_ev = {'cuhk03': labels['bounding_box_test']['cuhk03'] +
                            labels['query']['cuhk03'],
                'market': labels['bounding_box_test']['market'] +
                            labels['query']['market']}

    paths_ev = {'cuhk03': paths['bounding_box_test']['cuhk03'] +
                            paths['query']['cuhk03'],
                'market': paths['bounding_box_test']['market'] +
                            paths['query']['market']}

    query = [os.path.join(data_root, 'cuhk03', 'detected', 'images',
                '{:05d}'.format(int(q.split('_')[0])), q) for q
                 in paths['query']['cuhk03']] + \
                [os.path.join(data_root, 'Market-1501-v15.09.15', 'images',
                '{:05d}'.format(int(q.split('_')[0])), q) for q 
                in paths['query']['labeled']]
    gallery = [os.path.join(data_root, 'cuhk03', 'detected', 'images',
                '{:05d}'.format(int(g.split('_')[0])), g) for g
                in paths['bounding_box_test']['cuhk03']] + \
                [os.path.join(data_root, 'Market-1501-v15.09.15', 'images',
                '{:05d}'.format(int(g.split('_')[0])), g) for g 
                in paths['bounding_box_test']['market']]

    data = (labels_ev, paths_ev, query, gallery)

    return data, data_root


def get_single(labels, paths, data_root):
    labels_ev = labels['bounding_box_test'] + labels['query']
    paths_ev = paths['bounding_box_test'] + paths['query']
    
    query = [os.path.join(data_root, 'images',
            '{:05d}'.format(int(q.split('_')[0])), q) for q 
            in paths['query']]
    gallery = [os.path.join(data_root, 'images',
            '{:05d}'.format(int(g.split('_')[0])), g) for g 
            in paths['bounding_box_test']]

    data = (labels_ev, paths_ev, query, gallery)

    return data, data_root




