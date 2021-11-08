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


def create_loaders(data_root, num_workers, num_classes_iter=None, 
        num_elements_class=None, mode='single', trans='norm', 
        distance_sampler='only', val=0, seed=0, bssampling=None, 
        rand_scales=False, add_distractors=False, split='split_3'):

    config = {'bss': bssampling,'num_workers': num_workers,
              'nci': num_classes_iter, 'nec': num_elements_class,
              'trans': trans, 'ds': distance_sampler, 'mode': mode}

    if data_root != 'MOT17':
        # get dataset
        labels, paths = dataset.load_data(root=data_root, mode=mode, val=val,
                                            seed=seed, add_distractors=add_distractors)

        # get validation images
        labels, paths = labels[0], paths[0]
        data, data_root = get_validation_images(mode, labels, paths, data_root)
        query, gallery = data[2], data[3]
    else:
        labels = paths = data = query = None

    dl_tr = get_train_dataloader(config, labels, paths, data_root, rand_scales, split)
    dl_ev, dl_ev_gnn = get_val_dataloader(config, data, data_root, rand_scales=False, split=split)

    if query is None:
        query = dl_ev.dataset.query_paths
        gallery = dl_ev.dataset.gallery_paths

    return dl_tr, dl_ev, query, gallery, dl_ev_gnn


def get_train_dataloader(config, labels, paths, data_root, rand_scales, split='split_3'):
    # get dataset
    if data_root == 'MOT17':
        Dataset = get_sequence_class(split=split)
        Dataset = Dataset(mode='train')
    elif config['mode'] != 'all':
        Dataset = dataset.Birds(root=data_root,
                                labels=labels['bounding_box_train'],
                                paths=paths['bounding_box_train'],
                                trans=config['trans'], rand_scales=rand_scales)
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
        if key != -2:
            list_of_indices_for_each_class.append(ddict[key])

    sampler = CombineSampler(list_of_indices_for_each_class,
                                config['nci'], config['nec'],
                                batch_sampler=config['bss'], 
                                distractor_idx=Dataset.distractor_idx)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    # get dataloader
    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=config['nci']*config['nec'],
        shuffle=False,
        sampler=sampler,
        num_workers=0, #config['num_workers'],
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        #generator=g
    )
    
    return dl_tr


def get_val_dataloader(config, data, data_root, rand_scales=False, split='split_3'):
    if data is not None:
        labels_ev, paths_ev, query, gallery = data

    # get dataset
    if data_root == 'MOT17':
        dataset_ev = get_sequence_class(split=split)
        dataset_ev = dataset_ev(mode='test')

    elif config['mode'] != 'all':
        dataset_ev = dataset.Birds(
            root=data_root,
            labels=labels_ev,
            paths=paths_ev,
            trans=config['trans'],
            eval_reid=True,
            rand_scales=rand_scales)

    else:
        dataset_ev = dataset.All(
            root - data_root,
            labels=labels_ev,
            paths=paths_ev,
            trans=config['trans'],
            eval_reid=True
        )

    # dataloader
    if 'gnn' in config['mode'] or 'queryguided' in config['mode'].split('_'):
        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])

        # for batchwise evaluation
        if 'gnn' in config['mode'] and not 'queryguided' in config['mode'].split('_'):
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

        # for evaluation for query guided attention
        elif 'queryguided' in config['mode'].split('_'):
            sampler = QueryGuidedSampler(batch_size=256)
            dl_ev_gnn = torch.utils.data.DataLoader(
                dataset_ev,
                batch_size=256,
                shuffle=False,
                sampler=sampler,
                num_workers=0,
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
            pin_memory=True,
            drop_last=False
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


def get_single(labels, paths, data_root, sample_sub=True):
    logger.info("Sampling subset {}".format(sample_sub))
    if sample_sub:
        import random
        random.seed(0)
        gallery = labels['bounding_box_test']
        query = labels['query']

        query_ddict = defaultdict(list)
        for i, q in enumerate(query):
            query_ddict[q].append(i)
        query_ids = random.sample(list(query_ddict.keys()), k=300)
        query_samps = [random.choice(query_ddict[q]) for q in query_ids]

        gallery_ddict = defaultdict(list)
        for i, g in enumerate(gallery):
            gallery_ddict[g].append(i)

        gallery_samps = list()
        for g in gallery_ddict.values():
            random.shuffle(g)
            gallery_samps.extend(g[:3])
        # gallery_samps = [s for g in gallery_ddict.values() for s in random.sample(g, k=3)]

        labels_ev = [l for i, l in enumerate(labels['bounding_box_test']) if i in gallery_samps] \
            + [l for i, l in enumerate(labels['query']) if i in query_samps]
        paths_ev = [p for i, p in enumerate(paths['bounding_box_test']) if i in gallery_samps] \
             + [p for i, p in enumerate(paths['query']) if i in query_samps]
        
        query = [os.path.join(data_root, 'images',
            '{:05d}'.format(int(q.split('_')[0])), q) for i, q 
            in enumerate(paths['query']) if i in query_samps]
        gallery = [os.path.join(data_root, 'images',
                '{:05d}'.format(int(g.split('_')[0])), g) for i, g 
                in enumerate(paths['bounding_box_test']) if i in gallery_samps]

    else:
        labels_ev = labels['bounding_box_test'] + labels['query']
        paths_ev = paths['bounding_box_test'] + paths['query']
    
        query = [os.path.join(data_root, 'images',
                '{:05d}'.format(int(q.split('_')[0])), q) for q 
                in paths['query']]
        gallery = [os.path.join(data_root, 'images',
                '{:05d}'.format(int(g.split('_')[0])), g) for g 
                in paths['bounding_box_test']]

    data = (labels_ev, paths_ev, query, gallery)

    print(len(labels_ev), len(paths_ev), len(query), len(gallery))

    return data, data_root




