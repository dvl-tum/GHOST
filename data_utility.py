import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, CombineSamplerAdvanced, \
    CombineSamplerSuperclass, CombineSamplerSuperclass2, PretraingSampler, \
    DistanceSampler, DistanceSamplerMean, DistanceSamplerOrig, TrainTestCombi, \
    PseudoSampler, PseudoSamplerII, PseudoSamplerIII, PseudoSamplerVI
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import copy

logger = logging.getLogger('GNNReID.DataUtility')

def create_loaders(data_root, num_workers, size_batch, num_classes_iter=None,
                   num_elements_class=None, pretraining=False,
                   input_size=[384, 128], mode='single', trans= 'norm',
                   distance_sampler='only', val=0, m=100, seed=0, magnitude=15,
                   number_aug=0, num_classes=None):
    query, gallery = None, None
    if os.path.basename(data_root) != 'CARS' and os.path.basename(data_root) != 'CUB_200_2011' and os.path.basename(data_root) != 'Stanford_Online_Products':
        labels, paths = dataset.load_data(root=data_root, mode=mode, val=val, seed=seed)
        labels = labels[0]
        paths = paths[0]

    if mode == 'both':
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
                              '{:05d}'.format(int(q.split('_')[0])), q) for q in
                 paths['query']['detected']] + [
                    os.path.join(data_root, 'labeled', 'images',
                                 '{:05d}'.format(int(q.split('_')[0])), q) for q in
                    paths['query']['labeled']]
        gallery = [os.path.join(data_root, 'detected', 'images',
                                '{:05d}'.format(int(g.split('_')[0])), g) for g in
                   paths['bounding_box_test']['detected']] + [
                      os.path.join(data_root, 'labeled', 'images',
                                   '{:05d}'.format(int(g.split('_')[0])), g) for g in
                      paths['bounding_box_test']['labeled']]

    elif mode == 'all':
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
                 in
                 paths['query']['cuhk03']] + [
                    os.path.join(data_root, 'Market-1501-v15.09.15', 'images',
                                 '{:05d}'.format(int(q.split('_')[0])), q) for
                    q in
                    paths['query']['labeled']]
        gallery = [os.path.join(data_root, 'cuhk03', 'detected', 'images',
                                '{:05d}'.format(int(g.split('_')[0])), g) for g
                   in
                   paths['bounding_box_test']['cuhk03']] + [
                      os.path.join(data_root, 'Market-1501-v15.09.15', 'images',
                                   '{:05d}'.format(int(g.split('_')[0])), g)
                      for g in
                      paths['bounding_box_test']['market']]

    elif os.path.basename(data_root) != 'CUB_200_2011' and os.path.basename(data_root) != 'CARS' and os.path.basename(data_root) != 'Stanford_Online_Products':
        labels_ev = labels['bounding_box_test'] + labels['query']
        paths_ev = paths['bounding_box_test'] + paths['query']
        query = [os.path.join(data_root, 'images', '{:05d}'.format(int(q.split('_')[0])), q) for
                 q in paths['query']]
        gallery = [os.path.join(data_root, 'images', '{:05d}'.format(int(g.split('_')[0])), g)
                   for g in paths['bounding_box_test']]

    if mode != 'all' and os.path.basename(data_root) != 'CUB_200_2011' and os.path.basename(data_root) != 'CARS' and os.path.basename(data_root) != 'Stanford_Online_Products':
        Dataset = dataset.Birds(root=data_root,
                                labels=labels['bounding_box_train'],
                                paths=paths['bounding_box_train'],
                                trans=trans,
                                magnitude=magnitude,
                                number_aug=number_aug)
    elif os.path.basename(data_root) == 'CUB_200_2011' or os.path.basename(data_root) == 'CARS' or os.path.basename(data_root) == 'Stanford_Online_Products':
        Dataset = dataset.Birds_DML(
            root=data_root,
            labels=list(range(0, num_classes)),
            transform=trans)
    else:
        Dataset = dataset.All(root=data_root,
                                labels=labels['bounding_box_train'],
                                paths=paths['bounding_box_train'],
                                trans=trans,
                                magnitude=magnitude,
                                number_aug=number_aug)

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])

    if pretraining:
        sampler = PretraingSampler(list_of_indices_for_each_class)
        drop_last = False
    elif distance_sampler == 'orig_pre' or distance_sampler == 'orig_pre_soft' or distance_sampler == 'orig_only' or distance_sampler == 'orig_alternating':
        print(distance_sampler)
        sampler = DistanceSamplerOrig(num_classes_iter, num_elements_class, ddict, distance_sampler, m)
        drop_last = True
    elif distance_sampler == 'pre' or distance_sampler == 'pre_soft' or distance_sampler == 'only' or distance_sampler == 'alternating':
        print(distance_sampler)
        sampler = DistanceSampler(num_classes_iter, num_elements_class, ddict, distance_sampler, m)
        drop_last = True
    else:
        sampler = CombineSampler(list_of_indices_for_each_class,
                                 num_classes_iter, num_elements_class)
        #logger.info("Pseudo for training")
        #sampler = PseudoSamplerIII(num_classes_iter, num_elements_class)

        drop_last = True

    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=size_batch,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    '''
    logger.info("Random Training //////////")
    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    '''

    if pretraining:
        return dl_tr
    if mode != 'all' and mode != 'traintest' and mode != 'traintest_test' and \
            os.path.basename(data_root) != 'CUB_200_2011' and os.path.basename(data_root) != 'CARS' and os.path.basename(data_root) != 'Stanford_Online_Products':
        dataset_ev = dataset.Birds(
                root=data_root,
                labels=labels_ev,
                paths=paths_ev,
                trans=trans,
                eval_reid=True)
    elif (mode == 'traintest' or mode == 'traintest_test') and os.path.basename(data_root) != 'CUB_200_2011' and os.path.basename(data_root) != 'CARS' and os.path.basename(data_root) != 'Stanford_Online_Products':
        dataset_ev = dataset.Birds(
            root=data_root,
            labels=labels['query'],
            paths=paths['query'],
            trans=trans,
            eval_reid=True,
            labels_train=labels['bounding_box_train'],
            paths_train=paths['bounding_box_train'],
            labels_gallery=labels['bounding_box_test'],
            paths_gallery=paths['bounding_box_test'])
    elif (os.path.basename(data_root) == 'CUB_200_2011' or os.path.basename(data_root) == 'CARS' or os.path.basename(data_root) == 'Stanford_Online_Products') and  (mode == 'traintest' or mode == 'traintest_test'):
        if data_root == 'Stanford':
            class_end = 2 * num_classes - 2
        else:
            class_end = 2 * num_classes
        dataset_ev = dataset.Birds_DML(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            labels_train=list(range(0, num_classes)),
            transform=trans,
            eval_reid=True
        )
    elif os.path.basename(data_root) == 'CUB_200_2011' or os.path.basename(data_root) == 'CARS' or os.path.basename(data_root) == 'Stanford_Online_Products':
        if data_root == 'Stanford':
            class_end = 2 * num_classes - 2
        else:
            class_end = 2 * num_classes
        dataset_ev = dataset.Birds_DML(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            transform=trans,
            eval_reid=True
        )
    else:
        dataset_ev = dataset.All(
            root-data_root,
            labels=labels_ev,
            paths=paths_ev,
            trans=trans,
            eval_reid=True
        )


    if mode == 'gnn' or mode == 'gnn_test' or mode == 'gnn_hyper_search':
        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])

        '''if distance_sampler == 'orig_pre' or distance_sampler == 'orig_pre_soft' or distance_sampler == 'orig_only' or distance_sampler == 'orig_alternating':
            print(distance_sampler)
            sampler = DistanceSamplerOrig(num_classes_iter, num_elements_class,
                                          ddict, distance_sampler, m)
            drop_last = True
        elif distance_sampler == 'pre' or distance_sampler == 'pre_soft' or distance_sampler == 'only' or distance_sampler == 'alternating':
            print(distance_sampler)
            sampler = DistanceSampler(num_classes_iter, num_elements_class,
                                      ddict, distance_sampler, m)
            drop_last = True
        else:'''
        sampler = CombineSampler(list_of_indices_for_each_class,
                                 num_classes_iter, num_elements_class)
                                 
        drop_last = True

        dl_ev = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=size_batch,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True
        )

        dl_ev_gnn = None

    elif mode == 'pseudo' or mode == 'pseudo_test' or mode == 'pseudo_hyper_search':
        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])
        
        #sampler = CombineSampler(list_of_indices_for_each_class,
        #                        num_classes_iter, num_elements_class)
        sampler = PseudoSamplerIII(num_classes_iter, num_elements_class)
        #sampler = DistanceSampler(num_classes_iter, num_elements_class,
        #                              ddict, distance_sampler, 1)
        sampler.epoch = 2
        drop_last = True
        print("batch size {}".format(size_batch))
        dl_ev_gnn = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=size_batch,
            shuffle=False,
            sampler=sampler,
            num_workers=1,
            drop_last=drop_last,
            pin_memory=True)
        '''
        print("Random Pseudo")
        dl_ev_gnn = torch.utils.data.DataLoader(
            copy.deepcopy(dataset_ev),
            batch_size=128,
            shuffle=True,
            num_workers=1,
            pin_memory=True )
        '''
        dl_ev = torch.utils.data.DataLoader(
            copy.deepcopy(dataset_ev),
            batch_size=64,
            shuffle=True,
            num_workers=1,
            pin_memory=True )

    elif mode == 'knn' or mode == 'knn_test' or mode == 'knn_hyper_search':
        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])

        #sampler = CombineSampler(list_of_indices_for_each_class,
        #                        num_classes_iter, num_elements_class)
        sampler = PseudoSampler(num_classes_iter, num_elements_class)

        drop_last = True
        print("batch size {}".format(size_batch))
        dl_ev_gnn = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=size_batch,
            shuffle=False,
            sampler=sampler,
            num_workers=1,
            drop_last=drop_last,
            pin_memory=True)
        '''
        print("Random Pseudo")
        dl_ev_gnn = torch.utils.data.DataLoader(
            copy.deepcopy(dataset_ev),
            batch_size=128,
            shuffle=True,
            num_workers=1,
            pin_memory=True )
        '''
        dl_ev = torch.utils.data.DataLoader(
            copy.deepcopy(dataset_ev),
            batch_size=64,
            shuffle=True,
            num_workers=1,
            pin_memory=True )
    
    elif (mode == 'traintest' or mode == 'traintest_test') and \
            os.path.basename(data_root) != 'CUB_200_2011' and os.path.basename(data_root) != 'CARS' and os.path.basename(data_root) != 'Stanford_Online_Products':
        ddict_query = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys_query):
            ddict_query[label].append(idx)
            max_idx_query = idx

        list_of_indices_for_each_class_query = []
        for key in ddict_query:
            list_of_indices_for_each_class_query.append(ddict_query[key])

        ddict_train = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys_train):
            ddict_train[label].append(idx + 1 + max_idx_query) # +1 becuase idx starts from 0
            max_idx_train = idx + 1 + max_idx_query

        list_of_indices_for_each_class_train = []
        for key in ddict_train:
            list_of_indices_for_each_class_train.append(ddict_train[key])
        
        ddict_gallery = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys_gallery):
            ddict_gallery[label].append(idx + 1 + max_idx_train)
        
        list_of_indices_for_each_class_gallery = []
        for key in ddict_gallery:
            list_of_indices_for_each_class_gallery.append(ddict_gallery[key])
        
        sampler_backbone = TrainTestCombi(list_of_indices_for_each_class_query,
                                 num_classes_iter, num_elements_class,
                                 list_of_indices_for_each_class_train,
                                 list_of_indices_for_each_class_gallery, 
                                 backbone=True)

        sampler = TrainTestCombi(list_of_indices_for_each_class_query,
                                 num_classes_iter, num_elements_class,
                                 list_of_indices_for_each_class_train,
                                 list_of_indices_for_each_class_gallery)
        drop_last = True

        dl_ev_gnn = torch.utils.data.DataLoader(
            dataset_ev,
            sampler=sampler,
            batch_size=size_batch,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        dl_ev = torch.utils.data.DataLoader(
            dataset_ev,
            sampler= sampler_backbone,
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

    elif (mode == 'traintest' or mode == 'traintest_test') and \
            (os.path.basename(data_root) == 'CUB_200_2011' or os.path.basename(data_root) != 'CARS' or os.path.basename(data_root) != 'Stanford_Online_Products'):

        ddict_test = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys_test):
            ddict_test[label].append(idx)
            max_idx_test = idx

        list_of_indices_for_each_class_test = []
        for key in ddict_test:
            list_of_indices_for_each_class_test.append(ddict_test[key])

        ddict_train = defaultdict(list)
        for idx, label in enumerate(dataset_ev.ys_train):
            ddict_train[label].append(
                idx + 1 + max_idx_test)  # +1 becuase idx starts from 0
            max_idx_train = idx + 1 + max_idx_test

        list_of_indices_for_each_class_train = []
        for key in ddict_train:
            list_of_indices_for_each_class_train.append(ddict_train[key])

        sampler_backbone = TrainTestCombi(list_of_indices_for_each_class_test,
                                          num_classes_iter, num_elements_class,
                                          list_of_indices_for_each_class_train,
                                          backbone=True)

        sampler = TrainTestCombi(list_of_indices_for_each_class_test,
                                 num_classes_iter, num_elements_class,
                                 list_of_indices_for_each_class_train)
        drop_last = True

        dl_ev_gnn = torch.utils.data.DataLoader(
            dataset_ev,
            sampler=sampler,
            batch_size=size_batch,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        dl_ev = torch.utils.data.DataLoader(
            dataset_ev,
            sampler=sampler_backbone,
            batch_size=64,
            shuffle=False,
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

    return dl_tr, dl_ev, query, gallery, dl_ev_gnn


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


def show_dataset(img, y):
    for i in range(img.shape[0]):
        im = img[i, :, :, :].squeeze()
        x = im.numpy().transpose((1, 2, 0))
        plt.imshow(x)
        plt.axis('off')
        plt.title('Image of label {}'.format(y[i]))
        plt.show()


if __name__ == '__main__':
    # test
    roots = ['../../datasets/Market-1501-v15.09.15',
             '../../datasets/cuhk03/detected']

    for root in roots:
        dl_tr, dl_ev, q, g = create_loaders(data_root=root, size_batch=20,
                                            pretraining=False,
                                            num_workers=2, num_classes_iter=10,
                                            num_elements_class=2,
                                            both=0, trans='appearance',
                                            distance_sampler=1)
        for batch, y, path in dl_ev:
            print(y, path)
            print(batch.shape)
            break
        print()
        for batch, y in dl_tr:
            print(y)
            show_dataset(batch, y)
            break
        print()
        # print(q)

        # print(g)

        dl_tr = create_loaders(data_root=root,
                               input_size=224, size_batch=4, pretraining=True,
                               num_workers=2, num_classes_iter=2,
                               num_elements_class=2
                               )

        for batch, y in dl_tr:
            print(y)
            break
        quit()

    roots = ['../../datasets/cuhk03-np/detected',
             '../../datasets/cuhk03/detected',
             '../../datasets/cuhk03-np/detected',
             '../../datasets/cuhk03-np/labeled',
             '../../datasets/Market-1501-v15.09.15',
             '../../datasets/cuhk03/detected',
             '../../datasets/cuhk03/labeled']
