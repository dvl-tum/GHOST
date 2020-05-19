import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, CombineSamplerAdvanced, \
    CombineSamplerSuperclass, CombineSamplerSuperclass2, PretraingSampler
import numpy as np
import os


def create_loaders(data_root, num_workers, size_batch, num_classes_iter=None,
                   num_elements_class=None, pretraining=False,
                   input_size=224, both=0):
    labels, paths = dataset.load_data(root=data_root, both=both)
    labels = labels[0]
    paths = paths[0]

    if both:
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
                              q.split('_')[0][-5:], q) for q in
                 paths['query']['detected']] + [
                    os.path.join(data_root, 'labeled', 'images',
                                 q.split('_')[0][-5:], q) for q in
                    paths['query']['labeled']]
        gallery = [os.path.join(data_root, 'detected', 'images',
                                g.split('_')[0][-5:], g) for g in
                   paths['bounding_box_test']['detected']] + [
                      os.path.join(data_root, 'labeled', 'images',
                                   g.split('_')[0][-5:], g) for g in
                      paths['bounding_box_test']['labeled']]

    else:
        labels_ev = labels['bounding_box_test'] + labels['query']
        paths_ev = paths['bounding_box_test'] + paths['query']
        query = [os.path.join(data_root, 'images', q.split('_')[0][-5:], q) for
                 q in paths['query']]
        gallery = [os.path.join(data_root, 'images', g.split('_')[0][-5:], g)
                   for g in paths['bounding_box_test']]

    Dataset = dataset.Birds(
        root=data_root,
        labels=labels['bounding_box_train'],
        paths=paths['bounding_box_train'],
        transform=dataset.utils.make_transform(sz_crop=input_size))

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])

    if pretraining:
        sampler = PretraingSampler(list_of_indices_for_each_class)
        drop_last = False
    else:
        sampler = CombineSampler(list_of_indices_for_each_class,
                                 num_classes_iter, num_elements_class)
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

    if pretraining:
        return dl_tr

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=labels_ev,
            paths=paths_ev,
            transform=dataset.utils.make_transform(is_train=False,
                                                   sz_crop=input_size),
            eval_reid=True
        ),
        batch_size=50,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return dl_tr, dl_ev, query, gallery


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


if __name__ == '__main__':
    # test
    roots = ['../../datasets/cuhk03/detected']

    for root in roots:
        dl_tr, dl_ev, q, g = create_loaders(data_root=root,
                                            input_size=224, size_batch=4,
                                            pretraining=False,
                                            num_workers=2, num_classes_iter=2,
                                            num_elements_class=2,
                                            both=1)
        for batch, y, path in dl_ev:
            print(y, path)
            break
        print()
        for batch, y in dl_tr:
            print(y)
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
