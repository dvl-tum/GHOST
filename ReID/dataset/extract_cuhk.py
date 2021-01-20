from __future__ import print_function, absolute_import
import os.path as osp
import imageio
import numpy as np

import h5py

import json
import os
from zipfile import ZipFile

from collections import defaultdict

url = 'https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0'
md5 = '728939e58ad9f0ff53e521857dd8fb43'


def dump_(refs, pid, cam, fnames, data_type, root, matdata):
    image_dir = os.path.join(root, data_type, 'images')
    for ref in refs:
        img = matdata[ref][:].T
        if img.size == 0 or img.ndim < 2: break
        fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(fnames))
        person_dir = os.path.join(image_dir, '{:05d}'.format(pid))
        if not os.path.isdir(person_dir):
            os.makedirs(person_dir)
        imageio.imwrite(osp.join(person_dir, fname), img)
        fnames.append(fname)


def cuhk03(root: str = None, check_zip: str = None):
    root = os.path.dirname(root)

    if not osp.isdir(root):
        print("Extracting zip file")
        with ZipFile(check_zip) as z:
            z.extractall(os.path.dirname(check_zip))
        os.rename(check_zip[:-4], root)

    matdata = h5py.File(osp.join(root, 'cuhk-03.mat'), 'r')

    for data_type in ['labeled', 'detected']:
        if not os.path.isdir(os.path.join(root, data_type, 'images')):
            os.makedirs(os.path.join(root, data_type, 'images'))

        identities = list()
        view_counts = list()
        for data in matdata[data_type][0]:
            data = matdata[data][:].T
            view_counts.append(data.shape[0])
            for i in range(data.shape[0]):
                pid = len(identities)
                images = [[], []]
                dump_(data[i, :5], pid, 0, images[0], data_type, root, matdata)
                dump_(data[i, 5:], pid, 1, images[1], data_type, root, matdata)
                identities.append(images)

        # Save training and test splits
        splits = []
        vid_offsets = np.r_[0, np.cumsum(view_counts)]
        # shift image ids --> view count 0, 1, 2 originally, rest add. 107 img
        for ref in matdata['testsets'][0]:
            test_info = matdata[ref][:].T.astype(np.int32)
            test_pids = sorted(
                [int(vid_offsets[i - 1] + j - 1) for i, j in test_info])
            trainval_pids = list(
                set(range(vid_offsets[-1])) - set(test_pids))

            split = {'bounding_box_train': trainval_pids,
                     'query': test_pids,
                     'bounding_box_test': test_pids}
            splits.append(split)

        splits_paths = list() # paths of all splits
        splits_new = list() # labels of all splits

        # iterate over splits
        for split in splits:
            # paths and labels for one split
            split_paths, split_new = defaultdict(list), defaultdict(list)
            for type in ['bounding_box_train', 'bounding_box_test', 'query']:
                for img in os.listdir(os.path.join(root, data_type, 'images')):
                    if int(img) in split[type]:
                        split_paths[type].extend(os.listdir(os.path.join(root, data_type, 'images', img)))
                        split_new[type].extend([int(i.split('_')[0]) for i in os.listdir(os.path.join(root, data_type, 'images', img))])
                # check if len labels is equally long as paths
                assert len(split_paths[type]) == len(split_new[type])

            # check if no ids in train that are in test or query
            assert set(split_new['bounding_box_train']).isdisjoint(
                set(split_new['query']))
            assert set(split_new['bounding_box_train']).isdisjoint(
                set(split_new['bounding_box_test']))

            # add split dicts to list
            splits_paths.append(split_paths)
            splits_new.append(split_new)

        assert len(splits_paths) == len(splits_new)

        # store paths to files in json file
        with open(os.path.join(os.path.join(root, data_type), 'info.json'), 'w') as file:
            json.dump(splits_paths, file)

        # store paths to files in json file
        with open(os.path.join(os.path.join(root, data_type), 'labels.json'), 'w') as file:
            json.dump(splits_new, file)

