from __future__ import print_function, absolute_import
import os.path as osp
import imageio
import numpy as np

import h5py
import hashlib
from zipfile import ZipFile

import json
import os
import errno
import re
import hashlib
import shutil
from glob import glob
from zipfile import ZipFile


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

class DataPreprocessor(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_train=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]
        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_train, float):
            num_train = int(round(num * num_train))
        if num_train >= num or num_train < 0:
            raise ValueError("num_train exceeds total identities {}"
                             .format(num))

        train_pids = sorted(trainval_pids[:num_train])
        val_pids = sorted(trainval_pids[num_train:])
        self.split['train'] = train_pids
        self.split['val'] = val_pids

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, relabel=False)
        self.val = _pluck(identities, val_pids, relabel=False)
        self.trainval = _pluck(identities, trainval_pids, relabel=False)
        self.query = _pluck(identities, self.split['query'])
        self.gallery = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))


class CUHK03(DataPreprocessor):
    url = 'https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0'
    md5 = '728939e58ad9f0ff53e521857dd8fb43'

    def __init__(self, root, split_id=0, num_train=100, download=True):
        super(CUHK03, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_train)
        shutil.rmtree(self.exdir)

    def deref(self, ref):
        return self.matdata[ref][:].T

    def dump_(self, refs, pid, cam, fnames):
        for ref in refs:
            img = self.deref(ref)
            if img.size == 0 or img.ndim < 2: break
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(fnames))
            person_dir = os.path.join(self.images_dir, '{:05d}'.format(pid))
            if not os.path.isdir(person_dir):
                os.makedirs(person_dir)
            imageio.imwrite(osp.join(person_dir, fname), img)
            fnames.append(fname)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        raw_dir = osp.join(self.root, 'images')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(osp.dirname(self.root), 'cuhk03_release.zip')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        self.exdir = osp.join(osp.dirname(self.root), 'cuhk03_release')
        if not osp.isdir(self.exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=self.exdir)

        # Format
        mkdir_if_missing(self.images_dir)
        self.matdata = h5py.File(osp.join(self.exdir, 'cuhk03_release', 'cuhk-03.mat'), 'r')

        identities = []
        for labeled, detected in zip(
                self.matdata['labeled'][0], self.matdata['detected'][0]):
            labeled, detected = self.deref(labeled), self.deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                images = [[], []]
                self.dump_(labeled[i, :5], pid, 0, images[0])
                self.dump_(detected[i, :5], pid, 0, images[0])
                self.dump_(labeled[i, 5:], pid, 1, images[1])
                self.dump_(detected[i, 5:], pid, 1, images[1])
                identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'cuhk03', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save training and test splits
        splits = []
        view_counts = [self.deref(ref).shape[0] for ref in self.matdata['labeled'][0]]
        vid_offsets = np.r_[0, np.cumsum(view_counts)]
        for ref in self.matdata['testsets'][0]:
            test_info = self.deref(ref).astype(np.int32)
            test_pids = sorted(
                [int(vid_offsets[i-1] + j - 1) for i, j in test_info])
            trainval_pids = list(set(range(vid_offsets[-1])) - set(test_pids))
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
