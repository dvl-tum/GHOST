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

from .preprocessor import write_json
from .preprocessor import mkdir_if_missing
from .preprocessor import read_json
from.preprocessor import DataPreprocessor


class CUHK03(DataPreprocessor):
    url = 'https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0'
    md5 = '728939e58ad9f0ff53e521857dd8fb43'

    def __init__(self, root, split_id=0, num_train=100, download=True):
        super(CUHK03, self).__init__(root, split_id=split_id)
        self.exdir = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_train)
        if self.exdir is not None:
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
