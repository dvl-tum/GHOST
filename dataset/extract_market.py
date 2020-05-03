import os.path as osp
import numpy as np
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


class Market1501(DataPreprocessor):
    url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
    md5 = '65005ab7d12ec1c44de4eeafe813e68a'

    def __init__(self, root, split_id=0, num_train=100, download=True):
        super(Market1501, self).__init__(root, split_id=split_id)
        self.exdir = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        print(num_train)
        self.load(num_train)
        if self.exdir is not None:
            shutil.rmtree(self.exdir)

    def register(self, subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
        print(osp.join(self.exdir, 'Market-1501-v15.09.15', subdir, '*.jpg'))
        fpaths = sorted(glob(osp.join(self.exdir, 'Market-1501-v15.09.15', subdir, '*.jpg')))
        pids = set()
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored

            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= cam <= 6
            cam -= 1
            pids.add(pid)
            fname = ('{:08d}_{:02d}_{:04d}.jpg'
                     .format(pid, cam, len(self.identities[pid][cam])))
            self.identities[pid][cam].append(fname)
            person_dir = osp.join(self.images_dir, '{:05d}'.format(pid))
            mkdir_if_missing(person_dir)
            shutil.copy(fpath, osp.join(person_dir, fname))
        return pids

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        raw_dir = osp.join(self.root, 'images')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(osp.dirname(self.root), 'Market-1501-v15.09.15.zip')
        if osp.isfile(fpath) and \
                hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        self.exdir = osp.join(osp.dirname(self.root), 'Market-1501-v15.09.15')
        if not osp.isdir(self.exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=self.exdir)

        # Format
        mkdir_if_missing(self.images_dir)
        print("HELLO")
        # 1501 identities (+1 for background) with 6 camera views each
        self.identities = [[[] for _ in range(6)] for _ in range(1502)]
        print("HELLO")
        trainval_pids = self.register('bounding_box_train')
        gallery_pids = self.register('bounding_box_test')
        query_pids = self.register('query')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'Market1501', 'shot': 'multiple', 'num_cameras': 6,
                'identities': self.identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

