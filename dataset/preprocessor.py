import json
import os
import os.path as osp
import numpy as np


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

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


class DataPreprocessor(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.split = None
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_train=0.3):
        # load split
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

        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))

