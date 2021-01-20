import os
import os.path as osp


def make_dir(dir):
    if not osp.isdir(dir):
        os.makedirs(dir)