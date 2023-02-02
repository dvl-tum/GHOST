import os
import os.path as osp


def make_dir(dir):
    if not osp.isdir(dir):
        os.makedirs(dir)

class args(object):
    def __init__(self, d):
        self.__dict__ = d