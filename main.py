import yaml
import argparse
import torch
import logging
import warnings
import time
import os.path as osp
import os
from tracker import Tracker

logger = logging.getLogger('Tracker')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

warnings.filterwarnings("ignore")


def init_args():
    parser = argparse.ArgumentParser(description='AllReID tracker')
    parser.add_argument('--config_path', type=str,
                        default='config/config.yaml',
                        help='Path to config file')

    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))

    trainer = Tracker(device, time.time(), config['dataset'],
                      config['reid_net'], config['tracker'])
    trainer.train()


if __name__ == '__main__':
    args = init_args()
    main(args)
