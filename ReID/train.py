import yaml
import argparse
import torch
import logging
import warnings
import time
import os.path as osp
import os
import utils.utils as utils
from trainer import Trainer 


logger = logging.getLogger('GNNReID')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

fh = logging.FileHandler('train_reid.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

warnings.filterwarnings("ignore")


def init_args():
    parser = argparse.ArgumentParser(description='Person Re-ID with GNN')
    parser.add_argument('--config_path', type=str, default='config/config_MOT.yaml', help='Path to config file')

    return parser.parse_args() 


def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))

    save_folder_results = 'search_results'
    utils.make_dir(save_folder_results)
    save_folder_nets = 'search_results_net'
    utils.make_dir(save_folder_nets)
    
    splits = [['split_S4', 'split_S11'], ['split_S5', 'split_S9'], ['split_S2', 'split_S10', 'split_S13']]
    weights = ['models/0.8634204275534442resnet50_Market.pth',
               'models/0.8634204275534442resnet50_Market.pth',
               'models/0.8634204275534442resnet50_Market.pth']

    
    '''splits = [['50-50-1+split_S4', '50-50-1+split_S11', '50-50-1+split_S5', '50-50-1+split_S9', '50-50-1+split_S2', '50-50-1+split_S10', '50-50-1+split_S13'],
              ['50-50-2+split_S4', '50-50-2+split_S11', '50-50-2+split_S5', '50-50-2+split_S9', '50-50-2+split_S2', '50-50-2+split_S10', '50-50-2+split_S13']]
    weights = ['0.9625resnet50_MOT17.pth',
               '0.9541284403669725resnet50_MOT17.pth']'''
    for split, w in zip(splits, weights):
        print('#########', w, '#########')
        for s in split:
            config['mode'] = 'test'
            config['dataset']['split'] = s
            config['models']['encoder_params']['pretrained_path'] = w
            print(s, w)
            print(config)

            trainer = Trainer(config, save_folder_nets, save_folder_results, device,
                            timer=time.time())
            trainer.train()


if __name__ == '__main__':
    args = init_args()
    main(args)
