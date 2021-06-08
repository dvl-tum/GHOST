from numpy.core.function_base import linspace
from pandas.core import frame
import yaml
import argparse
import torch
import logging
import warnings
import time
import os.path as osp
import os
from src.reid_manager import ManagerReID
import numpy as np
import random
from torchreid import models
import torchreid

logger = logging.getLogger('AllReIDTracker')
logger.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s - %(message)s')
formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

warnings.filterwarnings("ignore")


def init_args():
    parser = argparse.ArgumentParser(description='AllReID tracker')
    parser.add_argument('--config_path', type=str,
                        default='config/config_tracker_analysis_R50Xent.yaml',
                        help='Path to config file')

    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))

    vis_thresh =  [(v-0.1, v) for v in np.linspace(0.9, 1.0, num=2)] #[(v-0.1, v) for v in np.linspace(0.1, 1, num=10)]
    size_thresh = [(v, v+25) for v in np.linspace(25, 775, num=31)]
    frame_dist_thresh = [(v, v+1) for v in np.linspace(0, 40, num=41)]
    size_diff_thresh = [(v, v-0.1) for v in np.linspace(3, 0.1, num=30)]
    
    var_dict = {'vis': vis_thresh, 'size': size_thresh, 'frame_dist': frame_dist_thresh, 'size_diff': size_diff_thresh}
    variables = list(var_dict.keys())

    # choose which ones to use
    use = ['vis', 'size_diff']
    for v in var_dict.keys():
        if v not in use:
            var_dict[v] = [0]
    
    several = len(var_dict) - list(var_dict.values()).count(0)
    several = 1

    if several:
        for v1 in var_dict[variables[0]]:
            for v2 in var_dict[variables[1]]: #size_diff_thresh: #frame_dist_thresh:
                for v3 in var_dict[variables[2]]:
                    for v4 in var_dict[variables[3]]:
                        print("{}, {}, {}, {}".format(v1, v2, v3, v4))
                        #print(val1, size_thresh, frame_dist_thresh, val)
                        config['tracker']['iou_thresh'] = v1
                        config['tracker']['size_thresh'] = v2
                        config['tracker']['frame_dist_thresh'] = v3 #frame_dist_thresh #val
                        config['tracker']['size_diff_thresh'] = v4 #val

                        manager = ManagerReID(device, time.time(), config['dataset'],
                                            config['reid_net'], config['tracker'])
                        manager._evaluate()
    else:
        for val in size_diff_thresh: #size_diff_thresh: #frame_dist_thresh:
            print("{}, {}, {}, {}".format(vis_thresh, size_thresh, frame_dist_thresh, val))
            #print(val1, size_thresh, frame_dist_thresh, val)
            config['tracker']['iou_thresh'] = vis_thresh
            config['tracker']['size_thresh'] = size_thresh
            config['tracker']['frame_dist_thresh'] = frame_dist_thresh #val
            config['tracker']['size_diff_thresh'] = val #size_diff_thresh #val

            manager = ManagerReID(device, time.time(), config['dataset'],
                                config['reid_net'], config['tracker'])
            manager._evaluate()
    quit()


    '''
    Analysis over the following four parameters:
        oracle_iou: 0, iou_thresh: 0.7  --> Occlusion
        oracle_size: 0, size_thresh: 1400   --> BB size
        oracle_frame_dist: 0, frame_dist_thresh: 10 --> duration of occlusion
        oracle_size_diff: 0, size_diff_thresh: 50   --> size change
    '''
    visibility_threshs = np.linspace(0.1, 1, num=10)
    size_threshs = np.concatenate((np.linspace(50, 275, num=10), np.linspace(300, 1200, num=10)))
    frame_dist_threshs = np.concatenate((np.linspace(1, 10, num=10), np.linspace(20, 200, num=10)))
    size_diff_threshs = np.linspace(0, 100, num=11)
    
    # which oracles to use
    oracles = [1, 1, 1, 1]

    # combinations    
    all_poss = [(i, j, k, m) for i in visibility_threshs for j in size_threshs for k in frame_dist_threshs for m in size_diff_threshs]
    random.shuffle(all_poss)
    selection = random.sample(all_poss, k=10)
    
    # visibility
    #selection = [(i, 0, 0, 0) for i in visibility_threshs]
    
    # size
    #selection = [(0, j, 0, 0) for j in size_threshs]
    
    # frame dist
    #selection = [(0, 0, k, 0) for k in frame_dist_threshs]
    
    # size diff
    #selection = [(0, 0, 0, m) for m in size_diff_threshs]
    logger.info('Using the following selection {}'.format(selection))
    
    for s in selection:
        '''
        config['tracker']['oracle_iou'] = oracles[0]
        config['tracker']['iou_thresh'] = s[0]
        config['tracker']['oracle_size'] = oracles[1]
        config['tracker']['size_thresh'] = s[1]
        config['tracker']['oracle_frame_dist'] = oracles[2]
        config['tracker']['frame_dist_thresh'] = s[2]
        config['tracker']['oracle_size_diff'] = oracles[3]
        config['tracker']['size_diff_thresh'] = s[3]
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('Switching to device {}'.format(device))
        logger.info(config) 
        if 'tracker' in args.config_path:
            manager = Manager(device, time.time(), config['dataset'],
                            config['reid_net'], config['tracker'], train=True)
            #manager.train()
            manager._evaluate(mode='test')
        elif 'proxy.yaml' in args.config_path.split('_'):
            manager = ProxyManager(device, time.time(), config['dataset'],
                            config['reid_net'], config['tracker'], train=True)
            manager.train()
        else:
            manager = ManagerReID(device, time.time(), config['dataset'],
                        config['reid_net'], config['tracker'])
            manager.train()


def analysis():
    encoder = models.build_model(name='resnet50', num_classes=1000)
    torchreid.utils.load_pretrained_weights(encoder, 'resnet50_market_xent.pth.tar') 

    

if __name__ == '__main__':
    args = init_args()
    main(args)
