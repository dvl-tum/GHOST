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
import src.reid_manager
#from src.reid_manager import ManagerReID
import numpy as np
import random
from torchreid import models
import torchreid
import copy

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
    print(config)

    vis_thresh =  [[np.round(v-0.1, decimals=1), np.round(v, decimals=1)] for v in np.linspace(0.1, 1.0, num=10)]
    vis_thresh[-1][-1] = 1.1
    vis_thresh = [tuple(v) for v in vis_thresh]
    size_thresh = [(0, 128), (128, 256), (256, 4000)] #[(np.round(v, decimals=1), np.round(v+25, decimals=1)) for v in np.linspace(25, 775, num=31)]
    frame_dist_thresh = [(0, 1.01/14), (1.01/14, 4/14), (4/14, 7/14), (7/14, 10/14), (10/14, 1), (1, 2), (2, 20)] #[(v, v+1) for v in np.linspace(0, 40, num=41)]
    size_diff_thresh =  [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 5.0)] #[(5.0, 2.0)] + [(np.round(v, decimals=1), np.round(v-0.1, decimals=1)) for v in np.linspace(2, -0.9, num=30)]
    gallery_vis_thresh = [[np.round(v-0.1, decimals=1), np.round(v, decimals=1)] for v in np.linspace(0.1, 1.0, num=10)]
    gallery_vis_thresh[-1][-1] = 1.1
    gallery_vis_thresh = [tuple(v) for v in gallery_vis_thresh]
    rel_gallery_vis_thresh = [(np.round(v-0.1, decimals=1), np.round(v, decimals=1)) for v in np.linspace(-0.9, 2, num=30)]
    only_next_frame = [1]
    occluder_thresh = [(np.round(v, decimals=1)-0.0001, np.round(v+1, decimals=1)-0.0001) for v in np.linspace(0, 4, num=5)]
    jaccard_thresh = [(np.round(v-0.2, decimals=1)-0.0001, np.round(v, decimals=1)-0.0001) for v in np.linspace(0.2, 1.2, num=6)]

    var_dict_init = {'vis': vis_thresh, 'size': size_thresh, 'frame_dist': frame_dist_thresh, 'size_diff': size_diff_thresh, 
                'gallery_vis': gallery_vis_thresh, 'rel_gallery_vis': rel_gallery_vis_thresh, 'only_next_frame': only_next_frame,
                'occluder': occluder_thresh, 'jaccard': jaccard_thresh}

    variables = list(var_dict_init.keys()) 

    # choose which ones to use 
    uses = [['vis'], ['size'], ['frame_dist'], ['size_diff'], ['gallery_vis', 'vis']] #['vis']
    for use in uses:
        print(use)
        var_dict = copy.deepcopy(var_dict_init)
        #use = ['gallery_vis', 'vis'] # vis, size, frame_dist, size_diff, gallery_vis
        for v in var_dict_init.keys():
            if v not in use:
                var_dict[v] = [0]

        several = len(var_dict) - list(var_dict.values()).count(0)
        several = 1

        precomp_dist, precomp_ys, precomp = None, None, None

        if several: 
            for v1 in var_dict[variables[0]]:
                for v2 in var_dict[variables[1]]: #size_diff_thresh: #frame_dist_thresh:
                    for v3 in var_dict[variables[2]]:
                        for v4 in var_dict[variables[3]]:
                            for v5 in var_dict[variables[4]]:
                                for v6 in var_dict[variables[5]]:
                                    for v7 in var_dict[variables[6]]:
                                        for v8 in var_dict[variables[7]]:
                                            for v9 in var_dict[variables[8]]:
                                                
                                                logger.info("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(v1, v2, v3, v4, v5, v6, v7, v8, v9))
                                                #print(val1, size_thresh, frame_dist_thresh, val)
                                                config['tracker']['iou_thresh'] = v1
                                                config['tracker']['size_thresh'] = v2
                                                config['tracker']['frame_dist_thresh'] = v3 #frame_dist_thresh #val
                                                config['tracker']['size_diff_thresh'] = v4 #val
                                                config['tracker']['gallery_vis_thresh'] = v5 #val
                                                config['tracker']['rel_gallery_vis_thresh'] = v6
                                                config['tracker']['only_next_frame'] = v7
                                                config['tracker']['occluder_thresh'] = v8
                                                config['tracker']['jaccard_thresh'] = v9 


                                                manager = src.reid_manager.ManagerReID(device, time.time(), config['dataset'],
                                                                    config['reid_net'], config['tracker'], experiment_name='Split_2')
                                                if precomp is not None:
                                                    manager.ys = precomp_ys
                                                    manager.dist = precomp_dist
                                                    manager.computed = precomp

                                                manager._evaluate()

                                                if precomp is None:
                                                    precomp_ys = manager.ys
                                                    precomp_dist = manager.dist
                                                    precomp = manager.computed

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
    pass
    #encoder = models.build_model(name='resnet50', num_classes=1000)
    #torchreid.utils.load_pretrained_weights(encoder, 'resnet50_market_xent.pth.tar') 

    

if __name__ == '__main__':
    args = init_args()
    main(args)
