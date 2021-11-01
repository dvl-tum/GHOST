import yaml
import argparse
import torch
import logging
import warnings
import time
from src.manager import Manager
from src.reid_manager import ManagerReID
from src.Det4ReID_manager import ManagerDet4ReID

'''import sys
import traceback

class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

sys.stdout = TracePrints()'''

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
                        default='config/config_tracker.yaml',
                        #default='config/config_reid.yaml',
                        #default='config/config_Det4ReID.yaml',
                        #default='config/config_proxy.yaml',
                        help='Path to config file')

    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    import random
    #random.seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))
    logger.info(config) 
    if 'tracker' in args.config_path:
        if config['tracker']['mode'] == 'hyper_search':
            num_iter = 30
        else:
            num_iter = 1

        for i in range(num_iter):
            if config['tracker']['mode'] == 'hyper_search':
                # act_reid_thresh, inact_reid_thresh, avg_inact: num, proxy
                
                config['tracker']['act_reid_thresh'] = random.uniform(0.25, 0.8) #0.088, 0.108
                config['tracker']['inact_reid_thresh'] = random.uniform(0.25, config['tracker']['act_reid_thresh']) #0.064, 0.084
                config['tracker']['avg_inact']['num'] = random.randint(0, 100)
                config['tracker']['avg_inact']['proxy'] = random.choice(['mean', 'median']) #'mode', 'mean', 'median'
                config['tracker']['avg_act']['num'] = random.randint(0, 100)
                config['tracker']['avg_act']['proxy'] = random.choice(['mean', 'median']) #'mean', 'median'
            logger.info('Iteration {}'.format(i+1))
            logger.info(config)
            manager = Manager(device, time.time(), config['dataset'],
                            config['reid_net'], config['tracker'], train=True)
            #manager.train()
            manager._evaluate(mode='test')
    elif 'proxy.yaml' in args.config_path.split('_'):
        manager = ProxyManager(device, time.time(), config['dataset'],
                          config['reid_net'], config['tracker'], train=True)
        manager.train()
    elif 'Det4ReID.yaml' in args.config_path.split('_'):
        manager = ManagerDet4ReID(device, time.time(), config['dataset'],
                          config['reid_net'], config['tracker'], train=True)
        manager.train()
    else:
        manager = ManagerReID(device, time.time(), config['dataset'],
                      config['reid_net'], config['tracker'])
        manager.train()

if __name__ == '__main__':
    args = init_args() 
    main(args)
