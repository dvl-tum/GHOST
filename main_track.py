from email.policy import default
import yaml
import argparse
import torch
import logging
import warnings
from src.manager import Manager
import torchvision

logger = logging.getLogger('AllReIDTracker')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(message)s')
formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

warnings.filterwarnings("ignore")

logger.info(torchvision.__version__)
logger.info(torch.__version__)


def init_args():
    parser = argparse.ArgumentParser(description='AllReID tracker')
    parser.add_argument('--config_path', type=str,
                        default='config/config_tracker.yaml',
                        help='Path to config file')
    parser.add_argument('--thresh', type=float, default=0.6)
    parser.add_argument('--act', type=float, default=0.70000001)
    parser.add_argument('--inact', type=float, default=0.7)
    parser.add_argument('--det_file', type=str, default='qdtrack.txt')

    return parser.parse_args()


def main_track(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))
    logger.info(config)
    if 'tracker' in args.config_path:

        config['tracker']['act_reid_thresh'] = args.act
        config['tracker']['inact_reid_thresh'] = args.inact
        config['tracker']['thresh'] = args.thresh
        config['dataset']['det_file'] = args.det_file
        logger.info(config)
        
        manager = Manager(
            device,
            config['dataset'],
            config['reid_net'],
            config['tracker'],
            config)

        manager._evaluate(
            mode='test',
            log=True,
            first=config['tracker']['first_all'])
   

if __name__ == '__main__':
    args = init_args()
    main_track(args)
