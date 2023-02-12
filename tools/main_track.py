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
    parser.add_argument('--det_conf', type=float, default=0.6)
    parser.add_argument('--act', type=float, default=0.70000001)
    parser.add_argument('--inact', type=float, default=0.7)
    parser.add_argument('--det_file', type=str, default='qdtrack.txt')
    parser.add_argument('--only_pedestrian', type=int, default=1)
    parser.add_argument('--inact_patience', type=int, default=50)
    parser.add_argument('--combi', type=str, default='sum')
    parser.add_argument('--store_feats', type=int, default=0)
    parser.add_argument('--store_dist', type=int, default=0)
    parser.add_argument('--on_the_fly', type=int, default=0)
    parser.add_argument('--do_inact', type=int, default=1)
    parser.add_argument('--splits', type=str, default="mot20_test")
    parser.add_argument('--len_thresh', type=int, default=0)
    parser.add_argument('--new_track_conf', type=float, default=0.6)
    parser.add_argument('--remove_unconfirmed', type=float, default=0)
    parser.add_argument('--last_n_frames', type=int, default=100000000)
    return parser.parse_args()


def main_track(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))
    logger.info(config)

    config['tracker']['act_reid_thresh'] = args.act
    config['tracker']['inact_reid_thresh'] = args.inact
    config['tracker']['det_conf'] = args.det_conf
    config['dataset']['det_file'] = args.det_file
    config['dataset']['only_pedestrian'] = args.only_pedestrian
    config['dataset']['splits'] = args.splits
    config['tracker']['inact_patience'] = args.inact_patience
    config['tracker']['motion_config']['combi'] = args.combi
    config['tracker']['store_feats'] = args.store_feats
    config['tracker']['store_dist'] = args.store_dist
    config['tracker']['on_the_fly'] = args.on_the_fly
    config['tracker']['avg_inact']['do'] = args.do_inact
    config['tracker']['length_thresh'] = args.len_thresh
    config['tracker']['new_track_conf'] = args.new_track_conf
    config['tracker']['remove_unconfirmed'] = args.remove_unconfirmed
    config['tracker']['motion_config']['last_n_frames'] = args.last_n_frames
    logger.info(config)

    manager = Manager(
        device,
        config['dataset'],
        config['reid_net'],
        config['tracker'],
        config)

    manager._evaluate(
        log=True,
        first=config['tracker']['first_all'])


if __name__ == '__main__':
    args = init_args()
    main_track(args)
