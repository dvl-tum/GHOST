import yaml
import argparse
import torch
import logging
import warnings
from src.manager import Manager
import torchvision


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
# formatter = logging.Formatter('%(asctime)s - %(message)s')
formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

warnings.filterwarnings("ignore")

logger.info(torchvision.__version__)
logger.info(torch.__version__)

while logger.handlers:
    logger.handlers.pop()


def init_args():
    parser = argparse.ArgumentParser(description='AllReID tracker')
    parser.add_argument('--config_path', type=str,
                        default='config/config_tracker.yaml',
                        help='Path to config file')

    return parser.parse_args()


def main(args):

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    import random

    # random.seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))
    logger.info(config)
    if 'tracker' in args.config_path:
        if config['tracker']['mode'] == 'hyper_search':
            num_iter = 30
        else:
            num_iter = 1

        # for basic reid
        if config['reid_net']['encoder_params']['net_type'] == 'resnet50_analysis':
            # reid patience 30
            if config['tracker']['inact_thresh'] == 30:
                # mean of active
                act = [10.43, 10.38, 10.8, 10.48, 10.25, 10.35, 10.28, 10.27, 7.79, 12.22]
                inact = [10.43, 10.38, 10.8, 10.48, 10.25, 10.35, 10.28, 10.27, 7.79, 12.22]
                # mean of all samples
                # act = [10.02, 10.25, 10.17, 9.77, 10.03, 10.17, 10.07, 10.29, 10.59, 10.98]
                # inact = [10.02, 10.25, 10.17, 9.77, 10.03, 10.17, 10.07, 10.29, 10.59, 10.98]
            # reid patience inf
            else:
                # mean of active
                act = [10.2, 10.3, 10.6, 10.5, 10.9, 10.2, 10.4, 10.3, 10.7, 10.1]
                inact = [10.2, 10.3, 10.6, 10.5, 10.9, 10.2, 10.4, 10.3, 10.7, 10.1]
                # mean of all samples
                # act = [8.60, 8.61, 8.65,  8.43, 8.37, 8.70, 8.59, 8.77, 9.55, 11.2]
                # act = [8.60, 8.61, 8.65,  8.43, 8.37, 8.70, 8.59, 8.77, 9.55, 11.2]
        elif args.config_path == 'config/config_tracker_basicreid.yaml':
            act = [0.7, 0.725, 0.67, 0.665, 0.675, 0.725, 0.71, 0.77, 0.77, 0.76]
            inact = [0.7, 0.725, 0.67, 0.665, 0.675, 0.725, 0.71, 0.77, 0.77, 0.76]
        else:
            # from median dist
            act = [0.7, 0.725, 0.67, 0.665, 0.675, 0.725, 0.71, 0.77, 0.77, 0.76]#0.75, 0.75, 0.75]
            inact = [0.75, 0.76, 0.73, 0.72, 0.74, 0.76, 0.75, 0.75, 0.73, 0.625]#0.54, 0.54, 0.54]

            # from each sample dist
            # act = [0.78, 0.8, 0.76, 0.79, 0.8, 0.77, 0.79, 0.78, 0.78, 0.76]
            # inact = [0.625, 0.785, 0.695, 0.75, 0.785, 0.74, 0.77, 0.745, 0.755, 0.695]

        det_files = [
            "qdtrack.txt",
            "CenterTrack.txt",
            "CSTrack.txt",
            "FairMOT.txt",
            "JDE.txt",
            "TraDeS.txt",
            "TransTrack.txt",
            "CenterTrackPub.txt",
            "center_track.txt",
            "tracktor_prepr_det.txt"]

        val_set = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        for det_file, a, ina, val in zip(det_files, act, inact, val_set):
            '''if det_file != "FairMOT.txt":
                continue'''

            config['dataset']['det_file'] = det_file
            config['dataset']['validation_set'] = val
            config['tracker']['act_reid_thresh'] = a
            config['tracker']['inact_reid_thresh'] = ina
            for i in range(num_iter):

                if config['tracker']['mode'] == 'hyper_search':
                    # act_reid_thresh, inact_reid_thresh, avg_inact: num, proxy

                    config['tracker']['act_reid_thresh'] = random.uniform(
                        0.3, 1.0)
                    config['tracker']['inact_reid_thresh'] = random.uniform(
                        0.3, config['tracker']['act_reid_thresh'])
                    # config['tracker']['avg_inact']['num'] = random.randint(0, 100)
                    # config['tracker']['avg_inact']['proxy'] = random.choice(['mode', 'mean', 'median'])
                    # config['tracker']['avg_act']['num'] = random.randint(0, 100)
                    # config['tracker']['avg_act']['proxy'] = random.choice(['mode', 'mean', 'median'])

                logger.info('Iteration {}'.format(i + 1))
                logger.info(config)
                print(config)
                manager = Manager(device, config['dataset'],
                                  config['reid_net'], config['tracker'])
                # manager.train()
                manager._evaluate(mode='test')
        

if __name__ == '__main__':
    args = init_args()
    main(args)
