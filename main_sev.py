from email.policy import default
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
    import random

    # random.seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))
    logger.info(config)
    if 'tracker' in args.config_path:

        # from median dist
        act = [0.7, 0.725, 0.67, 0.665, 0.675, 0.725, 0.71, 0.77, 0.77, 0.76, 0.77]
        inact = [0.75, 0.76, 0.73, 0.72, 0.74, 0.76, 0.75, 0.75, 0.73, 0.625, 0.76]

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
            "tracktor_prepr_det.txt",
            'ctracker.txt']

        train = False
        mot_20 = False
        get_test_set_results = False
        byte = False
        reid_ablation = False
        bdd = False
        thresh = args.thresh
        use_train_set = config['dataset']['half_train_set_gt']

        if mot_20:
            with open('config/config_tracker20.yaml', 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        elif bdd:
            with open('config/config_tracker_bdd.yaml', 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        if train:
            config['tracker']['motion_config']['ioa_threshold'] = 'learned'

        val_set = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]

        for i, (det_file, a, ina, val) in enumerate(zip(det_files, act, inact, val_set)):

            # for a in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            a = 0.7
            ina = 0.65

            if det_file != "center_track.txt":
                continue
                
            if bdd:
                det_file = args.det_file #"bdd100k.txt"
                if det_file == 'qdtrack.txt':
                    # config['dataset']['det_dir'] = 'out/qdtrack_orig2_bdd/'
                    # det_file = 'bdd100k.txt'
                    thresh = -10
                val = 0
                a = args.act
                ina = args.inact

            if use_train_set:
                det_file = det_file[:-4] + 'Train' + det_file[-4:]

            if train:
                if det_file != "tracktor_prepr_det.txt":
                    continue

            if get_test_set_results:
                val = 0
                config['dataset']['validation_set_gt'] = 0
                config['dataset']['splits'] = 'mot17_test' #'mot17_train_test'
                if not mot_20:
                    config['dataset']['detector'] = 'all'
                if mot_20:
                    config['dataset']['splits'] = 'mot20_test' #'mot20_train_test'

            if byte and not mot_20:
                det_file = "byte_val.txt"
                # det_file = 'all_train_byte.txt'
                a = 0.75 #0.75
                ina = 0.7 #0.54
                config['dataset']['validation_set_gt'] = 1
                val = 1
                if get_test_set_results:
                    det_file = 'bytetrack_text.txt'
                    config['dataset']['splits'] = 'mot17_test'

            if mot_20 and not byte:
                det_file = "tracktor_prepr_det.txt"
                a = 0.7 # 0.76 #0.75 # 0.75
                ina = 0.7 # 0.75 #0.54 # 0.54
                config['dataset']['validation_set_gt'] = 0
                val = 0

            if byte and mot_20:
                det_file = "bytetrack_train_MOT20.txt" #"byte_track_20.txt"
                a = 0.65 #0.75
                ina = 0.7 #0.54
                config['dataset']['validation_set_gt'] = 0
                val = 0
                if get_test_set_results:
                    det_file = "bytetrack_text_MOT20.txt" #"byte_track_20.txt"

            if reid_ablation:
                det_file = "center_track.txt"
                a = 0.7 #0.56 # 0.75 #0.59
                ina = 0.55 #0.62 #0.75 #0.61
                val = 1

            config['dataset']['det_file'] = det_file
            config['dataset']['validation_set'] = val
            config['tracker']['act_reid_thresh'] = a
            config['tracker']['inact_reid_thresh'] = ina

            logger.info('Iteration {}'.format(i + 1))
            logger.info(config)

            config['tracker']['thresh'] = thresh
            manager = Manager(device, config['dataset'],
                                config['reid_net'], config['tracker'], 
                                config)
            if train:
                manager._train(config['train'])

            manager._evaluate(mode='test', log=True)
            

if __name__ == '__main__':
    args = init_args()
    main_track(args)
