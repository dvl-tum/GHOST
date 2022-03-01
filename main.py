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
            if config['tracker']['inact_thresh'] == 30:
                # mean of all
                # act = [0.86, 0.83, 0.85, 0.85, 0.83, 0.83, 0.83, 0.82, 0.84, 0.93]
                # inact = [0.86, 0.83, 0.85, 0.85, 0.83, 0.83, 0.83, 0.82, 0.84, 0.93]
                # mean of active
                act = [1.02, 1.0, 1.03, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.01]
                inact = [1.02, 1.0, 1.03, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.01]
            else:
                # mean of all
                # act = [0.97, 0.91, 0.85, 0.85, 0.83, 0.83, 0.83, 0.82, 0.84, 0.93]
                # inact = [0.97, 0.91, 0.85, 0.85, 0.83, 0.83, 0.83, 0.82, 0.84, 0.93]
                # mean of active 
                act = [1.06, 1.05, 1.06, 1.04, 1.04, 1.04, 1.04, 1.04, 1.05, 1.01]
                inact = [1.06, 1.05, 1.06, 1.04, 1.04, 1.04, 1.04, 1.04, 1.05, 1.01]
        else:
            if config['tracker']['inact_thresh'] == 30:
                # from median dist init threshs
                # act = [0.74, 0.73, 0.73, 0.72, 0.75, 0.72, 0.71, 0.74, 0.72, 0.75]
                # inact = [0.71, 0.69, 0.71, 0.69, 0.73, 0.67, 0.65, 0.70, 0.64, 0.55]

                # from median dist
                # act = [0.73, 0.73, 0.73, 0.70, 0.75, 0.72, 0.70, 0.75, 0.73, 0.74]
                # inact = [0.71, 0.70, 0.72, 0.69, 0.72, 0.68, 0.64, 0.73, 0.67, 0.55]

                # from each sample dist
                # act = [0.78, 0.78, 0.77, 0.76, 0.79, 0.76, 0.76, 0.79, 0.79, 0.79]
                # inact = [0.78, 0.77, 0.78, 0.75, 0.79, 0.77, 0.73, 0.79, 0.77, 0.65]

                # best params:
                act = [0.7, 0.725, 0.67, 0.76, 0.675, 0.72, 0.71, 0.75, 0.73, 0.75]
                inact = [0.75, 0.76, 0.73, 0.75, 0.74, 0.68, 0.65, 0.73, 0.67, 0.65]

            else:
                # from median dist
                act = [0.7, 0.725, 0.67, 0.665, 0.675, 0.725, 0.71, 0.77, 0.77, 0.76]#0.75, 0.75, 0.75]
                inact = [0.75, 0.76, 0.73, 0.72, 0.74, 0.76, 0.75, 0.75, 0.73, 0.625]#0.54, 0.54, 0.54]

                # act = [0.76, 0.73, 0.75, 0.74, 0.73, 0.73, 0.73]
                # inact = [0.72, 0.66, 0.74, 0.7, 0.69, 0.73, 0.66]

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

        train = False
        # num_iter = 30
        mot_20 = False
        tmoh = False
        qd_dets = False
        fairmot_dets = False
        get_test_set_results = False
        byte = False
        reid_ablation = False

        use_train_set = config['dataset']['half_train_set_gt']

        if mot_20:
            with open('config/config_tracker20.yaml', 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        if train:
            config['tracker']['motion_config']['ioa_threshold'] = 'learned'

        val_set = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        for det_file, a, ina, val in zip(det_files, act, inact, val_set):

            if det_file != 'FairMOT.txt':
                continue

            # config['dataset']['validation_set_gt'] = 0
            # val = 0
            # config['dataset']['splits'] = 'mot17_test'

            if use_train_set:
                det_file = det_file[:-4] + 'Train' + det_file[-4:]

            # parameter mean
            a = 0.9 #'every'
            ina = 0.9 #'every'

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
                if det_file != "tracktor_prepr_det.txt":
                    continue

            if tmoh:
                # TMOH
                det_file = "tmoh.txt"
                a = 0.76
                ina = 0.67
                val = 0
                config['dataset']['validation_set_gt'] = 0
                config['dataset']['detector'] = 'all'

            if qd_dets:
                # QDTrack plain dets
                det_file = 'qdtrack_dets.txt'
                a = 0.65
                ina = 0.65
                val = 0

            if fairmot_dets:
                # FairmOT plain Dets
                det_file = 'FairMOT_detes_Same_As_Fairmot.txt' #'FairMOT_dets_NMS0.7.txt' #'FairMOT_dets.txt'
                a = 0.75
                ina = 0.6
                # config['dataset']['splits'] = 'debug_train'
                # config['tracker']['avg_inact']['num'] = 63
                # config['tracker']['avg_inact']['proxy'] = 'median'
                # onfig['tracker']['store_dist'] = 1
                # config['tracker']['motion_config']['apply_motion_model'] = 0
                val = 0

            if byte and not mot_20:
                # det_file = "byte_val.txt"
                a = 0.75 #0.75
                ina = 0.7 #0.54
                det_file = 'all_train_byte.txt'
                #det_file = 'bytetrack_text.txt' 
                config['dataset']['validation_set_gt'] = 0
                val = 0
                #config['dataset']['splits'] = 'mot17_test'

            if mot_20 and not byte:
                det_file = "tracktor_prepr_det.txt" #'TMOH.txt'
                a = 0.7 # 0.76 #0.75 # 0.75
                ina = 0.7 # 0.75 #0.54 # 0.54
                #val = 1
                #config['dataset']['validation_set_gt'] = 1

                config['dataset']['validation_set_gt'] = 0
                val = 0
                config['dataset']['splits'] = 'mot20_test'
            
            if byte and mot_20:
                det_file = "byte_track_20.txt"
                a = 0.65 #0.75
                ina = 0.7 #0.54
                config['dataset']['validation_set_gt'] = 0
                val = 0
                config['dataset']['splits'] = 'mot20_test'
            
            if reid_ablation:
                det_file = "center_track.txt"
                a = 0.7 #0.56 # 0.75 #0.59
                ina = 0.55 #0.62 #0.75 #0.61
                val = 1
            
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

                '''config['train']['lr'] = random.choice([0.1, 0.01, 0.001, 0.0001, 0.00001])
                config['train']['wd'] = random.choice([0.001, 0.0001, 0.00001, 0.000001, 0.0000001])
                config['train']['margin'] = random.choice([0.2, 0.3, 0.4, 0.5])
                config['train']['loss'] = random.choice(['l2', 'triplet', 'l2_weighted'])
                config['train']['loss_scale'] = random.choice([1, 5, 10, 15])'''

                logger.info('Iteration {}'.format(i + 1))
                logger.info(config)
                manager = Manager(device, config['dataset'],
                                  config['reid_net'], config['tracker'], 
                                  config)
                if train:
                    manager._train(config['train'])
                manager._evaluate(mode='test', log=True)
            quit()

if __name__ == '__main__':
    args = init_args()
    main(args)
