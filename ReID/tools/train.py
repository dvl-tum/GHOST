import yaml
import argparse
import torch
import warnings
import time
import utils.utils as utils
from utils.trainer import Trainer 

warnings.filterwarnings("ignore")


def init_args():
    parser = argparse.ArgumentParser(description='Person Re-ID with for MOT')
    parser.add_argument('--config_path', type=str, default='config/config_market.yaml', help='Path to config file')

    return parser.parse_args() 


def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_folder_results = 'search_results'
    utils.make_dir(save_folder_results)
    save_folder_nets = 'search_results_net'
    utils.make_dir(save_folder_nets)

    trainer = Trainer(config, save_folder_nets, save_folder_results, device,
        timer=time.time())
    trainer.train()


if __name__ == '__main__':
    args = init_args()
    main(args)
