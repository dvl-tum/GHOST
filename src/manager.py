from ReID import net
import os.path as osp
import os
from data.splits import _SPLITS
from .tracker import Tracker
from src.datasets.TrackingDataset import TrackingDataset
import logging
from collections import OrderedDict
import torchreid
from src.eval_track_eval import evaluate_track_eval
from src.eval_track_eval_bdd import evaluate_track_eval_bdd
import pandas as pd
import json
import sys
sys.path.append('/usr/wiss/seidensc/Documents/fast-reid')
from tools.train_net import get_model


classes = [
    'pedestrian',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'traffic light',
    'traffic sign',
    'other vehicle',
    'trailer',
    'other person']

classes_for_eval = {
    'pedestrian': 1,
    'rider': 2,
    # 'other person': 3,
    'car': 4,
    'bus': 5,
    'truck': 6,
    'train': 7,
    # 'trailer': 8,
    # 'other vehicle': 9,
    'motorcycle': 10,
    'bicycle': 11}

col_names=[
        'frame',
        'id',
        'bb_left',
        'bb_top',
        'bb_width',
        'bb_height',
        'conf',
        '?',
        'label',
        'vis']

col_names_short =[
        'frame',
        'id',
        'bb_left',
        'bb_top',
        'bb_width',
        'bb_height',
        'conf',
        # '?',
        'label',
        'vis']

frames = {'JDE': [299, 524, 418, 262, 326, 449, 368],
          'CSTrack': [298, 523, 417, 261, 325, 448, 368],
          'TraDeS': [299, 524, 418, 262, 326, 449, 374],
          'CenterTrack': [299, 524, 418, 262, 326, 449, 374],
          'CenterTrackPub': [299, 524, 418, 262, 326, 449, 374],
          'qdtrack': [299, 524, 418, 262, 326, 449, 374],
          'FairMOT': [300, 525, 419, 263, 327, 450, 369],
          'TransTrack': [299, 524, 418, 262, 326, 449, 374],
          'center_track': [300, 525, 419, 263, 327, 450, 375],
          'tracktor_prepr_det': [300, 525, 419, 263, 327, 450, 375]}


logger = logging.getLogger('AllReIDTracker.Manager')
# logger.propagate = False


class Manager():
    def __init__(self, device, dataset_cfg, reid_net_cfg, tracker_cfg, cfg):
        self.device = device
        self.reid_net_cfg = reid_net_cfg
        self.dataset_cfg = dataset_cfg
        self.tracker_cfg = tracker_cfg
        self.cfg = cfg

        # load ReID net
        self.loaders = self._get_loaders(dataset_cfg)
        self._get_models()

    def _train(self):
        raise NotImplementedError

    def _evaluate(self, mode='val', first=False, log=True):
        names = list()
        corresponding_gt = OrderedDict()

        # get tracking files
        i = 0
        
        for j, seq in enumerate(self.loaders[mode]):
            # first = feed sequence data through backbon and update statistics
            # before tracking
            
            logger.info(f"Sequence {j}/{len(self.loaders[mode])}")
            if first:
                self.reset_for_first(seq, i)
                i += 1

            # get gt bbs corresponding to detections for oracle evaluations
            if 'bdd' not in self.dataset_cfg['splits'] and 'test' not in \
                    self.dataset_cfg['splits'] and 'dance' not in self.dataset_cfg['splits']:
                self.get_corresponding_gt(seq, corresponding_gt)

            self.tracker.encoder = self.encoder
            self.tracker.track(seq[0], log=log)
            
            names.append(seq[0].name)

        # manually set experiment if already generated bbs
        # self.tracker.experiment = 'byte_dets_0.851840_evalBB:0_each_sample2:0.85:last_frame:0.77MM:1sum0.30.30.3InactPat:50ConfThresh:-0.6'
        # self.tracker.experiment = 'qdbdd' #'bytebdd'
        # self.tracker.experiment = 'dets_bdd_byte_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.9MM:1sum_0.40.30.30.3InactPat:50ConfThresh:0.35'
        # self.tracker.experiment = 'converted_byte_original'
        # self.tracker.experiment = 'converted_with_our_reid'
        # self.tracker.experiment = 'converted_with_our_reid_thresh'


        if log:
            logger.info(self.tracker.experiment)

        mota, idf1 = 0, 0

        # EVALUATION TRACKEVAL
        if 'bdd' in self.dataset_cfg['mot_dir']:
            _, _ = self.eval_track_eval_bdd(log)
        elif 'Dance' in self.dataset_cfg['mot_dir']:
            _, _ = self.eval_track_eval_dance(log)
        else:
            _, _ = self.eval_track_eval(log)
        
        return mota, idf1

    def _get_models(self):
        self._get_encoder()
        weight = self.reid_net_cfg['encoder_params']['pretrained_path'].split(
            '/')[-1][:8] if self.reid_net_cfg['encoder_params']['net_type'][
                :8] == 'resnet50' and self.reid_net_cfg['encoder_params'][
                    'net_type'][:8] != 'resnet50_analysis' else ''
        self.tracker = Tracker(self.tracker_cfg, self.encoder,
                               net_type=self.net_type,
                               output=self.reid_net_cfg['output'],
                               weight=weight,
                               data=self.dataset_cfg['det_file'],
                               device=self.device,
                               train_cfg=self.cfg['train'])

    def _get_encoder(self):
        self.net_type = self.reid_net_cfg['encoder_params']['net_type']
        if self.net_type == 'resnet50_analysis':
            # get pretrained resnet 50 from torchreid
            encoder = torchreid.models.build_model(
                name='resnet50', num_classes=1000)
            torchreid.utils.load_pretrained_weights(
                encoder, 'resnet50_market_xent.pth.tar')
            self.sz_embed = None
        elif self.net_type == "IBN":
            encoder = get_model()
        else:
            # own trained network
            encoder, self.sz_embed = net.load_net(
                self.reid_net_cfg['trained_on']['num_classes'],
                **self.reid_net_cfg['encoder_params'])

        self.encoder = encoder.to(self.device)

    def _get_loaders(self, dataset_cfg):
        # Initialize datasets
        loaders = dict()
        # get loaders for test / val mode of current split
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            self.dir = _SPLITS[dataset_cfg['splits']][mode]['dir']

            # tracking dataset
            if mode != 'train':
                dataset = TrackingDataset(
                    dataset_cfg['splits'],
                    seqs,
                    dataset_cfg,
                    self.dir,
                    net_type=self.reid_net_cfg['encoder_params']['net_type'],
                    dev=self.device)
                loaders[mode] = dataset
            elif mode == 'train':
                print('currently no train mode implemented :)')
                pass

        return loaders

    def reset_for_first(self, seq, i):
        experiment = self.tracker.experiment
        if self.tracker_cfg['store_dist']:
            distance = self.tracker.distance_
        self._get_models()
        if i == 0:
            experiment = 'first_' + experiment
        self.tracker.experiment = experiment
        logger.info('Run first for adaption')
        self.tracker.encoder = self.encoder
        if self.tracker_cfg['store_dist']:
            self.tracker.distance_ = distance
        self.encoder.train()
        self.tracker.track(seq[0], first=True)
        self.encoder.eval()

    def get_corresponding_gt(self, seq, corresponding_gt):
        df = seq[0].corresponding_gt
        df = df.drop(['bb_bot', 'bb_right'], axis=1)
        df = df.rename(
            columns={
                "frame": "FrameId",
                "id": "Id",
                "bb_left": "X",
                "bb_top": "Y",
                "bb_width": "Width",
                "bb_height": "Height",
                "conf": "Confidence",
                "label": "ClassId",
                "vis": "Visibility"})
        df = df.set_index(['FrameId', 'Id'])
        corresponding_gt[seq[0].name] = df

    def eval_track_eval(self, log=True, dir='val'):
        output_res, output_msg = evaluate_track_eval(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            gt_path=osp.join(self.dataset_cfg['mot_dir'], self.dir),
            log=log
        )
        hota = sum(output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["HOTA"]['HOTA'])/len(output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["HOTA"]['HOTA'])
        idf1 = output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["Identity"]['IDF1']
        mota = output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["CLEAR"]['MOTA']

        from csv import writer
 
        with open(self.dataset_cfg['splits'] + '.txt', 'a') as f:
            line = [self.tracker_cfg['avg_inact']['proxy'], self.tracker_cfg['avg_inact']['num']] if self.tracker_cfg['avg_inact']['do'] else ['last', 0]
            line += [self.tracker_cfg['act_reid_thresh']]
            line += [self.tracker_cfg['inact_reid_thresh']]
            line += [self.tracker_cfg['motion_config']['ioa_threshold']]
            line += [self.tracker_cfg['eval_bb']]
            line += [hota, mota, idf1]

            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f)
        
            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(line)
        
            # Close the file object
            f.close()

        return output_res, output_msg

    def eval_track_eval_dance(self, log=True, dir='val'):
        output_res, output_msg = evaluate_track_eval(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            gt_path='/storage/user/seidensc/datasets/DanceTrack/val',
            log=log
        )
        
        hota = sum(output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["HOTA"]['HOTA'])/len(output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["HOTA"]['HOTA'])
        idf1 = output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["Identity"]['IDF1']
        mota = output_res['MotChallenge2DBox'][self.tracker.experiment]['COMBINED_SEQ']['pedestrian']["CLEAR"]['MOTA']

        from csv import writer
 
        with open(self.dataset_cfg['splits'] + '.txt', 'a') as f:
            line = [self.tracker_cfg['avg_inact']['proxy'], self.tracker_cfg['avg_inact']['num']] if self.tracker_cfg['avg_inact']['do'] else ['last', 0]
            line += [self.tracker_cfg['act_reid_thresh']]
            line += [self.tracker_cfg['inact_reid_thresh']]
            line += [self.tracker_cfg['motion_config']['ioa_threshold']]
            line += [self.tracker_cfg['eval_bb']]
            line += [hota, mota, idf1]

            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f)
        
            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(line)
        
            # Close the file object
            f.close()

        return output_res, output_msg

    def eval_track_eval_bdd(self, log=True):
        self.MOT2BDD()
        output_res, output_msg = evaluate_track_eval_bdd(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            log=log
        )

        return output_res, output_msg

    def MOT2BDD(self, oracle_files=False):
        files = os.listdir(os.path.join('out', self.tracker.experiment))
        os.makedirs(os.path.join('out', self.tracker.experiment + '_orig'), exist_ok=True)
        for seq in files:
            print(seq)
            if seq[-4:] == 'json':
                continue
            if oracle_files:
                if 'qdtrack' not in self.tracker.experiment:
                    seq_df = pd.read_csv(os.path.join('out', self.tracker.experiment, seq, 'bdd100k.txt'), names=col_names_short, index_col=False)
                    def make_frame(i):
                        return int(i.split('-')[-1])
                    seq_df['frame'] = seq_df['frame'].apply(make_frame)
                else:
                    seq_df = pd.read_csv(os.path.join('out', self.tracker.experiment, seq), names=col_names, index_col=False)
            elif self.tracker.experiment == 'bytebdd':
                seq_df = pd.read_csv(os.path.join('out', self.tracker.experiment, seq), names=col_names_short, index_col=False)
            else:
                seq_df = pd.read_csv(os.path.join('out', self.tracker.experiment, seq), names=col_names, index_col=False)
            
            assert len(os.listdir('/storage/slurm/seidensc/datasets/BDD100/bdd100k/images/track/val/' + seq)) >= seq_df['frame'].unique().shape[0], seq

            if 'qdtrack' not in self.tracker.experiment and oracle_files:
                seq_df = seq_df[seq_df['conf'] > 0.4]

            det_list = list()
            for frame in seq_df['frame'].unique():
                frame_dict = dict()
                frame_df = seq_df[seq_df['frame'] == frame]
                frame_dict['name'] = seq + "-" + f"{frame:07d}.jpg"
                labels_list = list()

                for idx, row in frame_df.iterrows():
                    labels_dict = dict()
                    labels_dict['id'] = row['id']
                    labels_dict['category'] = classes[int(row['label'])]

                    if labels_dict['category'] not in classes_for_eval.keys():
                        continue
                    labels_dict['box2d'] = {
                        'x1': row['bb_left'],
                        'y1': row['bb_top'],
                        'x2': row['bb_left'] + row['bb_width'],
                        'y2': row['bb_top'] + row['bb_height']
                    }
                    labels_list.append(labels_dict)
                frame_dict['labels'] = labels_list
                det_list.append(frame_dict)

            with open(os.path.join('out', self.tracker.experiment, seq + '.json'), 'w') as f:
                json.dump(det_list, f)

            if oracle_files:
                import shutil
                os.rename(os.path.join('out', self.tracker.experiment, seq), os.path.join('out', self.tracker.experiment + '_orig', seq))
            else:
                os.rename(os.path.join('out', self.tracker.experiment, seq), os.path.join('out', self.tracker.experiment + '_orig', seq))
            


