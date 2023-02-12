from ReID import net
import os.path as osp
import os
from data.splits import _SPLITS
from .tracker import Tracker
from src.datasets.TrackingDataset import TrackingDataset
import logging
import torchreid
from src.eval_track_eval import evaluate_track_eval
from src.eval_track_eval_bdd import evaluate_track_eval_bdd
import pandas as pd
import json
import sys
sys.path.append('/usr/wiss/seidensc/Documents/fast-reid')
from tools.train_net import get_model
import numpy as np


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

col_names = [
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

col_names_short = [
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

    def _evaluate(self, first=False, log=True):
        names = list()

        # get tracking files
        i = 0

        for j, seq in enumerate(self.loaders):
            # first = feed sequence data through backbon and update statistics
            # before tracking

            logger.info(f"Sequence {j}/{len(self.loaders)}")
            if first:
                self.reset_for_first(seq, i)
                i += 1

            self.tracker.encoder = self.encoder
            self.tracker.track(seq[0], log=log)

            names.append(seq[0].name)

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
        self.tracker = Tracker(self.tracker_cfg, self.encoder,
                               net_type=self.net_type,
                               output=self.reid_net_cfg['output'],
                               data=self.dataset_cfg['det_file'],
                               device=self.device)

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
            encoder, self.sz_embed, _ = net.load_net(
                self.reid_net_cfg['trained_on']['num_classes'],
                **self.reid_net_cfg['encoder_params'])

        self.encoder = encoder.to(self.device)

    def _get_loaders(self, dataset_cfg):
        # get loaders for test / val mode of current split
        seqs = _SPLITS[dataset_cfg['splits']]['test']['seq']
        self.dir = _SPLITS[dataset_cfg['splits']]['test']['dir']

        # tracking dataset
        dataset = TrackingDataset(
            dataset_cfg['splits'],
            seqs,
            dataset_cfg,
            self.dir,
            net_type=self.reid_net_cfg['encoder_params']['net_type'],
            dev=self.device,
            assign_gt=self.tracker_cfg['store_feats'])
        # only assign gt if features should be stored

        return dataset

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

    def eval_track_eval(self, log=True, dir='val'):
        output_res, output_msg = evaluate_track_eval(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            gt_path=osp.join(self.dataset_cfg['gt_dir'], self.dir),
            log=log
        )

        return output_res, output_msg

    def eval_track_eval_dance(self, log=True, dir='val'):
        output_res, output_msg = evaluate_track_eval(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            gt_path=osp.join(self.dataset_cfg['gt_dir'], self.dir),
            log=log
        )

        return output_res, output_msg

    def eval_track_eval_bdd(self, log=True):
        '''self.MOT2BDD()
        output_res, output_msg = evaluate_track_eval_bdd(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            log=log
        )'''
        self.MOT2BDDTest()

        return output_res, output_msg

    def MOT2BDD(self):
        files = os.listdir(os.path.join('out', self.tracker.experiment))
        os.makedirs(os.path.join(
            'out', self.tracker.experiment + '_orig'), exist_ok=True)
        for seq in files:
            if seq[-4:] == 'json':
                continue

            seq_df = pd.read_csv(os.path.join(
                'out', self.tracker.experiment, seq), names=col_names, index_col=False)

            assert len(os.listdir('/storage/slurm/seidensc/datasets/BDD100/bdd100k/images/track/val/' + seq)
                       ) >= seq_df['frame'].unique().shape[0], seq

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

            os.rename(os.path.join('out', self.tracker.experiment, seq),
                      os.path.join('out', self.tracker.experiment + '_orig', seq))

    def MOT2BDDTest(self):

        out_orig = os.path.join('out', self.tracker.experiment + '_orig')
        out_subm = os.path.join('bdd_for_submission', self.tracker.experiment)
        os.makedirs(out_subm, exist_ok=True)
        image_dir = osp.join(
            self.dataset_cfg['mot_dir'],
            'images',
            'track',
            self.dir)

        col_names = [
            'frame', 'id', 'bb_left', 'bb_top', 'bb_width',
            'bb_height', 'conf', '?', 'label', 'vis']

        BDD_NAME_MAPPING = {
            1: "pedestrian", 2: "rider", 3: "car", 4: "truck",
            5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"}

        count = 0
        for seq in os.listdir(out_orig):
            if 'json' in seq:
                continue
            
            if seq in os.listdir(image_dir):
                count += 1
                df = pd.read_csv(os.path.join(out_orig, seq),
                                 names=col_names, index_col=False)

                df['label'] = df['label'].values + np.ones(df.shape[0])

                final_out = df.sort_values(by=['frame', 'id'])
                sequence_name = seq
                output_file_path = os.path.join(out_subm, seq + '.json')

                det_list = list()
                # Find the max frame
                df = df.reset_index()
                max_frame = int(
                    sorted(os.listdir(image_dir + '/' + seq))[-1][:-4][-4:])
                for frame in range(1, max_frame+1):
                    frame_dict = dict()
                    frame_df = final_out[final_out['frame'] == frame]
                    frame_dict['name'] = sequence_name + '/' + \
                        sequence_name + "-" + f"{frame:07d}.jpg"
                    frame_dict['index'] = int(frame - 1)
                    labels_list = list()
                    for idx, row in frame_df.iterrows():
                        labels_dict = dict()
                        labels_dict['id'] = row['id']
                        labels_dict['score'] = row['conf']
                        labels_dict['category'] = BDD_NAME_MAPPING[int(
                            row['label'])]
                        labels_dict['box2d'] = {
                            'x1': row['bb_left'],
                            'x2': row['bb_left'] + row['bb_width'],
                            'y1': row['bb_top'],
                            'y2': row['bb_top'] + row['bb_height']
                        }
                        labels_list.append(labels_dict)

                    frame_dict['labels'] = labels_list
                    det_list.append(frame_dict)

                with open(output_file_path, 'w') as f:
                    json.dump(det_list, f)
