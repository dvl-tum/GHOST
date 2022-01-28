import ReID
from ReID import net
import os.path as osp

import os
from data.splits import _SPLITS
from .tracker import Tracker
from src.datasets.TrackingDataset import TrackingDataset
import logging
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
from torchreid import models
import torchreid
from src.eval_fairmot import Evaluator
from src.eval_mpn_track import get_results, get_summary
from src.eval_track_eval import evaluate_track_eval


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
    def __init__(self, device, dataset_cfg, reid_net_cfg, tracker_cfg):
        self.device = device
        logger.info(reid_net_cfg)
        logger.info(dataset_cfg)
        logger.info(tracker_cfg)
        self.reid_net_cfg = reid_net_cfg
        self.dataset_cfg = dataset_cfg
        self.tracker_cfg = tracker_cfg

        logger.info(reid_net_cfg)
        logger.info(dataset_cfg)
        logger.info(tracker_cfg)

        # load ReID net
        self.loaders = self._get_loaders(dataset_cfg)
        self._get_models()

    def _evaluate(self, mode='val', first=False):
        names = list()
        corresponding_gt = OrderedDict()

        # get tracking files
        for seq in self.loaders[mode]:

            # first = feed sequence data through backbon and update statistics
            # before tracking
            if first:
                self.reset_for_first(seq)

            # get gt bbs corresponding to detections for oracle evaluations
            if self.dataset_cfg['splits'] != 'mot17_test':
                self.get_corresponding_gt(seq, corresponding_gt)

            self.tracker.encoder = self.encoder
            self.tracker.track(seq[0])
            names.append(seq[0].name)

        # print interaction and occlusion stats
        self.print_interaction_occlusion_stats()

        # manually set experiment if already generated bbs
        # self.tracker.experiment = osp.join(
        #     'OtherTrackersOrig',
        #     self.dataset_cfg['det_file'][:-4])
        logger.info(self.tracker.experiment)

        # EVALUATION FAIRMOT
        self.eval_fair_mot(names)

        # EVALUATION FROM MPNTRACK
        self.eval_mpn_track(names, corresponding_gt)

        # EVALUATION TRACKEVAL
        self.eval_track_eval()

        return None, None

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
                               data=self.dataset_cfg['det_file'])

    def _get_encoder(self):
        self.net_type = self.reid_net_cfg['encoder_params']['net_type']
        if self.net_type == 'resnet50_analysis':
            # get pretrained resnet 50 from torchreid
            encoder = torchreid.models.build_model(
                name='resnet50', num_classes=1000)
            torchreid.utils.load_pretrained_weights(
                encoder, 'resnet50_market_xent.pth.tar')
            self.sz_embed = None

        else:
            # own trained network
            encoder, self.sz_embed = net.load_net(
                self.reid_net_cfg['trained_on']['name'],
                self.reid_net_cfg['trained_on']['num_classes'],
                'test', attention=False,
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
                    dev=self.device)
                loaders[mode] = dataset

        return loaders

    def reset_for_first(self, seq):
        experiment = self.tracker.experiment
        self._get_models()
        self.tracker.experiment = experiment
        self.tracker.encoder = self.encoder
        self.tracker.track(seq[0], first=True)

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

    def print_interaction_occlusion_stats(self):
        for what, dd in zip(['Interaction', 'Occlusion'], [
                            self.tracker.interaction, self.tracker.occlusion]):
            logger.info('{} statistics'.format(what))
            for k, v in dd.items():
                logger.info(
                    '{}: \t {} \t {}'.format(
                        k, sum(v) / len(v), len(v)))
                print('{}: \t {} \t {}'.format(
                        k, sum(v) / len(v), len(v)))

    def eval_fair_mot(self, names):
        # EVALUATION FROM FAIRMOT oder so
        accs = list()
        for seq in names:
            evaluator = Evaluator(
                '/storage/slurm/seidensc/datasets/MOT/MOT17/train', seq, 'mot')
            accs.append(
                evaluator.eval_file(
                    os.path.join(
                        'out/' +
                        self.tracker.experiment,
                        seq)))

        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, names, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        logger.info(strsummary)

    def eval_mpn_track(self, names, corresponding_gt):
        if self.dataset_cfg['splits'] != 'mot17_test':
            if self.tracker_cfg['oracle']:
                # Use corresponding gt as gt
                # --> no false negatives
                accs, names = get_results(
                    names,
                    corr_gt_as_gt=corresponding_gt,
                    dir=self.dir,
                    tracker=self.tracker_cfg,
                    dataset_cfg=self.dataset_cfg)
                get_summary(accs, names, "Corresponding GT as GT")

                # Use corresponding gt as tracking results
                # --> perfect association, only FNs
                # --> how good can we be if we associate correctly
                accs, names = get_results(
                    names,
                    corr_gt_as_track_res=corresponding_gt,
                    dir=self.dir,
                    tracker=self.tracker_cfg,
                    dataset_cfg=self.dataset_cfg)
                get_summary(accs, names, "Corresponding GT as results")

            # Evaluate files from tracking
            accs, names = get_results(
                names,
                dir=self.dir,
                tracker=self.tracker,
                dataset_cfg=self.dataset_cfg)
            get_summary(accs, names, "Normal")

    def eval_track_eval(self):
        evaluate_track_eval(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg
        )
