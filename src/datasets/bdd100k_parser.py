import os.path as osp
import os
import pandas as pd
from torchvision.ops import box_iou
import numpy as np
import torch
from copy import deepcopy
import logging
import matplotlib
from .MOT17_parser import MOTLoader


logger = logging.getLogger('AllReIDTracker.BDDParser')


class BDDLoader(MOTLoader):
    def __init__(self, sequence, dataset_cfg, dir, mode='eval', only_pedestrian=True):
        self.dataset_cfg = dataset_cfg
        self.sequence = sequence
        self.only_pedestrian = dataset_cfg['only_pedestrian']
        self.train_mode = self.dataset_cfg['half_train_set_gt'] or mode == 'train'

        self.mot_dir = osp.join(dataset_cfg['mot_dir'])
        self.gt_dir = osp.join(dataset_cfg['gt_dir'], dir)
        self.det_dir = osp.join(dataset_cfg['det_dir'], dir)
        self.det_file = dataset_cfg['det_file']
        self.dir = dir

    def get_seqs(self, split='split-1', assign_gt=True):
        # iterate over sequences
        for s in self.sequence:
            # get gt and detections
            gt_file = osp.join(self.gt_dir, s, 'gt', 'gt_bdd100k.txt')
            det_file = osp.join(self.det_dir, s, 'det', self.det_file)
            self.get_seq_info(s)
            exist_gt = self.seq_info['has_gt'] and assign_gt
            self.get_dets(det_file, s)

            if exist_gt:
                self.get_gt(gt_file)

            # keep unclipped copy of detections and clip
            self.dets_unclipped = deepcopy(self.dets)
            self.dets = self.clip_boxes_to_image(df=self.dets)

            self.dets.sort_values(by='frame', inplace=True)
            self.dets_unclipped.sort_values(by='frame', inplace=True)

            # assign ground trudh
            self.dets['detection_id'] = np.arange(self.dets.shape[0])
            if exist_gt:
                self.assign_gt_clear(split)

            self.dets.attrs.update(self.seq_info)

            # if only using pedestrians remove rest of detections
            if self.only_pedestrian:
                self.dets = self.dets[self.dets['label'] == 0]
                self.gt = self.gt[self.gt['label'] == 0]

        return exist_gt

    def get_gt(self, gt_file):
        """
        Load
        """
        if osp.exists(gt_file):
            self.gt = pd.read_csv(
                gt_file,
                names=[
                    'frame',
                    'id',
                    'bb_left',
                    'bb_top',
                    'bb_width',
                    'bb_height',
                    'conf',
                    'label',
                    'vis'])

            # Coordinates are 1 based
            self.gt['bb_left'] -= 1
            self.gt['bb_top'] -= 1
            self.gt['bb_bot'] = (
                self.gt['bb_top'] +
                self.gt['bb_height']).values
            self.gt['bb_right'] = (
                self.gt['bb_left'] +
                self.gt['bb_width']).values

            # only keep tracking gt
            self.gt = self.gt[self.gt['label'].isin(
                [0, 1, 2, 3, 4, 5, 6, 7])].copy()

            def make_frame(i):
                return int(i.split('-')[-1])
            self.gt['frame'] = self.gt['frame'].apply(make_frame)
        else:
            self.gt = None

    def get_dets(self, det_file, s):
        img_dir = os.path.join(self.mot_dir, 'images', 'track', self.dir, s)
        if osp.exists(det_file):
            names = [
                'frame',
                'id',
                'bb_left',
                'bb_top',
                'bb_width',
                'bb_height',
                'conf',
                'label',
                'vis',
                '?']

            self.dets = pd.read_csv(
                det_file,
                names=names)

            # only keep tracking classes
            self.dets = self.dets[self.dets['label'].isin(
                [0, 1, 2, 3, 4, 5, 6, 7])].copy()

            # Coordinates are 1 based
            self.dets['bb_left'] -= 1
            self.dets['bb_top'] -= 1
            self.dets['bb_right'] = (
                self.dets['bb_left'] +
                self.dets['bb_width']).values  # - 1
            self.dets['bb_bot'] = (
                self.dets['bb_top'] +
                self.dets['bb_height']).values  # - 1

            self.dets['tracktor_id'] = self.dets['id']

            # add frame path
            def add_frame_path(i):
                return osp.join(
                    osp.join(img_dir, s + '-' + f"{i:07d}.jpg"))

            self.dets['frame_path'] = self.dets['frame'].apply(add_frame_path)
    
    def get_seq_info(self, s):
        """
        Get MOT17 like sequence info
        """
        seq_info = dict()
        path = os.path.join(self.mot_dir, 'images', 'track', self.dir, s)
        first_img = os.listdir(path)[0]
        img = matplotlib.image.imread(os.path.join(path, first_img))
        seq_info['name'] = s
        seq_info['imDir'] = path
        seq_info['seqLength'] = len(os.listdir(path))
        seq_info['imWidth'] = img.shape[1]
        seq_info['imHeight'] = img.shape[0]
        seq_info['has_gt'] = self.dir != 'test'
        self.seq_info = seq_info

    def assign_gt_clear(self, split='split-1'):
        """
        Assign ground truth same way as in clear metrics
        """
        cols = [
            'frame',
            'id',
            'bb_left',
            'bb_top',
            'bb_width',
            'bb_height',
            'conf',
            'label',
            'vis']
        self.corresponding_gt = pd.DataFrame(columns=cols)

        # check for consecutive
        if not self.checkConsecutive(
                set(sorted(self.dets['id'].values.tolist()))):
            print("non cosecutive dets")

        if not self.checkConsecutive(set(sorted(self.gt['id'].values.tolist()))):
            print("non cosecutive gt")
            self.gt = self.make_consecutive(self.gt)
            print("now consecutive: {}".format(self.checkConsecutive(
                set(sorted(self.gt['id'].values.tolist())))))

        num_gt_ids = len(set(sorted(self.gt['id'].values.tolist())))
        prev_timestep_tracker_id = np.nan * \
            np.zeros(num_gt_ids)  # For matching IDSW
        # distractor_classes = [2, 7, 8, 12]
        from scipy.optimize import linear_sum_assignment
        for frame in self.dets['frame'].unique():
            # get df entries of current frame
            frame_detects = self.dets[self.dets.frame == frame]
            frame_gt = self.gt[self.gt.frame == frame]

            # Compute IoU for each pair of detected / GT bounding box
            similarity = box_iou(torch.tensor(frame_gt[[
                'bb_top', 'bb_left', 'bb_bot', 'bb_right']].values).double(
            ), torch.tensor(frame_detects[[
                'bb_top', 'bb_left', 'bb_bot', 'bb_right']].values).double())
            # get initial match to remove distractor classes
            matching_scores = deepcopy(similarity)
            matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
            match_rows, match_cols = linear_sum_assignment(-matching_scores)
            actually_matched_mask = matching_scores[match_rows,
                                                    match_cols] > 0 + np.finfo(
                                                        'float').eps
            match_rows = match_rows[actually_matched_mask.numpy()]
            match_cols = match_cols[actually_matched_mask.numpy()]

            # get current IDs for score matrix
            tracker_ids_t = frame_detects['id'].values
            gt_ids_t = frame_gt['id'].values
            score_mat = (tracker_ids_t[np.newaxis, :] ==
                         prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])

            score_mat = 1000 * score_mat + similarity.numpy()
            score_mat[similarity < self.dataset_cfg[
                'gt_assign_min_iou'] - np.finfo('float').eps] = 0

            # Hungarian algorithm to find best matches
            # --> dropped gt (distractor & zero masked) + dropped frames (distractor)
            corresponding_gt, assigned_detects = linear_sum_assignment(
                -score_mat)

            actually_matched_mask = score_mat[corresponding_gt,
                                              assigned_detects] > 0 + np.finfo(
                                                  'float').eps
            corresponding_gt = corresponding_gt[actually_matched_mask]
            assigned_detects = assigned_detects[actually_matched_mask]

            matched_gt_ids = gt_ids_t[corresponding_gt]
            matched_tracker_ids = tracker_ids_t[assigned_detects]

            # get indices of assigned and unassigned frames
            assigned_detect_index = frame_detects.iloc[
                assigned_detects].index
            unassigned_detect_index = list(
                set(frame_detects.index.tolist()) - set(assigned_detect_index))

            # get IDs of assigned gt
            self.corresponding_gt = self.corresponding_gt.append(
                frame_gt.iloc[corresponding_gt])
            corresponding_id = frame_gt.iloc[
                corresponding_gt]['id'].values
            corresponding_vis = frame_gt.iloc[
                corresponding_gt]['vis'].values

            # set IDs and vis of assigned and unassigned
            self.dets.loc[assigned_detect_index, 'id'] = corresponding_id
            self.dets.loc[unassigned_detect_index,
                          'id'] = -1  # False Positives

            self.dets.loc[assigned_detect_index, 'vis'] = corresponding_vis
            self.dets.loc[unassigned_detect_index,
                          'vis'] = -1  # False Positives

            # update prev timestep tracker id
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        if self.dataset_cfg['validation_set']:
            self.dets = self.dets[self.dets['frame'] >
                                  self.dets['frame'].values.max() * 0.5]
        elif self.dataset_cfg['half_train_set_gt']:
            self.dets = self.dets[self.dets['frame'] <=
                                  self.dets['frame'].values.max() * 0.5]
