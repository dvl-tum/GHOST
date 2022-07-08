import os.path as osp
import os
import pandas as pd
from torchvision.ops import box_iou
# from torch_geometric.data import Dataset, Data
import numpy as np
import torch
from lapsolver import solve_dense
from copy import deepcopy
import logging


logger = logging.getLogger('AllReIDTracker.Parser')


class MOTLoader():
    def __init__(self, sequence, dataset_cfg, dir, mode='eval'):
        self.dataset_cfg = dataset_cfg
        self.sequence = sequence
        self.train_mode = self.dataset_cfg['half_train_set_gt'] or mode == 'train'

        self.mot_dir = osp.join(dataset_cfg['mot_dir'], dir)
        self.det_dir = osp.join(dataset_cfg['det_dir'], dir)
        self.det_file = dataset_cfg['det_file']

    def get_seqs(self, split='split-1', use_clear=True):
        for s in self.sequence:
            gt_file = osp.join(self.mot_dir, s, 'gt', 'gt.txt')
            exist_gt = os.path.isfile(gt_file)
            if self.det_file == 'gt.txt':
                det_file = osp.join(self.det_dir, s, 'gt', self.det_file)
            else:
                det_file = osp.join(self.det_dir, s, 'det', self.det_file)
            seq_file = osp.join(self.mot_dir, s, 'seqinfo.ini')

            self.get_seq_info(seq_file, gt_file, det_file)
            self.get_dets(det_file, s)
            if exist_gt:
                self.get_gt(gt_file)
            self.dets_unclipped = deepcopy(self.dets)
            self.dets = self.clip_boxes_to_image(df=self.dets)

            '''# clip gt for assignment
            if exist_gt:
                self.gt = self.clip_boxes_to_image(df=self.gt)'''

            self.dets.sort_values(by='frame', inplace=True)
            self.dets_unclipped.sort_values(by='frame', inplace=True)

            self.dets['detection_id'] = np.arange(self.dets.shape[0])
            if exist_gt:
                if use_clear:
                    self.assign_gt_clear(split)
                else:
                    self.assign_gt(split)
            self.dets.attrs.update(self.seq_info)

        return exist_gt

    def get_gt(self, gt_file):
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
            self.gt['bb_left'] -= 1  # Coordinates are 1 based
            self.gt['bb_top'] -= 1
            self.gt['bb_bot'] = (
                self.gt['bb_top'] +
                self.gt['bb_height']).values
            self.gt['bb_right'] = (
                self.gt['bb_left'] +
                self.gt['bb_width']).values

            self.gt = self.gt[self.gt['label'].isin([1, 2, 7, 8, 12])].copy()
        else:
            self.gt = None

    def get_dets(self, det_file, s):
        img_dir = osp.join(self.mot_dir, s, 'img1')

        if osp.exists(det_file):
            self.dets = pd.read_csv(
                det_file,
                names=[
                    'frame',
                    'id',
                    'bb_left',
                    'bb_top',
                    'bb_width',
                    'bb_height',
                    'conf',
                    'label',
                    'vis',
                    '?'])

            self.dets['bb_left'] -= 1  # Coordinates are 1 based
            self.dets['bb_top'] -= 1
            self.dets['bb_right'] = (
                self.dets['bb_left'] +
                self.dets['bb_width']).values  # - 1
            self.dets['bb_bot'] = (
                self.dets['bb_top'] +
                self.dets['bb_height']).values  # - 1

            self.dets['tracktor_id'] = self.dets['id']

            # add frame path
            def add_frame_path_MOT(i):
                if type(i) == float:
                    i = int(i)
                return osp.join(osp.join(img_dir, f"{i:06d}.jpg"))
            
            def add_frame_path_dance(i):
                if type(i) == float:
                    i = int(i)
                return osp.join(osp.join(img_dir, f"{i:08d}.jpg"))
            
            if 'Dance' in self.mot_dir:
                add_frame_path = add_frame_path_dance
            else:
                add_frame_path = add_frame_path_MOT

            self.dets['frame_path'] = self.dets['frame'].apply(add_frame_path)

    def get_seq_info(self, seq_file, gt_file, det_file):
        seq_info = pd.read_csv(seq_file, sep='=').to_dict()['[Sequence]']
        seq_info['path'] = osp.dirname(seq_file)
        seq_info['has_gt'] = osp.isfile(gt_file)
        seq_info['is_gt'] = det_file == gt_file
        self.seq_info = seq_info

    def clip_boxes_to_image(self, df):
        img_height, img_width = self.seq_info['imHeight'], self.seq_info['imWidth']
        img_height = np.array([int(img_height)] * df.shape[0])
        img_width = np.array([int(img_width)] * df.shape[0])

        # top and left
        initial_bb_top = df['bb_top'].values.copy()
        initial_bb_left = df['bb_left'].values.copy()
        df['bb_top'] = np.maximum(df['bb_top'].values, 0).astype(int)
        df['bb_left'] = np.maximum(df['bb_left'].values, 0).astype(int)

        # bottom and right
        initial_bb_bot = df['bb_bot'].values.copy()
        initial_bb_right = df['bb_right'].values.copy()
        df['bb_bot'] = np.minimum(img_height, df['bb_bot']).astype(int)
        df['bb_right'] = np.minimum(img_width, df['bb_right']).astype(int)

        # width and height
        bb_top_diff = df['bb_top'].values - initial_bb_top
        bb_left_diff = df['bb_left'].values - initial_bb_left
        df['bb_height'] -= bb_top_diff
        df['bb_width'] -= bb_left_diff
        df['bb_height'] = np.minimum(
            img_height - df['bb_top'],
            df['bb_height']).astype(int)
        df['bb_width'] = np.minimum(
            img_width - df['bb_left'],
            df['bb_width']).astype(int)

        # double check
        conds = (df['bb_width'] > 0) & (df['bb_height'] > 0)
        conds = conds & (df['bb_right'] > 0) & (df['bb_bot'] > 0)
        conds = conds & (
            df['bb_left'] < img_width) & (
            df['bb_top'] < img_height)
        df = df[conds].copy()

        return df

    def assign_gt_clear(self, split='split-1'):
        if split == '50-50-1' or split == '50-50-2':
            test_data = pd.read_csv('test_data.csv')
            test_data = test_data[test_data['Sequence'] ==
                                  '-'.join(self.sequence[0].split('-')[:-1])]
            test_data_gt = self.gt.iloc[test_data['path'].values.tolist()]
            test_data_ids = test_data_gt['id'].unique()

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

        def checkConsecutive(l):
            return sorted(l) == list(range(min(l), max(l) + 1))

        def make_consecutive(df):
            labs = set(sorted(df['id'].values.tolist()))
            lab_map = {l: i for i, l in enumerate(labs)}
            new_labs = [lab_map[l] for l in df['id'].values]
            df['id'] = new_labs
            return df

        if not checkConsecutive(
                set(sorted(self.dets['id'].values.tolist()))):
            print("non cosecutive dets")

        if not checkConsecutive(set(sorted(self.gt['id'].values.tolist()))):
            print("non cosecutive gt")
            self.gt = make_consecutive(self.gt)
            print("now consecutive: {}".format(checkConsecutive(
                set(sorted(self.gt['id'].values.tolist())))))

        num_gt_ids = len(set(sorted(self.gt['id'].values.tolist())))

        prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
        prev_timestep_tracker_id = np.nan * \
            np.zeros(num_gt_ids)  # For matching IDSW
        distractor_classes = [2, 7, 8, 12]
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

            # remove tracked bbs that correspond to distractor classes
            is_distractor_class = np.isin(
                frame_gt['label'].values[match_rows],
                distractor_classes)
            to_remove_tracker = match_cols[is_distractor_class]
            frame_detects_drop = deepcopy(frame_detects).drop(
                frame_detects.index.values[to_remove_tracker])
            similarity = np.delete(similarity, to_remove_tracker, axis=1)

            # only keep gt of person class + confidence = 1
            gt_zero_marked = frame_gt['conf'].values
            gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                (np.equal(frame_gt['label'].values, 1))

            # remove gt its that are not of person class or have confidence 0
            similarity = similarity[gt_to_keep_mask]
            frame_gt_drop = deepcopy(frame_gt).drop(
                frame_gt.index.values[~gt_to_keep_mask])

            # get current IDs for score matrix
            tracker_ids_t = frame_detects_drop['id'].values
            gt_ids_t = frame_gt_drop['id'].values
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

            # get unassigned bbs
            # unassigned_detect = np.array(list(set(range(frame_detects_drop.shape[0])) - set(assigned_detects)))

            # get indices of assigned and unassigned frames
            assigned_detect_index = frame_detects_drop.iloc[
                assigned_detects].index
            unassigned_detect_index = list(
                set(frame_detects.index.tolist()) - set(assigned_detect_index))

            # get IDs of assigned gt
            self.corresponding_gt = self.corresponding_gt.append(
                frame_gt_drop.iloc[corresponding_gt])
            corresponding_id = frame_gt_drop.iloc[
                corresponding_gt]['id'].values
            corresponding_vis = frame_gt_drop.iloc[
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

        if split == '50-50-1':
            self.dets = self.dets[self.dets['id'].isin(test_data_ids)]
        elif split == '50-50-2':
            self.dets = self.dets[~self.dets['id'].isin(test_data_ids)]

        if self.dataset_cfg['validation_set'] and not self.train_mode:
            self.dets = self.dets[self.dets['frame'] >
                                  self.dets['frame'].values.max() * 0.5]
        elif self.dataset_cfg['validation_set'] and self.train_mode:
            self.dets = self.dets[self.dets['frame'] <=
                                  self.dets['frame'].values.max() * 0.5]

        if self.dataset_cfg['drop_unassigned']:
            mask = self.dets['id'] != -1
            self.dets = self.dets[mask]

    def assign_gt(self, split='split-1'):
        if split == '50-50-1' or split == '50-50-2':
            test_data = pd.read_csv('test_data.csv')
            test_data = test_data[test_data['Sequence'] ==
                                  '-'.join(self.sequence[0].split('-')[:-1])]
            test_data_gt = self.gt.iloc[test_data['path'].values.tolist()]
            test_data_ids = test_data_gt['id'].unique()

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
        if len(set(self.dets['frame'].unique().tolist()).intersection(set(
                self.gt['frame'].unique().tolist()))) < 0.75 * len(self.gt['frame'].unique().tolist()):
            if np.min(self.dets['frame'].unique()) == 1:
                max_f = np.max(self.gt['frame'].unique()) - \
                    np.max(self.dets['frame'].unique())
                self.gt['frame'] = self.gt['frame'] - max_f

        if self.seq_info['has_gt']:
            for frame in self.dets['frame'].unique():
                # get df entries of current frame
                frame_detects = self.dets[self.dets.frame == frame]
                frame_gt = self.gt[self.gt.frame == frame]

                # Compute IoU for each pair of detected / GT bounding box
                iou_matrix = box_iou(torch.tensor(frame_detects[[
                    'bb_top', 'bb_left', 'bb_bot', 'bb_right']].values),
                    torch.tensor(frame_gt[[
                    'bb_top', 'bb_left', 'bb_bot', 'bb_right']].values))
                iou_matrix[iou_matrix <
                           self.dataset_cfg['gt_assign_min_iou']] = 0  # np.nan
                dist = 1 - iou_matrix
                assigned_detects, corresponding_gt = solve_dense(dist)
                unassigned_detect = np.array(
                    list(set(range(frame_detects.shape[0])) - set(assigned_detects)))

                # get indices of assigned and unassigned frames
                assigned_detect_index = frame_detects.iloc[assigned_detects].index
                unassigned_detect_index = frame_detects.iloc[unassigned_detect].index

                # get IDs of assigned gt
                self.corresponding_gt = self.corresponding_gt.append(
                    frame_gt.iloc[corresponding_gt])
                corresponding_id = frame_gt.iloc[corresponding_gt]['id'].values
                corresponding_vis = frame_gt.iloc[corresponding_gt]['vis'].values

                # set IDs and vis of assigned and unassigned
                self.dets.loc[assigned_detect_index, 'id'] = corresponding_id
                self.dets.loc[unassigned_detect_index,
                              'id'] = -1  # False Positives

                self.dets.loc[assigned_detect_index, 'vis'] = corresponding_vis
                self.dets.loc[unassigned_detect_index,
                              'vis'] = -1  # False Positives

        if split == '50-50-1':
            self.dets = self.dets[self.dets['id'].isin(test_data_ids)]
        elif split == '50-50-2':
            self.dets = self.dets[~self.dets['id'].isin(test_data_ids)]

        if self.dataset_cfg['validation_set']:
            self.dets = self.dets[self.dets['frame'] >
                                  self.dets['frame'].values.max() * 0.5]

        # pd.set_option('display.max_columns', None)
        # print(self.dets[self.dets['id'] == -1])
        # self.dets[self.dets['id'] == -1].to_csv('unassigned_check.csv')

        if self.dataset_cfg['drop_unassigned']:
            mask = self.dets['id'] != -1
            self.dets = self.dets[mask]
            self.dets_unclipped = self.dets_unclipped[mask]

        # print(self.dets.shape)il
        # print(self.dets_unclipped.shape)
        # print(self.dets)
        # print(self.dets_unclipped)
        # quit()
