from cProfile import label
from collections import defaultdict
import copy
from re import L
from matplotlib.pyplot import imsave
from numpy.core.fromnumeric import size

from pandas.core.indexing import IndexingMixin
import torch
import sklearn.metrics
#from tracking_wo_bnw.src.tracktor.utils import interpolate
import os
import numpy as np
import logging
from lapsolver import solve_dense
import json
from math import floor
from src.tracking_utils import bisoftmax, get_proxy, Track, multi_predict, get_iou_kalman
from src.base_tracker import BaseTracker
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn


logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker(BaseTracker):
    def __init__(
            self,
            tracker_cfg,
            encoder,
            net_type='resnet50',
            output='plain',
            weight='No',
            data='tracktor_preprocessed_files.txt',
            device='cpu',
            train_cfg=None):
        super(
            Tracker,
            self).__init__(
            tracker_cfg,
            encoder,
            net_type,
            output,
            weight,
            data,
            device,
            train_cfg)
        self.short_experiment = defaultdict(list)

    def track(self, seq, first=False, log=True):
        '''
        first - feed all bounding boxes through net first for bn stats update
        seq -   sequence instance for iteratration and with meta information
                like name or lentgth
        '''
        self.log = log
        if self.log:
            logger.info(self.experiment)
            logger.info(
                "Tracking sequence {} of lenght {}".format(
                    seq.name, seq.num_frames))
        self.setup_seq(seq, first)

        # batch norm experiemnts I
        self.normalization_before(seq, first=first)

        # iterate over frames
        for i, (frame, _, path, boxes, _, gt_ids, vis,
                random_patches, whole_im, frame_size, _, conf, label) in enumerate(seq):
            # batch norm experiments II
            self.normalization_experiments(random_patches, frame, i)
            if 'bdd' in path:
                self.frame_id = int(path.split('/')[-1].split('-')[-1].split('.')[0])
            else:
                self.frame_id = int(path.split(os.sep)[-1][:-4])
            if self.frame_id % 20 == 0:
                logger.info(self.frame_id)

            detections = list()

            # forward pass
            feats = self.get_features(frame)

            # just feeding for bn stats update
            if i == 0:
                print('Mode in training mode: ', self.encoder.training)
            if first:
                continue
            
            # iterate over bbs in current frame
            for f, b, gt_id, v, c, l in zip(feats, boxes, gt_ids, vis, conf, label):
                if (b[3] - b[1]) / (b[2] - b[0]
                                    ) < self.tracker_cfg['h_w_thresh']:
                    if 'byte' in self.data and c < 0.6:
                        continue
                    if 'bdd' in self.data and c < self.tracker_cfg['thresh']:
                        continue
                    detection = {
                        'bbox': b,
                        'feats': f,
                        'im_index': self.frame_id,
                        'gt_id': gt_id,
                        'vis': v,
                        'conf': c,
                        'frame': self.frame_id,
                        'label': l}
                    detections.append(detection)

                    if self.store_feats:
                        self.add_feats_to_storage(detections)

            # apply motion compensation to stored track positions
            if self.motion_model_cfg['motion_compensation']:
                print("NC")
                self.motion_compensation(whole_im, i)

            # association over frames
            tr_ids = self._track(detections, i, frame=frame)

            if self.store_visualization:
                self.visualize(detections, tr_ids, path, seq.name, i+1)

        # just fed for bn stats update
        if first:
            logger.info('Done with pre-tracking feed...')
            i = 0
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if i == 0:
                        print(m, m.running_mean, m.running_var)
                        i += 1
            return

        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)

        # write results
        self.write_results(self.output_dir, seq.name)

        # reset thresholds if every / tbd
        self.reset_threshs()

        # store dist to json file
        if self.store_dist:
            path = os.path.join('distances', self.experiment + 'distances.json')
            with open(path, 'w') as jf:
                json.dump(self.distance_, jf)
        if self.store_feats:
            os.makedirs('features', exist_ok=True)
            path = os.path.join('features', self.experiment + 'features.json')
            with open(path, 'w') as jf:
                json.dump(self.features_, jf)

    def _track(self, detections, i, frame=None):
        # just add all bbs to self.tracks / intitialize in the first frame
        if len(self.tracks) == 0:
            tr_ids = list()
            for detection in detections:
                self.tracks[self.id] = Track(
                    track_id=self.id,
                    **detection,
                    kalman=self.kalman,
                    kalman_filter=self.kalman_filter)
                tr_ids.append(self.id)
                self.id += 1

        # association over frames for frame > 0
        elif i > 0:
            # get hungarian matching
            if len(detections) > 0:
                if not self.tracker_cfg['avg_inact']['proxy'] == 'each_sample':
                    dist, row, col, ids = self.get_hungarian_with_proxy(
                        detections, sep=self.tracker_cfg['assign_separately'])
                else:
                    dist, row, col, ids = self.get_hungarian_each_sample(
                        detections, sep=self.tracker_cfg['assign_separately'])
            else:
                dist, row, col, ids = 0, 0, 0, 0
            if dist is not None:
                # get bb assignment
                tr_ids = self.assign(
                    detections=detections,
                    dist=dist,
                    row=row,
                    col=col,
                    ids=ids,
                    sep=self.tracker_cfg['assign_separately'])
        return tr_ids

    def get_hungarian_each_sample(self, detections, nan_over_classes=True, sep=False):
        # get new detections
        x = torch.stack([t['feats'] for t in detections])
        dist_all, ids = list(), list()

        if nan_over_classes:
            labels_dets = np.array([t['label'] for t in detections])
        
        # if use each sample for active frames
        if not self.tracker_cfg['avg_act']['do'] and len(detections) > 0:
            y = torch.stack([t.feats for t in self.tracks.values()])
            ids.extend([i for i in self.tracks.keys()])
            dist = self.dist(x, y).T
            dist = dist.cpu().numpy()
            if nan_over_classes:
                labels = np.array([t.label[-1] for t in self.tracks.values()])
                label_mask = np.atleast_2d(labels).T == np.atleast_2d(labels_dets)
                dist[~label_mask] = np.nan
            dist_all.extend([d for d in dist])
        else:
            for id, tr in self.tracks.items():
                y = torch.stack(tr.past_feats)
                ids.append(id)
                dist = self.dist(x, y)
                dist = dist.cpu().numpy()

                if self.tracker_cfg['avg_act']['num'] == 1:
                    dist_all.append(np.min(dist, axis=1))
                elif self.tracker_cfg['avg_act']['num'] == 2:
                    dist_all.append(np.mean(dist, axis=1))
                elif self.tracker_cfg['avg_act']['num'] == 3:
                    dist_all.append(np.max(dist, axis=1))
                elif self.tracker_cfg['avg_act']['num'] == 4:
                    dist_all.append(
                        (np.max(dist, axis=1) + np.min(dist, axis=1)) / 2)

        num_active = len(ids)

        # get distances to inactive tracklets (inacht thresh = 100000)
        curr_it = {k: track for k, track in self.inactive_tracks.items()
                   if track.inactive_count <= self.inact_thresh}
        if len(curr_it) > 0:
            if not self.tracker_cfg['avg_inact']['do']:
                y = torch.stack([t.feats for t in curr_it.values()])
                ids.extend([i for i in curr_it.keys()])
                dist = self.dist(x, y).T
                dist = dist.cpu().numpy()
                dist_all.extend([d for d in dist])
            else:
                for id, tr in curr_it.items():
                    y = torch.stack(tr.past_feats)
                    ids.append(id)
                    dist = self.dist(x, y)
                    dist = dist.cpu().numpy()

                    if self.tracker_cfg['avg_inact']['num'] == 1:
                        dist = np.min(dist, axis=1)
                    elif self.tracker_cfg['avg_inact']['num'] == 2:
                        dist = np.mean(dist, axis=1)
                    elif self.tracker_cfg['avg_inact']['num'] == 3:
                        dist = np.max(dist, axis=1)
                    elif self.tracker_cfg['avg_inact']['num'] == 4:
                        dist = (np.max(dist, axis=1) + np.min(dist, axis=1))/2

                    if nan_over_classes:
                        label_mask = np.atleast_2d(np.array(tr.label[-1])) == \
                            np.atleast_2d(labels_dets).T
                        dist[~label_mask.squeeze()] = np.nan
                    dist_all.append(dist)

        num_inactive = len(curr_it)

        # solve assignment problem
        dist = np.vstack(dist_all).T

        # update thresholds
        self.update_thresholds(dist, num_active, num_inactive)
        
        if self.motion_model_cfg['apply_motion_model']:
            if not self.kalman:
                self.motion()
                iou = self.get_motion_dist(detections, curr_it)
            else:
                self.motion(only_vel=True)
                stracks = multi_predict(
                    self.tracks,
                    curr_it,
                    self.shared_kalman)
                iou = get_iou_kalman(stracks, detections)
            dist = self.combine_motion_appearance(
                iou,
                dist)

        if self.nan_first:
            dist[:, :num_active] = np.where(dist[:, :num_active] <=
                self.act_reid_thresh, dist[:, :num_active], np.nan)
            dist[:, num_active:] = np.where(dist[:, num_active:] <=
                self.inact_reid_thresh, dist[:, num_active:], np.nan)

        if not sep:
            row, col = solve_dense(dist)
        else:
            dist_act = dist[:, :num_active]
            row, col = solve_dense(dist_act)
            if num_active > 0:
                dist_inact = dist[:, num_active:]
            else:
                dist_inact = None
            dist = [dist_act, dist_inact]

        if self.store_dist:
            self._add_dist(detections, curr_it, num_active, num_inactive, dist)

        return dist, row, col, ids

    def _add_dist(self, detections, curr_it, num_active, num_inactive, dist):
        gt_n = [v['gt_id'] for v in detections]
        gt_t = list()
        if num_active:
            gt_t += [track.gt_id for track in self.tracks.values()]
        if num_inactive:
            gt_t += [track.gt_id for track in curr_it.values()]
        self.add_dist_to_storage(
            gt_n, gt_t, num_active, num_inactive, dist)

    def get_hungarian_with_proxy(self, detections, sep=False):
        # instantiate
        ids = list()
        y_inactive, y = None, None

        x = torch.stack([t['feats'] for t in detections])
        height = [int(v['bbox'][3]-v['bbox'][1]) for v in detections]

        # Get active tracklets
        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = get_proxy(
                    curr_it=self.tracks,
                    mode='act',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y = torch.stack([track.feats for track in self.tracks.values()])

            ids += list(self.tracks.keys())
            num_active = len(ids)

        # get inactive tracklets (inacht thresh = 100000)
        curr_it = {k: track for k, track in self.inactive_tracks.items()
                   if track.inactive_count <= self.inact_thresh}
        # if there are inactive tracks that fall into inact_thresh
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = get_proxy(
                    curr_it=curr_it,
                    mode='inact',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y_inactive = torch.stack([track.feats
                                         for track in curr_it.values()])

            if len(self.tracks) > 0 and not sep:
                y = torch.cat([y, y_inactive])
            elif not sep:
                y = y_inactive
                num_active = 0

            ids += [k for k in curr_it.keys()]
            num_inactive = len(curr_it)

        # if no active or inactive tracks --> return and instantiate all dets
        # new
        elif len(curr_it) == 0 and len(self.tracks) == 0:
            for detection in detections:
                self.tracks[self.id] = Track(
                    track_id=self.id,
                    **detection,
                    kalman=self.kalman,
                    kalman_filter=self.kalman_filter)
                self.id += 1
            return None, None, None, None
        # if there are active but no inactive
        else:
            num_inactive = 0

        # compute distance
        if not sep:
            if not self.tracker_cfg['use_bism']:
                dist = self.dist(x, y)
                dist = dist.cpu().numpy()
            else:
                dist = 1 - bisoftmax(x.cpu(), y.cpu())

            if self.motion_model_cfg['apply_motion_model']:
                self.motion()
                iou = self.get_motion_dist(detections, curr_it)
                dist = self.combine_motion_appearance(
                    iou,
                    dist)

            # update thresholds
            self.update_thresholds(dist, num_active, num_inactive)

            if self.nan_first:
                dist[:, :num_active] = np.where(dist[:, :num_active] <=
                    self.act_reid_thresh, dist[:, :num_active], np.nan)
                dist[:, num_active:] = np.where(dist[:, num_active:] <=
                    self.inact_reid_thresh, dist[:, num_active:], np.nan)

            row, col = solve_dense(dist)
        else:
            dist_act = self.dist(x, y)
            dist_act = dist_act.cpu().numpy()
            
            row, col = solve_dense(dist_act)
            if y_inactive is not None:
                dist_inact = self.dist(x, y_inactive)
                dist_inact = dist_inact.cpu().numpy()
            else:
                dist_inact = None
            dist = [dist_act, dist_inact]

        if self.store_dist:
            self._add_dist(detections, curr_it, num_active, num_inactive, dist)

        return dist, row, col, ids

    def assign(self, detections, dist, row, col, ids, sep=False):
        # assign tracks from hungarian
        active_tracks = list()
        tr_ids = [None for _ in range(len(detections))]
        if len(detections) > 0:
            if not sep:
                assigned = self.assign_act_inact_same_time(
                    row, col, dist, detections, active_tracks, ids, tr_ids)
            else:
                assigned = self.assign_separatly(
                    row, col, dist, detections, active_tracks, ids, tr_ids)

        # move tracks not used to inactive tracks
        keys = list(self.tracks.keys())
        for k in keys:
            if k not in active_tracks:
                self.inactive_tracks[k] = self.tracks[k]
                del self.tracks[k]
                self.inactive_tracks[k].inactive_count += 0

        # increase inactive count by one
        for k in self.inactive_tracks.keys():
            self.inactive_tracks[k].inactive_count += 1

        for i in range(len(detections)):
            if i not in assigned:
                self.tracks[self.id] = Track(
                    track_id=self.id,
                    **detections[i],
                    kalman=self.kalman,
                    kalman_filter=self.kalman_filter)
                tr_ids[i] = self.id
                self.id += 1
        return tr_ids

    def assign_act_inact_same_time(
            self,
            row,
            col,
            dist,
            detections,
            active_tracks,
            ids,
            tr_ids):
        # assigned contains all new detections that have been assigned
        assigned = list()
        for r, c in zip(row, col):

            # assign tracks to active tracks if reid distance < thresh
            if ids[c] in self.tracks.keys() and \
                    dist[r, c] < self.act_reid_thresh:
                
                self.tracks[ids[c]].add_detection(**detections[r])
                active_tracks.append(ids[c])
                assigned.append(r)
                tr_ids[r] = ids[c]

            # assign tracks to inactive tracks if reid distance < thresh
            elif ids[c] in self.inactive_tracks.keys() and \
                    dist[r, c] < self.inact_reid_thresh:
                # move inactive track to active
                self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                del self.inactive_tracks[ids[c]]
                self.tracks[ids[c]].inactive_count = 0

                self.tracks[ids[c]].add_detection(**detections[r])
                active_tracks.append(ids[c])
                assigned.append(r)
                tr_ids[r] = ids[c]

        return set(assigned)

    def assign_separatly(
            self,
            row,
            col,
            dist,
            detections,
            active_tracks,
            ids,
            tr_ids):
        # assign active tracks first
        assigned = self.assign_act_inact_same_time(
            row,
            col,
            dist[0],
            detections,
            active_tracks,
            ids[:dist[0].shape[1]],
            tr_ids)

        # assign inactive tracks
        if dist[1] is not None:
            # only use detections that have not been assigned yet
            # u = unassigned
            u = sorted(
                list(set(list(range(dist[0].shape[0]))) - assigned))

            if len(u) != 0:
                dist[1] = dist[1][u, :]

                row_inact, col_inact = solve_dense(dist[1])
                assigned_2 = self.assign_act_inact_same_time(
                    row=row_inact,
                    col=col_inact,
                    dist=dist[1],
                    detections=[t for i, t in enumerate(detections) if i in u],
                    active_tracks=active_tracks,
                    ids=ids[dist[0].shape[1]:],
                    tr_ids=tr_ids)
                assigned_2 = set(
                    [u for i, u in enumerate(u) if i in assigned_2])
                assigned.update(assigned_2)

        return assigned
