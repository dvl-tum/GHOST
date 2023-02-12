from collections import defaultdict
import torch
import os
import numpy as np
import logging
from lapsolver import solve_dense
import json
from src.tracking_utils import get_proxy, Track, multi_predict, get_iou_kalman
from src.base_tracker import BaseTracker
from tqdm import tqdm


logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker(BaseTracker):
    def __init__(
            self,
            tracker_cfg,
            encoder,
            net_type='resnet50',
            output='plain',
            data='tracktor_preprocessed_files.txt',
            device='cpu'):
        super(
            Tracker,
            self).__init__(
            tracker_cfg,
            encoder,
            net_type,
            output,
            data,
            device)
        self.short_experiment = defaultdict(list)
        self.inact_patience = self.tracker_cfg['inact_patience']

    def track(self, seq, first=False, log=True):
        '''
        first - feed all bounding boxes through net first for bn stats update
        seq -   sequence instance for iteratration and with meta information
                like name or lentgth
        '''
        self.log = log
        if self.log:
            logger.info(
                "Tracking sequence {} of lenght {}".format(
                    seq.name, seq.num_frames))

        # initalize variables for sequence
        self.setup_seq(seq, first)

        # batch norm experiemnts I - before iterating over sequence
        self.normalization_before(seq, first=first)
        self.prev_frame = 0
        i = 0
        # iterate over frames
        for frame_data in tqdm(seq, total=len(seq)):
            frame, path, boxes, gt_ids, vis, \
                random_patches, whole_im, conf, label = frame_data
            # log if in training mode
            if i == 0:
                print(f'Network in training mode: {self.encoder.training}')
            self.i = i

            # batch norm experiments II on the fly
            self.normalization_experiments(random_patches, frame, i, seq)

            # get frame id
            if 'bdd' in path:
                self.frame_id = int(path.split(
                    '/')[-1].split('-')[-1].split('.')[0])
            else:
                self.frame_id = int(path.split(os.sep)[-1][:-4])

            # detections of current frame
            detections = list()

            # forward pass
            feats = self.get_features(frame)

            # if just feeding for bn stats update
            if first:
                continue

            # iterate over bbs in current frame
            for f, b, gt_id, v, c, l in zip(feats, boxes, gt_ids, vis, conf, label):
                if (b[3] - b[1]) / (b[2] - b[0]
                                    ) < self.tracker_cfg['h_w_thresh']:
                    if c < self.tracker_cfg['det_conf']:
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

                    # store features
                    if self.store_feats:
                        self.add_feats_to_storage(detections)

            # apply motion compensation to stored track positions
            if self.motion_model_cfg['motion_compensation']:
                self.motion_compensation(whole_im, i)

            # association over frames
            tr_ids = self._track(detections, i)

            # visualize bounding boxes
            if self.store_visualization:
                self.visualize(detections, tr_ids, path, seq.name, i+1)

            # get previous frame
            self.prev_frame = self.frame_id

            # increase count
            i += 1

        # just fed for bn stats update
        if first:
            logger.info('Done with pre-tracking feed...')
            return

        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)

        # write results
        self.write_results(self.output_dir, seq.name)

        # reset thresholds if every / tbd for next sequence
        self.reset_threshs()

        # store features
        if self.store_feats:
            os.makedirs('features', exist_ok=True)
            path = os.path.join('features', self.experiment + 'features.json')
            logger.info(f'Storing features to {path}...')
            if os.path.isfile(path):
                with open(path, 'r') as jf:
                    features_ = json.load(jf)
                self.features_.update(features_)

            with open(path, 'w') as jf:
                json.dump(self.features_, jf)
            self.features_ = defaultdict(dict)

    def _track(self, detections, i):
        # get inactive tracks with inactive < patience
        self.curr_it = {k: track for k, track in self.inactive_tracks.items()
                        if track.inactive_count <= self.inact_patience}

        # just add all bbs to self.tracks / intitialize in the first frame
        if len(self.tracks) == 0 and len(self.curr_it) == 0:
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

                # get proxy features of tracks first and compute distance then
                if not self.tracker_cfg['avg_inact']['proxy'] == 'each_sample':
                    dist, row, col, ids = self.get_hungarian_with_proxy(
                        detections, sep=self.tracker_cfg['assign_separately'])

                # get aveage of distances to all detections in track --> proxy dist
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

    def last_frame(self, ids, tracks, x, nan_over_classes, labels_dets):
        """
        Get distance of detections to last frame of tracks
        """
        y = torch.stack([t.feats for t in tracks.values()])
        ids.extend([i for i in tracks.keys()])
        dist = self.dist(x, y).T

        # set distance between matches of different classes to nan
        if nan_over_classes:
            labels = np.array([t.label[-1] for t in tracks.values()])
            label_mask = np.atleast_2d(labels).T == np.atleast_2d(labels_dets)
            dist[~label_mask] = np.nan
        return dist

    def proxy_dist(self, tr, x, nan_over_classes, labels_dets):
        """
        Compute proxy distances using all detections in given track
        """
        # get distance between detections and all dets of track
        y = torch.stack(tr.past_feats)
        dist = self.dist(x, y)

        # reduce
        if self.tracker_cfg['avg_inact']['num'] == 1:
            dist = np.min(dist, axis=1)
        elif self.tracker_cfg['avg_inact']['num'] == 2:
            dist = np.mean(dist, axis=1)
        elif self.tracker_cfg['avg_inact']['num'] == 3:
            dist = np.max(dist, axis=1)
        elif self.tracker_cfg['avg_inact']['num'] == 4:
            dist = (np.max(dist, axis=1) + np.min(dist, axis=1))/2
        elif self.tracker_cfg['avg_inact']['num'] == 5:
            dist = np.median(dist, axis=1)

        # nan over classes
        if nan_over_classes:
            label_mask = np.atleast_2d(np.array(tr.label[-1])) == \
                np.atleast_2d(labels_dets).T
            dist[~label_mask.squeeze()] = np.nan

        return dist

    def get_hungarian_each_sample(
            self, detections, nan_over_classes=True, sep=False):
        """
        Get distances using proxy distance, i.e., the average distance
        to all detections in given track
        """

        # get new detections
        x = torch.stack([t['feats'] for t in detections])
        dist_all, ids = list(), list()

        # if setting dist values between classes to nan before hungarian
        if nan_over_classes:
            labels_dets = np.array([t['label'] for t in detections])

        # distance to active tracks
        if len(self.tracks) > 0:

            # just compute distance to last detection of active track
            if not self.tracker_cfg['avg_act']['do'] and len(detections) > 0:
                dist = self.last_frame(
                    ids, self.tracks, x, nan_over_classes, labels_dets)
                dist_all.extend([d for d in dist])

            # if use each sample for active frames
            else:
                for id, tr in self.tracks.items():
                    # get distance between detections and all dets of track
                    ids.append(id)
                    dist = self.proxy_dist(
                        tr, x, nan_over_classes, labels_dets)
                    dist_all.append(dist)

        # get number of active tracks
        num_active = len(ids)

        # get distances to inactive tracklets (inacht thresh = 100000)
        curr_it = self.curr_it
        if len(curr_it) > 0:
            if not self.tracker_cfg['avg_inact']['do']:
                dist = self.last_frame(
                    ids, curr_it, x, nan_over_classes, labels_dets)
                dist_all.extend([d for d in dist])
            else:
                for id, tr in curr_it.items():
                    ids.append(id)
                    dist = self.proxy_dist(
                        tr, x, nan_over_classes, labels_dets)
                    dist_all.append(dist)

        # stack all distances
        dist = np.vstack(dist_all).T

        dist, row, col = self.solve_hungarian(
            dist, num_active, detections, curr_it, sep)

        return dist, row, col, ids

    def solve_hungarian(self, dist, num_active, detections, curr_it, sep):
        """
        Solve hungarian assignment
        """
        # update thresholds
        self.update_thresholds(dist, num_active, len(curr_it))

        # get motion distance
        if self.motion_model_cfg['apply_motion_model']:

            # simple linear motion model
            if not self.kalman:
                self.motion()
                iou = self.get_motion_dist(detections, curr_it)

            # kalman fiter
            else:
                self.motion(only_vel=True)
                stracks = multi_predict(
                    self.tracks,
                    curr_it,
                    self.shared_kalman)
                iou = get_iou_kalman(stracks, detections)

            # combine motion distances
            dist = self.combine_motion_appearance(iou, dist)

        # set values larger than thershold to nan --> impossible assignment
        if self.nan_first:
            dist[:, :num_active] = np.where(
                dist[:, :num_active] <= self.act_reid_thresh, dist[:, :num_active], np.nan)
            dist[:, num_active:] = np.where(
                dist[:, num_active:] <= self.inact_reid_thresh, dist[:, num_active:], np.nan)

        # solve at once
        if not sep:
            row, col = solve_dense(dist)

        # solve active first and inactive later
        else:
            dist_act = dist[:, :num_active]
            row, col = solve_dense(dist_act)
            if num_active > 0:
                dist_inact = dist[:, num_active:]
            else:
                dist_inact = None
            dist = [dist_act, dist_inact]

        return dist, row, col

    def get_hungarian_with_proxy(self, detections, sep=False):
        """
        Use proxy feature vectors for distance computation
        """
        # instantiate
        ids = list()
        y_inactive, y = None, None

        x = torch.stack([t['feats'] for t in detections])

        # Get active track proxies
        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = get_proxy(
                    curr_it=self.tracks,
                    mode='act',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y = torch.stack(
                    [track.feats for track in self.tracks.values()])
            ids += list(self.tracks.keys())
        # get num active tracks
        num_active = len(ids)

        # get inactive tracks with inactive < patience
        curr_it = {k: track for k, track in self.inactive_tracks.items()
                   if track.inactive_count <= self.inact_patience}
        # get inactive track proxies
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

            if len(self.tracks) > 0:
                y = torch.cat([y, y_inactive])
            else:
                y = y_inactive

            ids += [k for k in curr_it.keys()]

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

        # get distance between proxy features and detection features
        dist = self.dist(x, y)

        # solve hungarian
        dist, row, col = self.solve_hungarian(
            dist, num_active, detections, curr_it, sep)

        return dist, row, col, ids

    def assign(self, detections, dist, row, col, ids, sep=False):
        """
        Filter hungarian assignments using matching thresholds
        either assigning active and inactive together or separately
        """
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
                unconfirmed = len(
                    self.tracks[k]) >= 2 if self.tracker_cfg['remove_unconfirmed'] else True
                if unconfirmed:
                    self.inactive_tracks[k] = self.tracks[k]
                    self.inactive_tracks[k].inactive_count = 0
                del self.tracks[k]

        # increase inactive count by one
        for k in self.inactive_tracks.keys():
            self.inactive_tracks[k].inactive_count += self.frame_id - \
                self.prev_frame

        # start new track with unassigned detections if conf > thresh
        for i in range(len(detections)):
            if i not in assigned and detections[i]['conf'] > self.tracker_cfg['new_track_conf']:
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
        """
        Assign active and inactive at the same time
        """
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
        """
        Assign active and inactive one after another
        """
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
