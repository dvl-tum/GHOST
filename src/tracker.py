from collections import defaultdict
import copy
from re import L
from matplotlib.pyplot import imsave

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
from src.tracking_utils import bisoftmax, get_proxy, add_ioa  # , get_reid_performance
from src.base_tracker import BaseTracker
import torchvision.transforms as T
import torch.nn.functional as F


logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker(BaseTracker):
    def __init__(
            self,
            tracker_cfg,
            encoder,
            net_type='resnet50',
            output='plain',
            weight='No',
            data='tracktor_preprocessed_files.txt'):
        super(
            Tracker,
            self).__init__(
            tracker_cfg,
            encoder,
            net_type,
            output,
            weight,
            data)

    def track(self, seq, first=False):
        '''
        first - feed all bounding boxes through net first for bn stats update
        seq -   sequence instance for iteratration and with meta information
                like name or lentgth
        '''
        logger.info(
            "Tracking sequence {} of lenght {}".format(
                seq.name, seq.num_frames))
        self.setup_seq(seq, first)

        # batch norm experiemnts I
        self.normalization_before(seq, first)

        # iterate over frames
        for i, (frame, _, path, boxes, _, gt_ids, vis,
                random_patches, whole_im, frame_size) in enumerate(seq):

            # batch norm experiments II
            self.normalization_experiments(random_patches, frame, i)

            self.frame_id = int(path.split(os.sep)[-1][:-4])
            print(i+1, self.frame_id, len(self.tracks), len(self.inactive_tracks))

            if self.debug:
                self.event_dict[self.seq][self.frame_id] = list()
            tracks = list()

            # forward pass
            with torch.no_grad():
                if self.net_type == 'resnet50_analysis':
                    feats = self.encoder(frame)
                else:
                    _, feats = self.encoder(frame, output_option=self.output)

            # add features of whole blurred image
            if self.tracker_cfg['use_blur']:
                blurred_feats = self.get_blurred_feats(whole_im, boxes)
                feats = torch.cat([feats, blurred_feats], dim=1)
                # feats = feats + blurred_feats

            # just feeding for bn stats update
            if first:
                continue

            # iterate over bbs in current frame
            for f, b, gt_id, v in zip(feats, boxes, gt_ids, vis):
                if (b[3] - b[1]) / (b[2] - b[0]
                                    ) < self.tracker_cfg['h_w_thresh']:
                    track = {
                        'pos': b,
                        'last_pos': list(),
                        'last_v': None,
                        'bbox': b,
                        'feats': f,
                        'im_index': self.frame_id,
                        'id': gt_id,
                        'vis': v}
                    tracks.append(track)

                    if self.store_dist:
                        v = 0.999 if v == 1.0 else v
                        v = floor(v * 10) / 10
                        self.distance_[
                            self.seq]['visibility_count'][v] += 1

                    if self.save_embeddings_by_id:
                        _f = f.cpu().numpy().tolist()
                        self.embeddings_by_id[seq.name][gt_id].append([i, _f])

            # apply motion compensation to stored track positions
            if self.motion_model_cfg['motion_compensation']:
                self.motion_compensation(whole_im, i)

            # add intersection over area to each bb
            self.curr_interaction, self.curr_occlusion = add_ioa(
                tracks, self.seq, self.interaction, self.occlusion, frame_size)

            if self.store_visualization:
                self.visualize(tracks, path, seq.name, i+1)

            # association over frames
            self._track(tracks, i, frame=frame)

        # just fed for bn stats update
        if first:
            logger.info('Done with pre-tracking feed...')
            return

        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)

        # compute reid performance
        if self.tracker_cfg['get_reid_performance']:
            cmc, aps = get_reid_performance()
            logger.info(
                "r-1 {}, r-5 {}, r-8 {}".format(cmc[0], cmc[5], cmc[7]))
            logger.info("mAP {}".format(np.mean(aps)))

        # write results
        self.write_results(self.output_dir, seq.name)

        # reset thresholds if every / tbd
        self.reset_threshs()

        # store dist to json file
        if self.store_dist:
            logger.info(
                "Storing distance information of {} to {}".format(
                    seq.name,
                    self.experiment +
                    'distances.json'))
            with open(self.experiment + 'distances.json', 'w') as jf:
                json.dump(self.distance_, jf)

        # print errors and store error events
        if self.debug:
            if '13' in seq.name:
                logger.info(self.errors)
            with open(self.experiment + 'event_dict.json', 'w') as jf:
                json.dump(self.event_dict, jf)

        # save embeddings by id for further investigation
        if self.save_embeddings_by_id:
            with open(self.experiment + 'embeddings_by_id.json', 'w') as jf:
                json.dump(self.embeddings_by_id, jf)

    def get_blurred_feats(self, whole_im, boxes):
        blurrer = T.GaussianBlur(kernel_size=(29, 29), sigma=5)
        self.encoder.eval()
        ims = list()
        for box in boxes:
            blurred = copy.deepcopy(whole_im)
            blurred = blurrer(blurred)
            blurred[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
                whole_im[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            blurred = F.interpolate(blurred.unsqueeze(dim=0), scale_factor=(1/8, 1/8))
            ims.append(blurred)

        ims = torch.stack(ims).squeeze()
        if len(ims.shape) < 4:
            ims = ims.unsqueeze(0)
        with torch.no_grad():
            if self.net_type == 'resnet50_analysis':
                blurred_feats = self.encoder(ims)
            else:
                _, blurred_feats = self.encoder(ims, output_option=self.output)

        self.encoder.train()

        return blurred_feats

    def _track(self, tracks, i, frame=None):
        # just add all bbs to self.tracks / intitialize in the first frame
        if i == 0:
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.tracks[self.id][-1]['last_pos'].append(self.tracks[self.id][-1]['pos']) 
                self.id += 1

        # association over frames for frame > 0
        elif i > 0:
            # get hungarian matching
            if not self.tracker_cfg['avg_inact']['proxy'] == 'each_sample':
                dist, row, col, ids = self.get_hungarian_with_proxy(
                    tracks, sep=self.tracker_cfg['assign_separately'])
            else:
                dist, row, col, ids = self.get_hungarian_each_sample(tracks)

            if dist is not None:
                # get bb assignment
                self.assign(
                    tracks=tracks,
                    dist=dist,
                    row=row,
                    col=col,
                    ids=ids,
                    sep=self.tracker_cfg['assign_separately'])

    def get_hungarian_each_sample(self, tracks):
        # get new detections
        x = torch.stack([t['feats'] for t in tracks])
        gt_n = [v['id'] for v in tracks]

        # get distances to active tracks
        gt_t, dist_all, ids = list(), list(), list()
        gt_t += [v[-1]['id'] for v in self.tracks.values()]

        # if use each sample for active frames
        if not self.tracker_cfg['avg_act']['do'] and len(tracks) > 0:
            y = torch.stack([t[-1]['feats'] for t in self.tracks.values()])
            ids.extend([i for i in self.tracks.keys()])
            dist = sklearn.metrics.pairwise_distances(
                x.cpu().numpy(), y.cpu().numpy(), metric='cosine').T
            dist_all.extend([d for d in dist])
        else:
            for id, tr in self.tracks.items():
                y = torch.stack([t['feats'] for t in tr])
                ids.append(id)
                dist = sklearn.metrics.pairwise_distances(
                    x.cpu().numpy(), y.cpu().numpy(), metric='cosine')

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
        curr_it = {k: v for k, v in self.inactive_tracks.items()
                   if v[-1]['inact_count'] <= self.inact_thresh}
        if len(curr_it) > 0:
            gt_t += [v[-1]['id'] for v in curr_it.values()]
            for id, tr in curr_it.items():
                y = torch.stack([t['feats'] for t in tr])
                ids.append(id)
                dist = sklearn.metrics.pairwise_distances(
                    x.cpu().numpy(), y.cpu().numpy(), metric='cosine')
                if self.tracker_cfg['avg_inact']['num'] == 1:
                    dist_all.append(np.min(dist, axis=1))
                elif self.tracker_cfg['avg_inact']['num'] == 2:
                    dist_all.append(np.mean(dist, axis=1))
                elif self.tracker_cfg['avg_inact']['num'] == 3:
                    dist_all.append(np.max(dist, axis=1))
                elif self.tracker_cfg['avg_inact']['num'] == 4:
                    dist_all.append(
                        (np.max(dist, axis=1) + np.min(dist, axis=1)) / 2)
        num_inactive = len([k for k, v in curr_it.items()])

        # update thresholds
        self.update_thresholds(dist, num_inactive, num_inactive)

        # solve assignment problem
        dist = np.vstack(dist_all).T

        if self.nan_first:
            dist[:, :num_active] = np.where(dist[:, :num_active] <=
                self.act_reid_thresh, dist[:, :num_active], np.nan)
            dist[:, num_active:] = np.where(dist[:, num_active:] <=
                self.inact_reid_thresh, dist[:, num_active:], np.nan)

        if self.motion_model_cfg['apply_motion_model']:
            self.motion()
            iou = self.get_motion_dist(tracks)
            dist = self.combine_motion_appearance(iou, dist, tracks)

        row, col = solve_dense(dist)

        # store distances
        if self.store_dist:
            self.add_dist_to_storage(
                gt_n, gt_t, num_active, num_inactive, dist)

        return dist, row, col, ids

    def get_hungarian_with_proxy(self, tracks, sep=False):
        # instantiate
        ids, gt_t = list(), list()
        y_inactive, y = None, None

        x = torch.stack([t['feats'] for t in tracks])
        gt_n = [v['id'] for v in tracks]

        # Get active tracklets
        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = get_proxy(
                    curr_it=self.tracks,
                    mode='act',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])

            ids += list(self.tracks.keys())
            gt_t += [v[-1]['id'] for v in self.tracks.values()]
            num_active = len(ids)

        # get inactive tracklets (inacht thresh = 100000)
        curr_it = {k: v for k, v in self.inactive_tracks.items()
                   if v[-1]['inact_count'] <= self.inact_thresh}
        # if there are inactive tracks that fall into inact_thresh
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = get_proxy(
                    curr_it=curr_it,
                    mode='inact',
                    tracker_cfg=self.tracker_cfg,
                    mv_avg=self.mv_avg)
            else:
                y_inactive = torch.stack([v[-1]['feats']
                                         for v in curr_it.values()])

            if len(self.tracks) > 0 and not sep:
                y = torch.cat([y, y_inactive])
            elif not sep:
                y = y_inactive
                num_active = 0

            ids += [k for k, v in curr_it.items()]
            gt_t += [v[-1]['id'] for v in curr_it.values()]
            num_inactive = len([k for k, v in curr_it.items()])

        # if no active or inactive tracks --> return and instantiate all dets
        # new
        elif len(curr_it) == 0 and len(self.tracks) == 0:
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
            return None, None, None, None
        # if there are active but no inactive
        else:
            num_inactive = 0

        # compute distance
        if not sep:
            if not self.tracker_cfg['use_bism']:
                dist = sklearn.metrics.pairwise_distances(
                    x.cpu().numpy(), y.cpu().numpy(), metric='cosine')
            else:
                dist = 1 - bisoftmax(x.cpu(), y.cpu())

            if self.store_dist:
                self.add_dist_to_storage(
                    gt_n, gt_t, num_active, num_inactive, dist)

            # update thresholds
            self.update_thresholds(dist, num_active, num_inactive)

            if self.nan_first:
                dist[:, :num_active] = np.where(dist[:, :num_active] <=
                    self.act_reid_thresh, dist[:, :num_active], np.nan)
                dist[:, num_active:] = np.where(dist[:, num_active:] <=
                    self.act_reid_thresh, dist[:, num_active:], np.nan)

            # row represent current frame
            # col represents last frame + inactiva tracks
            # row, col = scipy.optimize.linear_sum_assignment(dist)
            row, col = solve_dense(dist)
        else:
            dist_act = sklearn.metrics.pairwise_distances(
                x.cpu().numpy(), y.cpu().numpy(), metric='cosine')
            row, col = solve_dense(dist_act)
            if y_inactive is not None:
                dist_inact = sklearn.metrics.pairwise_distances(
                    x.cpu().numpy(),
                    y_inactive.cpu().numpy(),
                    metric='cosine')  # 'euclidean')#'cosine')
            else:
                dist_inact = None
            dist = [dist_act, dist_inact]

        return dist, row, col, ids

    def assign(self, tracks, dist, row, col, ids, sep=False):
        # assign tracks from hungarian
        active_tracks = list()

        if not sep:
            assigned = self.assign_act_inact_same_time(
                row, col, dist, tracks, active_tracks, ids)
        else:
            assigned = self.assign_separatly(
                row, col, dist, tracks, active_tracks, ids)

        # move tracks not used to inactive tracks
        keys = list(self.tracks.keys())
        for k in keys:
            if k not in active_tracks:
                self.inactive_tracks[k] = self.tracks[k]
                del self.tracks[k]
                for i in range(len(self.inactive_tracks[k])):
                    self.inactive_tracks[k][i]['inact_count'] = 0

        # increase inactive count by one
        for k in self.inactive_tracks.keys():
            for i in range(len(self.inactive_tracks[k])):
                self.inactive_tracks[k][i]['inact_count'] += 1

        # tracks that have not been assigned by hungarian
        if self.debug:
            gt_available = [t[-1]['id'] for t in self.tracks.values()]
            gt_available_k = [k for k in self.tracks.keys()]
            gt_available_b = [t[-1]['id'] for t in self.inactive_tracks.values()]
            gt_available_bk = [k for k in self.inactive_tracks.keys()]

        for i in range(len(tracks)):
            if i not in assigned:
                if self.debug:
                    ioa = self.round1(tracks[i]['ioa'])
                    vis = self.round1(tracks[i]['vis'])
                    _det = [self.frame_id, vis, ioa, tracks[i]['id']]

                    if tracks[i]['id'] in gt_available:
                        self.errors['unassigned_act_ioa_' + ioa] += 1
                        self.errors['unassigned_act_vis_' + vis] += 1

                        ind = gt_available.index(tracks[i]['id'])
                        tr_id = gt_available_k[ind]
                        ioa_gt = self.tracks[tr_id][-1]['ioa']
                        vis_gt = self.tracks[tr_id][-1]['vis']
                        frame_gt = self.tracks[tr_id][-1]['im_index']
                        id_gt = self.tracks[tr_id][-1]['id']
                        _gt = [frame_gt, vis_gt, ioa_gt, id_gt]

                        event = ['Unassigned', 'Act'] + _gt + _det
                        self.event_dict[self.seq][self.frame_id].append(event)

                    if tracks[i]['id'] in gt_available_b:
                        self.errors['unassigned_inact_ioa_' + ioa] += 1
                        self.errors['unassigned_inact_vis_' + vis] += 1

                        ind = gt_available_b.index(tracks[i]['id'])
                        tr_id = gt_available_bk[ind]
                        ioa_gt = self.inactive_tracks[tr_id][-1]['ioa']
                        vis_gt = self.inactive_tracks[tr_id][-1]['vis']
                        frame_gt = self.inactive_tracks[tr_id][-1]['im_index']
                        id_gt = self.inactive_tracks[tr_id][-1]['id']
                        _gt = [frame_gt, vis_gt, ioa_gt, id_gt]

                        event = ['Unassigned', 'Inact'] + _gt + _det
                        self.event_dict[self.seq][self.frame_id].append(event)

                self.tracks[self.id].append(tracks[i])
                self.tracks[self.id][-1]['last_pos'].append(self.tracks[self.id][-1]['pos'])
                self.id += 1

    def assign_act_inact_same_time(
            self,
            row,
            col,
            dist,
            tracks,
            active_tracks,
            ids):
        # assigned contains all new detections that have been assigned
        assigned = list()
        act_thresh = 1000 if self.nan_first else self.act_reid_thresh
        inact_thresh = 1000 if self.nan_first else self.inact_reid_thresh

        for r, c in zip(row, col):
            # get reid threshold scale
            scale = max(0.4, (1-tracks[r]['ioa'])**(1/10)) if \
                self.scale_thresh_ioa else 1

            # get detection information if debug
            if self.debug:
                ioa = self.round1(tracks[r]['ioa'])
                vis = self.round1(tracks[r]['vis'])
                _det = [self.frame_id, vis, ioa, tracks[r]['id']]

            # assign tracks to active tracks if reid distance < thresh
            if ids[c] in self.tracks.keys() and \
               (dist[r, c] < act_thresh * scale or self.nan_first):

                # generate error event if debug
                if self.tracks[ids[c]][-1]['id'] != tracks[r]['id'] \
                  and self.debug:
                    self.errors['wrong_assigned_act_ioa_' + ioa] += 1
                    self.errors['wrong_assigned_act_vis_' + vis] += 1

                    ioa_gt = self.tracks[ids[c]][-1]['ioa']
                    vis_gt = self.tracks[ids[c]][-1]['vis']
                    frame_gt = self.tracks[ids[c]][-1]['im_index']
                    id_gt = self.tracks[ids[c]][-1]['id']
                    _gt = [frame_gt, vis_gt, ioa_gt, id_gt]

                    event = ['WrongAssignment', 'Act'] + _gt + _det
                    self.event_dict[self.seq][self.frame_id].append(event)

                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])
                assigned.append(r)

            # assign tracks to inactive tracks if reid distance < thresh
            elif ids[c] in self.inactive_tracks.keys() and \
               (dist[r, c] < inact_thresh * scale or self.nan_first):

                # generate error event if debug
                if self.inactive_tracks[ids[c]][-1]['id'] != tracks[r]['id'] \
                  and self.debug:
                    self.errors['wrong_assigned_inact_ioa_' + ioa] += 1
                    self.errors['wrong_assigned_inact_vis_' + vis] += 1

                    ioa_gt = self.inactive_tracks[ids[c]][-1]['ioa']
                    vis_gt = self.inactive_tracks[ids[c]][-1]['vis']
                    frame_gt = self.inactive_tracks[ids[c]][-1]['im_index']
                    id_gt = self.inactive_tracks[ids[c]][-1]['id']
                    _gt = [frame_gt, vis_gt, ioa_gt, id_gt]

                    event = ['WrongAssignment', 'Inct'] + _gt + _det
                    self.event_dict[self.seq][self.frame_id].append(event)

                # move inactive track to active
                self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                del self.inactive_tracks[ids[c]]
                for i in range(len(self.tracks[ids[c]])):
                    del self.tracks[ids[c]][i]['inact_count']

                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])
                assigned.append(r)

        return set(assigned)

    def assign_separatly(self, row, col, dist, tracks, active_tracks, ids):

        # assign active tracks first
        assigned = self.assign_act_inact_same_time(
            row, col, dist[0], tracks, active_tracks, ids[:dist[0].shape[1]])

        # assign inactive tracks
        if dist[1] is not None:
            # only use detections that have not been assigned yet
            unassigned = sorted(
                list(set(list(range(dist[0].shape[0]))) - assigned))

            if len(unassigned) != 0:
                dist[1] = dist[1][unassigned, :]

                row_inact, col_inact = solve_dense(dist[1])
                assigned_2 = self.assign_act_inact_same_time(
                    row=row_inact,
                    col=col_inact,
                    dist=dist[1],
                    tracks=[t for i, t in enumerate(tracks) if i in unassigned],
                    active_tracks=active_tracks,
                    ids=ids[dist[0].shape[1]:])
                assigned_2 = set(
                    [u for i, u in enumerate(unassigned) if i in assigned_2])
                assigned.update(assigned_2)

        return assigned
