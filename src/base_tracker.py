
from collections import defaultdict
import torch.nn as nn

import torch
# from tracking_wo_bnw.src.tracktor.utils import interpolate
import os
import numpy as np
import os.path as osp
import csv
import logging
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from src.tracking_utils import get_center, get_height, get_width,\
    make_pos, warp_pos, bbox_overlaps, is_moving, frame_rate
import cv2
import copy
from src.kalman import KalmanFilter


logger = logging.getLogger('AllReIDTracker.BaseTracker')


class BaseTracker():
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

        self.kalman = tracker_cfg['kalman']
        if self.kalman:
            self.shared_kalman = KalmanFilter()
            self.kalman_filter = KalmanFilter()
        else:
            self.kalman_filter = None

        self.log = True
        self.train_cfg = train_cfg
        self.device = device
        self.round1_float = lambda x: round(10 * x) / 10

        self.net_type = net_type
        self.encoder = encoder

        self.tracker_cfg = tracker_cfg
        self.motion_model_cfg = tracker_cfg['motion_config']
        self.output = output if not tracker_cfg['use_bism'] else 'norm'

        self.inact_thresh = tracker_cfg['inact_thresh']
        self.act_reid_thresh = tracker_cfg['act_reid_thresh']
        self.inact_reid_thresh = tracker_cfg['inact_reid_thresh']
        self.output_dir = tracker_cfg['output_dir']
        self.nan_first = tracker_cfg['nan_first']
        self.scale_thresh_ioa = tracker_cfg['scale_thresh_ioa']

        self.warp_mode_dct = {
            'Translation': cv2.MOTION_TRANSLATION,
            'Affine': cv2.MOTION_AFFINE,
            'Euclidean': cv2.MOTION_EUCLIDEAN,
            'Homography': cv2.MOTION_HOMOGRAPHY
        }

        self.data = data

        self.get_name(weight)
        logger.info("Executing experiment {}...".format(self.experiment))

        os.makedirs(osp.join(self.output_dir, self.experiment), exist_ok=True)
        if self.tracker_cfg['eval_bb']:
            self.encoder.eval()

        self.store_visualization = self.tracker_cfg['visualize']
        if self.store_visualization:
            self.init_vis()

        self.store_dist = self.tracker_cfg['store_dist']
        if self.store_dist:
            self.distance_ = defaultdict(dict)
        
        self.store_feats = self.tracker_cfg['store_feats']
        if self.store_feats:
            self.features_ = defaultdict(dict)

    def dist(self, x, y):
        if self.tracker_cfg['distance'] == 'cosine':
            return 1 - F.cosine_similarity(x[:, :, None], y.t()[None, :, :])
        else:
            return F.pairwise_distance(x[:, :, None], y.t()[None, :, :])

    def strfrac2float(self, x):
        if type(x) == int or type(x) == float:
            return x
        return float(x.split('/')[0]) / float(x.split('/')[-1])

    def get_features(self, frame):
        # forward pass
        with torch.no_grad():
            if self.net_type == 'resnet50_analysis':
                feats = self.encoder(frame)
            else:
                _, feats = self.encoder(frame, output_option=self.output)

        return feats

    def make_results(self):
        results = defaultdict(dict)
        for i, ts in self.tracks.items():
            for im_index, bbox, label in zip(ts.past_im_indices, ts.bbox, ts.label):
                results[i][im_index] = bbox.tolist() + [label]
        return results

    def _remove_short_tracks(self, all_tracks):
        tracks_new = dict()
        for k, v in all_tracks.items():
            if len(v) > self.tracker_cfg['length_thresh']:
                tracks_new[k] = v
        return tracks_new

    def write_results(self, output_dir, seq_name):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        all_tracks = self.make_results()

        if self.tracker_cfg['length_thresh']:
            all_tracks = self._remove_short_tracks(all_tracks)

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        output_dir = os.path.join(output_dir, self.experiment)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frames = list()
        with open(osp.join(output_dir, seq_name), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    frames.append(list)
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame,
                         i + 1,
                         x1 + 1,
                         y1 + 1,
                         x2 - x1,
                         y2 - y1,
                         -1, -1, bb[4], -1])
        print(set(frames))

    def get_name(self, weight):
        inact_avg = self.tracker_cfg['avg_inact']['proxy'] + str(
            self.tracker_cfg['avg_inact']['num']) if self.tracker_cfg['avg_inact']['do'] else 'last_frame'
        act_avg = self.tracker_cfg['avg_act']['proxy'] + str(
            self.tracker_cfg['avg_act']['num']) if self.tracker_cfg['avg_act']['do'] else 'last_frame'
        self.experiment = inact_avg + ':' + str(self.tracker_cfg['inact_reid_thresh']) + \
            ':' + act_avg + ':' + str(self.tracker_cfg['act_reid_thresh'])
        
        if self.tracker_cfg['random_patches_several_frames']:
            self.experiment += 'random_patches_several_frames:1'
        if self.tracker_cfg['several_frames']:
            self.experiment += 'several_frames:1'
        if self.tracker_cfg['random_patches']:
            self.experiment += 'random_patches:1'
        if self.tracker_cfg['random_patches_first']:
            self.experiment += 'random_patches_first:1'
        if self.tracker_cfg['running_mean_seq']:
            self.experiment += 'running_mean_seq:1'
        if self.tracker_cfg['first_batch']:
            self.experiment += 'first_batch:1'
        if self.tracker_cfg['running_mean_seq_reset']:
            self.experiment += 'running_mean_seq_reset:1'
        if self.tracker_cfg['first_batch_reset']:
            self.experiment += 'first_batch_reset:1'

        if self.tracker_cfg['use_bism']:
            self.experiment += 'bism:1'

        if self.tracker_cfg['nan_first']:
            self.experiment += 'nanfirst:1'

        if self.motion_model_cfg['motion_compensation']:
            self.experiment += 'MCOM:1' # Only Moving

        if self.motion_model_cfg['apply_motion_model']:
            self.experiment += 'MM:1'
            if self.tracker_cfg['kalman']:
                self.experiment += 'Kalman'
            else:
                self.experiment += str(self.motion_model_cfg['ioa_threshold'])

                if str(self.motion_model_cfg['ioa_threshold']) == 'learned':
                    train = '_'.join([str(v) for v in self.train_cfg.values()])
                    self.experiment += train
                else:
                    self.experiment += str(self.round1_float(self.strfrac2float(self.motion_model_cfg['lambda_mot'])))
                    self.experiment += str(self.round1_float(self.strfrac2float(self.motion_model_cfg['lambda_temp'])))
                    self.experiment += str(self.round1_float(self.strfrac2float(self.motion_model_cfg['lambda_occ'])))

        self.experiment = '_'.join([self.data[:-4],
                                    weight,
                                    'evalBB:' + str(self.tracker_cfg['eval_bb']),
                                    self.experiment])

        self.experiment += 'InactPat:' + str(self.tracker_cfg['inact_thresh'])
        self.experiment += 'ConfThresh:' + str(self.tracker_cfg['thresh'])

        if self.log:
            logger.info(self.experiment)

    def normalization_experiments(self, random_patches, frame, i):
        '''

        Normalization experiments:
         * Reset bn stats and use random patches to update bn stats
         * Reset bn stats and use random patches of the first frame to update bn stats
         * Use running mean of bbs as bn stats
         * Reset bn stats and use running mean of bbs as bn stats
         * Use bbs of the first batch
         * Reset bn stats and use bbs of the first batch

        '''
        # use random patches
        if self.tracker_cfg['random_patches'] or self.tracker_cfg['random_patches_first']:
            if i == 0:
                which_frame = 'each frame' if self.tracker_cfg['random_patches'] else 'first frame'
                if self.log:
                    logger.info(
                        "Using random patches of {}...".format(which_frame))

            if not self.tracker_cfg['random_patches_first'] or i == 0:
                self.encoder.train()
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1  # means no running mean
                with torch.no_grad():
                    _, _ = self.encoder(
                        random_patches, output_option=self.output)
                self.encoder.eval()

        elif self.tracker_cfg['running_mean_seq'] or self.tracker_cfg['running_mean_seq_reset']:
            if i == 0 and self.log:
                logger.info("Using moving average of seq...")

            self.encoder.train()
            if self.tracker_cfg['running_mean_seq_reset'] and i == 0:
                if self.log:
                    logger.info(
                        "Resetting BatchNorm statistics and use first batch as initial mean/std...")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1
                        m.first_batch_mean = True
            elif self.tracker_cfg['running_mean_seq_reset'] and i == 1:
                if self.log:
                    logger.info("Setting mometum to 0.1 again...")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.momentum = 0.1
            with torch.no_grad():
                _, _ = self.encoder(frame, output_option=self.output)
            self.encoder.eval()

        elif i == 0 and (self.tracker_cfg['first_batch'] or self.tracker_cfg['first_batch_reset']):
            if self.log:
                logger.info("Runing first batch in train mode...")

            self.encoder.train()
            if self.tracker_cfg['first_batch_reset']:
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1  # means no running mean, i.e., uses first batch wo initialization
                if self.log:
                    logger.info("Resetting BatchNorm statistics...")
            with torch.no_grad():
                _, _ = self.encoder(frame, output_option=self.output)
            self.encoder.eval()

    def normalization_before(self, seq, k=5, first=False):
        '''
        Run the first k frames in train mode
        '''
        if first:
            # logger.info('NOT resetting BatchNorm statistics...')
            if self.log:
                logger.info('Resetting BatchNorm statistics...')
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()

        if self.tracker_cfg['random_patches_several_frames'] or self.tracker_cfg['several_frames']:
            what_input = 'bounding boxes' if self.tracker_cfg['several_frames'] else 'random patches'
            if self.log:
                logger.info(
                    "Using {} of the first {} frames to initialize BatchNorm statistics...".format(
                        k, what_input))
            self.encoder.train()

            for i, (frame, _, _, _, _, _, _, random_patches,
                    _, _, _, _) in enumerate(seq):
                if what_input == 'random patches':
                    inp = random_patches
                else:
                    inp = frame
                with torch.no_grad():
                    _, _ = self.encoder(inp, output_option=self.output)
                if i >= k:
                    break
            self.encoder.eval()
            seq.random_patches = False

    def setup_seq(self, seq=None, first=False, seq_name=None):
        if seq_name is None and "MOT" in seq:
            if len(seq.name.split('-')) > 2:
                self.seq = f"Sequence{int(seq.name.split('-')[1])}_" + \
                    seq.name.split('-')[2]
            else:
                self.seq = f"Sequence{int(seq.name.split('-')[1])}"
            seq_name = seq.name
        elif seq_name is None:
            self.seq = seq.name
            seq_name = seq.name
        else:
            self.seq = seq_name

        self.seq_name = seq.name

        # add distance dict for seq
        if self.store_dist:
            self.init_dist()
        if self.store_feats:
            self.init_feats()

        self.thresh_every = True if self.act_reid_thresh == "every" else False
        self.thresh_tbd = True if self.act_reid_thresh == "tbd" else False

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        self.mv_avg = dict()
        self.id = 0

        # set backbone into evaluation mode
        if self.tracker_cfg['eval_bb'] and not first:
            self.encoder.eval()
        elif first and self.log:
            logger.info('Feeding sequence data before tracking once')

        # if random patches should be sampled
        if seq is not None:
            seq.random_patches = self.tracker_cfg['random_patches'] or self.tracker_cfg[
                'random_patches_first'] or self.tracker_cfg['random_patches_several_frames']
        self.is_moving = is_moving(seq_name)
        self.frame_rate = frame_rate(seq_name)
        if self.log:
            logger.info("Frame rate: {}".format(self.frame_rate))

    def reset_threshs(self):
        self.act_reid_thresh = 'every' if self.thresh_every else self.act_reid_thresh
        self.inact_reid_thresh = 'every' if self.thresh_every else self.inact_reid_thresh
        self.act_reid_thresh = 'tbd' if self.thresh_tbd else self.act_reid_thresh
        self.inact_reid_thresh = 'tbd' if self.thresh_tbd else self.inact_reid_thresh

    def update_thresholds(self, dist, num_active, num_inactive):
        # update active threshold
        # self.act_reid_thresh == 'tbd' only in frame 1
        if (self.act_reid_thresh == 'tbd' or self.thresh_every) and num_active > 0:
            act = np.atleast_2d(
                np.array(
                    [1] *
                    num_active +
                    [0] *
                    num_inactive)) == np.atleast_2d(
                np.ones(
                    dist.shape[0])).T

            if self.thresh_every:
                self.act_reid_thresh = np.mean(
                    dist[act]) - 0 * np.std(dist[act])
            elif self.thresh_tbd:
                self.act_reid_thresh = np.mean(
                    dist[act]) - 0.5 * np.std(dist[act])

        # update inactive threshold
        if (self.inact_reid_thresh ==
                'tbd' or self.thresh_every) and num_inactive > 0:
            act = np.atleast_2d(
                np.array(
                    [1] *
                    num_active +
                    [0] *
                    num_inactive)) == np.atleast_2d(
                np.ones(
                    dist.shape[0])).T

            if self.thresh_every:
                self.inact_reid_thresh = np.mean(
                    dist[~act]) - 2 * np.std(dist[~act])
            elif self.thresh_tbd:
                self.inact_reid_thresh = np.mean(
                    dist[~act]) - 1 * np.std(dist[~act])

    def visualize(self, detections, tr_ids, path, seq, frame):
        if frame == 1:
            os.makedirs(
                osp.join('visualizations', self.experiment, seq),
                exist_ok=True)
        img = matplotlib.image.imread(path)
        figure, ax = plt.subplots(1)
        figure.set_size_inches(img.shape[1] / 100, img.shape[0] / 100)
        for tr_id, d in zip(tr_ids, detections):
            if tr_id not in self.id_to_col.keys():
                self.id_to_col[tr_id] = self.color_list[self.col]
                self.col += 1
                
            # add rectangle
            rect = matplotlib.patches.Rectangle(
                (d['bbox'][0], d['bbox'][1]),
                d['bbox'][2] - d['bbox'][0],
                d['bbox'][3] - d['bbox'][1],
                edgecolor=self.id_to_col[tr_id],
                facecolor="none",
                linewidth=2)
            ax.add_patch(rect)

            # add text with id, ioa and visibility
            text = ', '.join(
                [str(tr_id), str(d['gt_id'])])
            plt.text(
                d['bbox'][0] - 15,
                d['bbox'][1] + (d['bbox'][3] - d['bbox'][1]) / 2,
                text,
                va='center',
                rotation='vertical',
                c=self.id_to_col[tr_id],
                fontsize=8)

        ax.imshow(img)
        plt.axis('off')
        plt.savefig(
            osp.join(
                'visualizations',
                self.experiment,
                seq,
                f"{frame:06d}.jpg"),
            bbox_inches='tight',
            pad_inches=0,
            dpi=100)
        plt.close()

    def init_vis(self):
        colors = mcolors.CSS4_COLORS
        color_list = list(colors.keys())
        self.color_list = 10 * color_list
        random.shuffle(color_list)
        self.id_to_col = dict()
        self.col = 0
        self.round2 = lambda x: str(round(100 * x) / 100)

    def init_dist(self):
        self.distance_[self.seq]['inact_dist_same'] = list()
        self.distance_[self.seq]['act_dist_same'] = list()
        self.distance_[self.seq]['inact_dist_diff'] = list()
        self.distance_[self.seq]['act_dist_diff'] = list()
        self.distance_[self.seq]['active_inactive'] = list()
        self.distance_[self.seq]['same_class_mat'] = list()
        self.distance_[self.seq]['dist'] = list()

    def init_feats(self):
        self.features_[self.seq] = dict()
        
    @staticmethod
    def plot_single_image(imgs, name):
        import matplotlib.pyplot as plt

        for i, img in enumerate(imgs):
            img = img.permute(1, 2, 0)
            figure, ax = plt.subplots(1)
            figure.set_size_inches(img.shape[1] / 100, img.shape[0] / 100)
            ax.imshow(np.asarray(img))
            plt.savefig(str(i) + + name + '.png', dpi=100)
            plt.close()

    def motion_compensation(self, whole_image, im_index):
        # adapted from tracktor
        self.warp_matrix_nomr = None
        if im_index > 0:
            # im1 = reference frame, im2 = frame to convert
            # changed this from tracktor --> current img = reference
            im1 = np.transpose(whole_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.motion_model_cfg['num_iter_mc'],
                self.motion_model_cfg['termination_eps_mc'])
            _, warp_matrix = cv2.findTransformECC(
                im1_gray, im2_gray, warp_matrix, self.warp_mode_dct[
                    self.motion_model_cfg['warp_mode']], criteria, None, 15)
            warp_matrix = torch.from_numpy(warp_matrix)
            self.warp_matrix_nomr = np.linalg.norm(warp_matrix[:, -1])
            if self.is_moving and self.motion_model_cfg['motion_compensation']:
                for tracks in [self.tracks, self.inactive_tracks]:
                    for k, track in tracks.items():
                        for i, pos in enumerate(track.last_pos):
                            if isinstance(pos, torch.Tensor):
                                pos = pos.cpu().numpy()
                            pos = np.squeeze(warp_pos(np.atleast_2d(pos), warp_matrix))
                            track.last_pos[i] = pos
                            # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

        self.last_image = whole_image

    def motion_step(self, track):
        # adapted from tracktor; no need for dt as it is always one frame and
        # velocity is per frame and it is updated every frame
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(
                *center_new,
                get_width(track.pos),
                get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self, approximate=True):
        # adapted from tracktor
        """Applies a simple linear motion model that considers the last n_steps steps."""
        height = list()
        for track in self.tracks.values():
            if len(track.last_pos) > 1:
                last_pos = np.asarray(track.last_pos)
                # avg velocity between each pair of consecutive positions in
                # t.last_pos
                if self.motion_model_cfg['center_only']:
                    if approximate:
                        vs_c = [get_center(p2) - get_center(p1)
                                for p1, p2 in zip(last_pos, last_pos[1:])]
                    else:
                        frames = np.asarray(track.past_frames)
                        dt = [p2 - p1 for p1, p2 in zip(frames, frames[1:])]
                        vs_c = [(get_center(p2) - get_center(p1))/t
                                    for p1, p2, t in zip(last_pos, last_pos[1:], dt)]
                else:
                    height.append((last_pos[:, 3]-last_pos[:, 1]).mean())
                    if approximate:
                        vs_c = [get_center(p2) - get_center(p1)
                                for p1, p2 in zip(last_pos, last_pos[1:])]
                        vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]
                    else:
                        frames = np.asarray(track.past_frames)
                        dt = [f2 - f1 for f1, f2 in zip(frames, frames[1:])]
                        vs = [(p2 - p1)/t for p1, p2, t in zip(last_pos, last_pos[1:], dt)]
                        vs_c = [(get_center(p2) - get_center(p1))/t
                                    for p1, p2, t in zip(last_pos, last_pos[1:], dt)]

                track.update_v(np.stack(vs).mean(axis=0))
                track.last_vc = np.stack(vs_c).mean(axis=0)
                self.motion_step(track)                

        for track in self.inactive_tracks.values():
            if len(track.last_pos) > 1:
                self.motion_step(track)

    def get_motion_dist(self, detections, curr_it):
        act_pos = [track.pos for track in self.tracks.values()]
        inact_pos = [track.pos for track in curr_it.values()]
        pos = torch.from_numpy(np.asarray(act_pos + inact_pos))
        det_pos = torch.from_numpy(np.asarray(
            [t['bbox'] for t in detections]))
        iou = bbox_overlaps(det_pos, pos)
        iou = 1 - iou
        return iou

    def combine_motion_appearance(self, iou, dist):
        # init distances
        dist_emb = copy.deepcopy(dist)
        if type(iou) != np.ndarray:
            dist_iou = iou.cpu().numpy()
        else: 
            dist_iou = iou

        # weighted of threshold
        if self.motion_model_cfg['ioa_threshold'] == 'sum':
            dist = (dist_emb + dist_iou)*0.5

        elif self.motion_model_cfg['ioa_threshold'] == 'motion':
            dist = dist_iou

        elif self.motion_model_cfg['ioa_threshold'] == 'SeperateAssignment':
            dist = [dist_emb, dist_iou]

        return dist

    def add_feats_to_storage(self, detections):
        detections_to_save = copy.deepcopy(detections)
        for d in detections_to_save:
            d['bbox'] = d['bbox'].tolist()
            d['feats'] = d['feats'].tolist()
        self.features_[self.seq][self.frame_id] = detections_to_save

    def add_dist_to_storage(self, gt_n, gt_t, num_active, num_inactive, dist):
        # make at least 2d for mask computation
        gt_n = np.atleast_2d(np.array(gt_n))
        gt_t = np.atleast_2d(np.array(gt_t))
        
        # masks to remove unassigned bbs/tracks
        keep_rows = gt_n != -1
        keep_rows = keep_rows.squeeze()
        keep_cols = gt_t != -1
        keep_cols = keep_cols.squeeze()

        # generate same class matrix
        same = gt_t == gt_n.T

        # generate active matrix
        act = np.atleast_2d(
            np.array(
                [1] *
                num_active +
                [0] *
                num_inactive)) == np.atleast_2d(
            np.ones(
                dist.shape[0])).T
        
        # Filter out
        same = same[keep_rows, :]
        act = act[keep_rows, :]
        dist = dist[keep_rows, :]

        # if there are rows left
        if keep_rows.tolist():
            # remove unnessecary 3rd dim
            act = act[0] if len(act.shape) == 3 else act
            dist = dist[0] if len(dist.shape) == 3 else dist
            same = same[0] if len(same.shape) == 3 else same

            # remove tracks with unassigned class
            act = act[:, keep_cols]
            dist = dist[:, keep_cols]
            same = same[:, keep_cols]

        self.distance_[self.seq]['inact_dist_same'].extend(
            dist[same & ~act].tolist())
        self.distance_[self.seq]['act_dist_same'].extend(
            dist[same & act].tolist())
        self.distance_[self.seq]['inact_dist_diff'].extend(
            dist[~same & ~act].tolist())
        self.distance_[self.seq]['act_dist_diff'].extend(
            dist[~same & act].tolist())
        self.distance_[self.seq]['active_inactive'].append(act.tolist())
        self.distance_[self.seq]['same_class_mat'].append(same.tolist())
        self.distance_[self.seq]['dist'].append(dist.tolist())


