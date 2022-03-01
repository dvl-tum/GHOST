
from collections import defaultdict
from posixpath import sep
from numpy.core.fromnumeric import shape

from numpy.lib.function_base import diff

import torch
#from tracking_wo_bnw.src.tracktor.utils import interpolate
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
    make_pos, warp_pos, bbox_overlaps, is_moving, frame_rate, WeightPredictor
import cv2
import sklearn
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
            weight_pred=None,
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
        self.hill = lambda x, y: x/(x+y)
        self.hill_vis = lambda x, y: x/(x+y)

        def sig(a, k=1, v_min=0, v_max=1, shift=0):
            res = 1/(1+np.exp(-(a+shift)*k))
            res = v_min + res * (v_max-v_min)
            return res
        self.sig = sig

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

        self.distance_ = defaultdict(dict)
        self.interaction = defaultdict(list)
        self.occlusion = defaultdict(list)

        self.store_dist = self.tracker_cfg['store_dist']

        self.store_visualization = self.tracker_cfg['visualize']
        if self.store_visualization:
            self.init_vis()

        self.debug = self.tracker_cfg['debug']
        if self.debug:
            self.init_debug()

        self.save_embeddings_by_id = self.tracker_cfg['save_embeddings_by_id']
        if self.save_embeddings_by_id:
            self.embeddings_by_id = dict()

        self.vel_dict = defaultdict(list)

        if self.motion_model_cfg['ioa_threshold'] == 'learned':
            self.weight_pred = weight_pred

        self.make_weight_pred_dataset = self.tracker_cfg['save_dataset']
        if self.make_weight_pred_dataset:
            self.weight_pred_dataset = dict()

        self.emc_dict = defaultdict(list)

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
            for im_index, bbox in zip(ts.past_im_indices, ts.bbox):
                results[i][im_index] = bbox
        return results

    def _remove_short_tracks(self, all_tracks):
        tracks_new = dict()
        for k, v in all_tracks.items():
            if len(v) > self.tracker_cfg['length_thresh']:
                tracks_new[k] = v
        if self.log:
            logger.info(
                "Removed {} short tracks".format(
                    len(all_tracks) -
                    len(tracks_new)))
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

        with open(osp.join(output_dir, seq_name), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
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
                         -1, -1, -1, -1])

    def get_name(self, weight):
        inact_avg = self.tracker_cfg['avg_inact']['proxy'] + str(
            self.tracker_cfg['avg_inact']['num']) if self.tracker_cfg['avg_inact']['do'] else 'last_frame'
        act_avg = self.tracker_cfg['avg_act']['proxy'] + str(
            self.tracker_cfg['avg_act']['num']) if self.tracker_cfg['avg_act']['do'] else 'last_frame'
        self.experiment = inact_avg + ':' + str(self.tracker_cfg['inact_reid_thresh']) + \
            ':' + act_avg + ':' + str(self.tracker_cfg['act_reid_thresh'])

        if self.tracker_cfg['use_bism']:
            self.experiment += 'bism:1'

        if self.tracker_cfg['nan_first']:
            self.experiment += 'nanfirst:1'

        if self.tracker_cfg['scale_thresh_ioa']:
            self.experiment += 'IOAscale:1'

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
            
        if self.tracker_cfg['active_proximity']:
            self.experiment += 'ActProx:1'

        self.experiment = '_'.join([self.data[:-4],
                                    weight,
                                    'evalBB:' + str(self.tracker_cfg['eval_bb']),
                                    self.experiment])
        
        self.experiment += 'InactPat:' + str(self.tracker_cfg['inact_thresh'])
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
                    _) in enumerate(seq):
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

    def inst_dist(self):
        self.distance_[self.seq]['inact_dist_same'] = list()
        self.distance_[self.seq]['act_dist_same'] = list()
        self.distance_[self.seq]['inact_dist_diff'] = list()
        self.distance_[self.seq]['act_dist_diff'] = list()
        self.distance_[self.seq]['interaction_mat'] = list()
        self.distance_[self.seq]['occlusion_mat'] = list()
        self.distance_[self.seq]['active_inactive'] = list()
        self.distance_[self.seq]['same_class_mat'] = list()
        self.distance_[self.seq]['dist'] = list()
        self.distance_[self.seq]['size'] = list()
        self.distance_[self.seq]['iou_dist'] = list()
        self.distance_[self.seq]['inactive_count'] = list()
        self.distance_[self.seq]['iou_dist_diff'] = {
            'same': defaultdict(list),
            'diff': defaultdict(list)}

        self.distance_[
            self.seq]['visibility_count'] = {
            -1: 0,
            0: 0,
            0.1: 0,
            0.2: 0,
            0.3: 0,
            0.4: 0,
            0.5: 0,
            0.6: 0,
            0.7: 0,
            0.8: 0,
            0.9: 0}
        
        self.distance_[self.seq]['prob_dict'] = dict()

    def setup_seq(self, seq=None, first=False, seq_name=None):
        if seq_name is None:
            if len(seq.name.split('-')) > 2:
                self.seq = f"Sequence{int(seq.name.split('-')[1])}_" + \
                    seq.name.split('-')[2]
            else:
                self.seq = f"Sequence{int(seq.name.split('-')[1])}"
            seq_name = seq.name
        else:
            self.seq = seq_name

        self.seq_name = seq.name

        self.thresh_every = True if self.act_reid_thresh == "every" else False
        self.thresh_tbd = True if self.act_reid_thresh == "tbd" else False

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        self.mv_avg = dict()
        self.id = 0
        if self.store_dist:
            self.inst_dist()

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

        if self.save_embeddings_by_id:
            self.embeddings_by_id[seq.name] = defaultdict(list)

        if self.make_weight_pred_dataset:
            self.weight_pred_dataset[seq.name] = defaultdict(dict)

    def reset_threshs(self):
        self.act_reid_thresh = 'every' if self.thresh_every else self.act_reid_thresh
        self.inact_reid_thresh = 'every' if self.thresh_every else self.inact_reid_thresh
        self.act_reid_thresh = 'tbd' if self.thresh_tbd else self.act_reid_thresh
        self.inact_reid_thresh = 'tbd' if self.thresh_tbd else self.inact_reid_thresh

    def add_dist_to_storage(self, gt_n, gt_t, num_active, num_inactive, dist, height):
        gt_n = np.atleast_2d(np.array(gt_n))
        gt_t = np.atleast_2d(np.array(gt_t))

        same_class = gt_t == gt_n.T
        act = np.atleast_2d(
            np.array(
                [1] *
                num_active +
                [0] *
                num_inactive)) == np.atleast_2d(
            np.ones(
                dist.shape[0])).T

        self.distance_[self.seq]['inact_dist_same'].extend(
            dist[same_class & ~act].tolist())
        self.distance_[self.seq]['act_dist_same'].extend(
            dist[same_class & act].tolist())
        self.distance_[self.seq]['inact_dist_diff'].extend(
            dist[~same_class & ~act].tolist())
        self.distance_[self.seq]['act_dist_diff'].extend(
            dist[~same_class & act].tolist())
        self.distance_[self.seq]['interaction_mat'].append(
            self.curr_interaction.tolist())
        self.distance_[
            self.seq]['occlusion_mat'].append(
            self.curr_occlusion.tolist())
        self.distance_[self.seq]['size'].append(height)
        self.distance_[self.seq]['active_inactive'].append(act.tolist())
        self.distance_[self.seq]['same_class_mat'].append(same_class.tolist())
        self.distance_[self.seq]['dist'].append(dist.tolist())

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
                [str(tr_id), str(d['gt_id']), self.round2(d['ioa']), self.round2(d['vis']), self.round2(d['conf'])])
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

    def init_debug(self):
        self.errors = defaultdict(int)
        self.event_dict = defaultdict(dict)
        self.round1 = lambda x: str(round(10 * x) / 10)

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
            self.emc_dict[self.seq_name].append(warp_matrix.tolist())
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

    def motion(self, approximate=True, only_vel=False):
        # adapted from tracktor
        """Applies a simple linear motion model that considers the last n_steps steps."""
        self.overall_velocity = list()
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

                        # print(np.stack(vs).mean(axis=0)-np.stack(vs_).mean(axis=0))
                if not only_vel:
                    track.update_v(np.stack(vs).mean(axis=0))
                    track.last_vc = np.stack(vs_c).mean(axis=0)
                    self.motion_step(track)
                self.overall_velocity.append(np.linalg.norm(
                    track.last_vc, ord=2))

        if not only_vel:
            for track in self.inactive_tracks.values():
                if len(track.last_pos) > 1:
                    self.motion_step(track)

        # get mean velicoty by (vc/h)*frame_rate
        if len(self.overall_velocity):
            self.overall_velocity = (np.asarray(
                self.overall_velocity)/np.asarray(height)).mean() \
                    * self.frame_rate
            self.vel_dict[self.seq].append(self.overall_velocity)
        else:
            self.overall_velocity = None

    def get_motion_dist(self, detections, curr_it):
        act_pos = [track.pos for track in self.tracks.values()]
        inact_pos = [track.pos for track in curr_it.values()]
        pos = torch.from_numpy(np.asarray(act_pos + inact_pos))
        det_pos = torch.from_numpy(np.asarray(
            [t['bbox'] for t in detections]))
        iou = bbox_overlaps(det_pos, pos)
        iou = 1 - iou
        return iou

    def combine_motion_appearance(self, iou, dist, detections, num_active, num_inactive, inactive_counts, gt_n=None, gt_t=None, curr_it=None):
        # occlusion
        ioa = torch.tensor([d['ioa'] for d in detections])
        ioa = ioa.repeat(dist.shape[1], 1).T.numpy()

        ioa_tr = torch.tensor([t.ioa for t in self.tracks.values()] + [t.ioa for t in curr_it.values()])
        ioa_tr = ioa_tr.repeat(dist.shape[0], 1).numpy()

        # init distances
        dist_emb = copy.deepcopy(dist)
        if type(iou) != np.ndarray:
            dist_iou = iou.cpu().numpy()
        else: 
            dist_iou = iou

        if self.store_dist:
            all_counts = [0] * num_active + inactive_counts
            self.distance_[self.seq]['iou_dist'].append(dist_iou.tolist())
            self.distance_[self.seq]['inactive_count'].append(all_counts)
            gt_n = np.atleast_2d(np.array(gt_n))
            gt_t = np.atleast_2d(np.array(gt_t))
            same_class = gt_t == gt_n.T
            same = dist_iou[:, :num_active][same_class[:, :num_active]].tolist()
            diff = dist_iou[:, :num_active][~same_class[:, :num_active]].tolist()
            self.distance_[self.seq]['iou_dist_diff']['same'][0].extend(same)
            self.distance_[self.seq]['iou_dist_diff']['diff'][0].extend(diff)
            for i, ic in enumerate(inactive_counts):
                same = dist_iou[:, num_active+i][same_class[:, num_active+i]].tolist()
                diff = dist_iou[:, num_active+i][~same_class[:, num_active+i]].tolist()
                self.distance_[self.seq]['iou_dist_diff']['same'][ic].extend(same)
                self.distance_[self.seq]['iou_dist_diff']['diff'][ic].extend(diff)

        # weighted of threshold
        if self.motion_model_cfg['ioa_threshold'] == 'sum':
            dist = (dist_emb + dist_iou)*0.5

        elif self.motion_model_cfg['ioa_threshold'] == 'penalty':
            w_e = (ioa + ioa_tr)/2 + np.finfo(float).eps

            w_1_m = np.array([0] * num_active + inactive_counts)/self.frame_rate
            w_1_m = self.sig(w_1_m, k=-2, v_max=2, v_min=0)
            w_1_m = np.repeat(
                np.expand_dims(w_1_m, axis=1), dist_emb.shape[0], axis=1).T
            if self.overall_velocity is not None:
                w_2_m = self.sig(self.overall_velocity, k=-80, v_max=2, v_min=0)
                w_2_m = w_2_m * np.ones(dist_emb.shape)
            else:
                w_2_m = np.zeros(dist_emb.shape)
            w_m = (w_1_m + w_2_m)/2 + np.finfo(float).eps

            dist = (1+w_e) * dist_emb + (1+w_m) * dist_iou

        elif self.motion_model_cfg['ioa_threshold'] == 'real_dist':
            l_1 = self.strfrac2float(self.motion_model_cfg['lambda_temp'])
            l_2 = self.strfrac2float(self.motion_model_cfg['lambda_occ'])
            l_3 = self.strfrac2float(self.motion_model_cfg['lambda_mot'])

            w_1_m = np.array([0] * num_active + inactive_counts)/self.frame_rate
            w_1_m = self.sig(w_1_m, k=-2, v_max=2, v_min=0)
            w_1_m = np.repeat(
                np.expand_dims(w_1_m, axis=1), dist_emb.shape[0], axis=1).T + np.finfo(float).eps

            w_1_r = np.array([0] * num_active + inactive_counts)/self.frame_rate
            w_1_r = self.sig(w_1_r, k=-4, v_max=2-0.3, v_min=0.3)
            w_1_r = np.repeat(
                np.expand_dims(w_1_r, axis=1), dist_emb.shape[0], axis=1).T + np.finfo(float).eps

            w_2_m = self.sig(ioa, k=6,  v_max=1, v_min=0.9) + np.finfo(float).eps
            w_2_r = self.sig(ioa, k=6,  v_max=1, v_min=-1) + np.finfo(float).eps

            if self.overall_velocity is not None:
                vel = self.overall_velocity #self.warp_matrix_nomr 
                w_3_m = -self.sig(vel/self.frame_rate, k=-120, v_max=-1, v_min=0, shift=-0.1)
                w_3_m = w_3_m * np.ones(dist_emb.shape) + np.finfo(float).eps
                w_3_r = -self.sig(vel/self.frame_rate, k=-80, v_max=-1.0, v_min=-0.96, shift=-0.1)
                w_3_r = w_3_r * np.ones(dist_emb.shape) + np.finfo(float).eps
            else:
                w_3_m = np.ones(dist_emb.shape)
                w_3_r = np.ones(dist_emb.shape)

            w_m = l_1 * w_1_m**(-2)/(w_1_m**(-2)+w_1_r**(-2)) + l_2 * w_2_m**(-2)/(w_2_m**(-2)+w_2_r**(-2)) + l_3 * w_3_m**(-2)/(w_3_m**(-2)+w_3_r**(-2))
            w_r = l_1 * w_1_r**(-2)/(w_1_m**(-2)+w_1_r**(-2)) + l_2 * w_2_r**(-2)/(w_2_m**(-2)+w_2_r**(-2)) + l_3 * w_3_r**(-2)/(w_3_m**(-2)+w_3_r**(-2))

            dist = w_m*dist_iou + w_r*dist_emb

        elif self.motion_model_cfg['ioa_threshold'] == 'KKT':
            # INACTIVE
            w_1 = np.array([0] * num_active + inactive_counts) / self.frame_rate
            # w_1 = self.hill(np.array([0] * num_active + inactive_counts), 40)
            w_1 = np.repeat(
                np.expand_dims(w_1, axis=1), dist_emb.shape[0], axis=1).T + 1.0

            # OCCLUSION LEVEL
            w_2 = ioa + 1.0

            # motion: the larger the motion, the less reliable motion
            # this means: the larger motion, the more weight to reid
            if self.overall_velocity is not None:
                # w_3 = self.hill(self.overall_velocity, 2) * np.ones(dist_emb.shape) + 1.0
                w_3 = (self.overall_velocity/self.frame_rate) * np.ones(dist_emb.shape) + 1.0
            else:
                w_3 = np.zeros(dist_emb.shape) + 1.0
            
            denom = w_1**(-2) + w_2**(-2) + w_3**(-2)

            dist = w_1**(-2)/denom * dist_iou + \
                w_2**(-2)/denom * dist_emb + \
                    w_3**(-2)/denom * dist_iou

        elif self.motion_model_cfg['ioa_threshold'] == 'motion':
            dist = dist_iou

        elif 'GuiJen' in self.motion_model_cfg['ioa_threshold']:
            dist[:, :num_active] = (
                dist_emb[:, :num_active] + dist_iou[:, :num_active])*0.5

            o = float(self.motion_model_cfg['ioa_threshold'].split('_')[-1])

            w = self.hill(np.array([0] * num_active + inactive_counts), o)
            w = np.repeat(
                np.expand_dims(w, axis=1), dist_emb.shape[0], axis=1).T

            dist[:, num_active:] = w[:, num_active:] * dist_emb[:, num_active:] + \
                (1-w[:, num_active:]) * dist_iou[:, num_active:]

        elif self.motion_model_cfg['ioa_threshold'] == 'sumFairMOTNan':
            dist = dist_emb + dist_iou
            dist[:, :num_active] = np.where(
                dist[:, :num_active] == 1,
                np.nan,
                dist[:, :num_active])

        elif self.motion_model_cfg['ioa_threshold'] == 'SeperateAssignment':
            dist = [dist_emb, dist_iou]

        elif self.motion_model_cfg['ioa_threshold'] == 'weightedSameRange':
            dist_emb = (dist_emb-dist_emb.min())/(dist_emb.max()-dist_emb.min())
            dist_iou = (dist_iou-dist_iou.min())/(dist_iou.max()-dist_iou.min())
            dist = (1-ioa) * dist_emb + ioa * dist_iou
        
        elif 'MotionTemporalAndOcclusion' in self.motion_model_cfg['ioa_threshold']:
            # define lambdas for mixtures
            l_1 = self.strfrac2float(self.motion_model_cfg['lambda_temp'])
            l_2 = self.strfrac2float(self.motion_model_cfg['lambda_occ'])
            l_3 = self.strfrac2float(self.motion_model_cfg['lambda_mot'])

            # h values for hill function
            ic = float(self.motion_model_cfg['ioa_threshold'].split('_')[-3])
            v = float(self.motion_model_cfg['ioa_threshold'].split('_')[-2])
            o = float(self.motion_model_cfg['ioa_threshold'].split('_')[-1])
            
            sig = False

            # INACTIVE
            # the larger the inactive count the more wieght to reid
            if not sig:
                w_1 = self.hill(np.array([0] * num_active + inactive_counts), ic)
                w_1 = np.repeat(
                    np.expand_dims(w_1, axis=1), dist_emb.shape[0], axis=1).T
            else:
                w_1 = np.array([0] * num_active + inactive_counts)/self.frame_rate
                # w_1 = self.sig(w_1, 3*np.log(1+np.sqrt(3)), v_min=0.5, v_max=1, shift=-1.5)
                w_1 = self.sig(w_1, 3*np.log(1+np.sqrt(3)), v_min=-1, v_max=1, shift=0)
                w_1 = np.repeat(
                    np.expand_dims(w_1, axis=1), dist_emb.shape[0], axis=1).T

            # OCCLUSION LEVEL
            # the larger the occlusion the more weight to motion
            if not sig:
                if o == 0:
                    w_2 = ioa
                else:
                    w_2 = self.hill(ioa, o)
            else:
                # w_2 = self.sig(ioa, 8*np.log(1+np.sqrt(3)), v_min=0.5, v_max=1, shift=-0.5)
                w_2 = self.sig(ioa, 8*np.log(1+np.sqrt(3)), v_min=-1, v_max=1, shift=0)

            # motion: the larger the motion, the less reliable motion
            # this means: the larger motion, the more weight to reid
            if self.overall_velocity is not None:
                if not sig:
                    w_3 = self.hill(self.overall_velocity, v)
                    w_3 = w_3 * np.ones(dist_emb.shape)
                else:
                    w_3 = self.sig(self.overall_velocity, 2*np.log(1+np.sqrt(3)), v_min=0.5, v_max=1, shift=-1.5)
                    w_3 = self.sig(self.overall_velocity, 2*np.log(1+np.sqrt(3)), v_min=-1, v_max=1, shift=0)
            else:
                w_3 = 0.5 * np.ones(dist_emb.shape)

            # probability that embedding / motion dist is better
            p_emb = l_1 * w_1 + l_2 * (1-w_2) + l_3 * w_3
            p_mot = l_1 * (1-w_1) + l_2 * (w_2) + l_3 * (1-w_3)

            if self.store_dist:
                self.distance_[self.seq]['prob_dict'][
                    self.frame_id] = p_emb.tolist()

            dist = p_emb * dist_emb + p_mot * dist_iou

        elif self.motion_model_cfg['ioa_threshold'] == 'weighted':
            # dist[:, :num_active] = (1-ioa[:, :num_active]) * dist_emb[:, :num_active] + ioa[:, :num_active] * dist_iou[:, :num_active]
            dist = (1-ioa) * dist_emb + ioa * dist_iou

        elif self.motion_model_cfg['ioa_threshold'] == 'weighted_inter':
            dist = (1-inter) * dist_emb + inter * dist_iou
            # dist = np.clip(1-inter/2, a_min=0, a_max=None) * dist_emb + np.clip(inter/2, a_min=None, a_max=1) * dist_iou
        elif self.motion_model_cfg['ioa_threshold'] == 'learned':
            inact = np.array([0] * num_active + inactive_counts)/self.frame_rate
            inact = np.repeat(
                np.expand_dims(inact, axis=1), dist_emb.shape[0], axis=1).T
            if self.overall_velocity is not None:
                vel = np.ones(inact.shape) * self.overall_velocity
            else:
                vel = np.ones(inact.shape) * 0.5
            ioa = torch.flatten(torch.from_numpy(ioa))
            inact = torch.flatten(torch.from_numpy(inact))
            vel = torch.flatten(torch.from_numpy(vel))

            if self.train_cfg['input_dim'] == 3:
                inp = torch.stack([inact, ioa, vel]).to(self.device).T
                with torch.no_grad():
                    weights = self.weight_pred(inp)
                weights_emb = weights.reshape(dist_emb.shape).cpu().numpy()
                weights_ioa = (1 - weights).reshape(dist_emb.shape).cpu().numpy()
                # make dist with weights
                dist = dist_emb * weights_emb + \
                    dist_iou * weights_ioa
            else:
                emb = torch.flatten(torch.from_numpy(dist_emb))
                iou = torch.flatten(torch.from_numpy(dist_iou))
                inp = torch.stack([inact, ioa, vel, emb, iou]).to(self.device).T
                with torch.no_grad():
                    dist = self.weight_pred(inp)
                dist = dist.reshape(dist_emb.shape).cpu().numpy()
            
        else:
            # get masks
            ioa_mask = ioa > self.motion_model_cfg['ioa_threshold']
            area_out_mask = area_out > self.motion_model_cfg['area_out_threshold']
            num_occ_mask = num_occ >= self.motion_model_cfg['num_occ_thresh']
            inter_mask = inter > self.motion_model_cfg['inter_threshold']
            num_inter_mask = num_inter >= self.motion_model_cfg['num_inter_thresh']

            dist_emb[ioa_mask] = 0
            dist_iou[~ioa_mask] = 0

            # dist_emb[(area_out_mask | ioa_mask | num_occ_mask | inter_mask)] = 0
            # dist_iou[(~area_out_mask & ~ioa_mask & ~num_occ_mask & ~inter_mask)] = 0

            # dist_emb[(ioa_mask & num_occ_mask) | area_out_mask] = 0
            # dist_iou[(~ioa_mask | ~num_occ_mask) & ~area_out_mask] = 0

            # dist_emb[(inter_mask & num_inter_mask) | area_out_mask] = 0
            # dist_iou[(~inter_mask | ~num_inter_mask) & ~area_out_mask] = 0

            # dist[:, :num_active] = dist_emb[:, :num_active] + dist_iou[:, :num_active]
            dist = dist_emb + dist_iou
            # dist = dist_iou

        if self.make_weight_pred_dataset:
            frame_info = {
                'iou': dist_iou.tolist(),
                'emb': dist_emb.tolist(),
                'ioa': ioa.tolist(),
                'vel': self.overall_velocity,
                'act_counts': [0] * num_active + inactive_counts,
                'gt_n': gt_n.tolist(),
                'gt_t': gt_t.tolist()}
            self.weight_pred_dataset[self.seq_name][self.frame_id] = frame_info

        return dist, ioa

    def active_proximity(self, dist, num_active, detections):
        active_pos = np.asarray([track.pos for track in self.tracks.values()])
        det_pos = np.asarray([t['bbox'] for t in detections])

        center_active = get_center(active_pos)
        center_active = np.atleast_2d(center_active.T)
        center_det = get_center(det_pos)
        center_det = np.atleast_2d(center_det.T)

        center_dist = sklearn.metrics.pairwise_distances(
                    center_det, center_active, metric='euclidean')
        
        center_dist = center_dist < 100

        #dist[:, :num_active] = np.where(center_dist > 20, np.nan,  dist[:, :num_active])

        return center_dist

