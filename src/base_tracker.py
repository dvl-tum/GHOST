
from collections import defaultdict
import torch.nn as nn
import torch
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
    make_pos, warp_pos, bbox_overlaps, is_moving, frame_rate, mot_fps, bisoftmax
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
            data='tracktor_preprocessed_files.txt',
            device='cpu'):

        # initialize all variables
        self.kalman = tracker_cfg['kalman']
        if self.kalman:
            self.shared_kalman = KalmanFilter()
            self.kalman_filter = KalmanFilter()
        else:
            self.kalman_filter = None

        self.log = True
        self.device = device
        self.round1_float = lambda x: round(10 * x) / 10

        self.net_type = net_type
        self.encoder = encoder

        self.tracker_cfg = tracker_cfg
        self.motion_model_cfg = tracker_cfg['motion_config']
        self.output = output if not tracker_cfg['use_bism'] else 'norm'

        self.inact_patience = tracker_cfg['inact_patience']
        self.act_reid_thresh = tracker_cfg['act_reid_thresh']
        self.inact_reid_thresh = tracker_cfg['inact_reid_thresh']
        self.output_dir = tracker_cfg['output_dir']
        self.nan_first = tracker_cfg['nan_first']

        self.warp_mode_dct = {
            'Translation': cv2.MOTION_TRANSLATION,
            'Affine': cv2.MOTION_AFFINE,
            'Euclidean': cv2.MOTION_EUCLIDEAN,
            'Homography': cv2.MOTION_HOMOGRAPHY
        }

        self.data = data
        # get experiment name
        self.get_name()
        logger.info("Executing experiment {}...".format(self.experiment))

        # make output dir
        os.makedirs(osp.join(self.output_dir, self.experiment), exist_ok=True)

        # set in eval mode if wanted
        if not self.tracker_cfg['on_the_fly']:
            self.encoder.eval()

        # if visualization of bounding boxes initliaze
        self.store_visualization = self.tracker_cfg['visualize']
        if self.store_visualization:
            self.init_vis()

        # if store features for analysis initialize
        self.store_feats = self.tracker_cfg['store_feats']
        if self.store_feats:
            self.features_ = defaultdict(dict)

        self.fps = mot_fps

    def dist(self, x, y):
        """
        Compute distance using cosine distance or euclidean
        or utilize bisoftmax as diatnce measures
        """
        if not self.tracker_cfg['use_bism']:
            if self.tracker_cfg['distance'] == 'cosine':
                dist = 1 - \
                    F.cosine_similarity(x[:, :, None], y.t()[None, :, :])
            else:
                dist = F.pairwise_distance(x[:, :, None], y.t()[None, :, :])

            return dist.cpu().numpy()
        else:
            dist = 1 - bisoftmax(x.cpu(), y.cpu())
            return dist.numpy()

    def strfrac2float(self, x):
        """
        Convert number given as string to float
        """
        if type(x) == int or type(x) == float:
            return x
        return float(x.split('/')[0]) / float(x.split('/')[-1])

    def get_features(self, frame):
        """
        Compute reid feature vectors
        """
        # forward pass
        with torch.no_grad():
            if self.net_type == 'resnet50_analysis' or self.net_type == "IBN":
                feats = self.encoder(frame)
                feats = F.normalize(feats, p=2, dim=1)
            else:
                _, feats = self.encoder(frame, output_option=self.output)

        return feats

    def make_results(self):
        """
        Get results dict: dictionary with 1 dictionary for every track:
            {..., i: [x1,y1,x2,y2,label], ...}
        i is track id
        """
        results = defaultdict(dict)
        for i, ts in self.tracks.items():
            for im_index, bbox, label in zip(ts.past_im_indices, ts.bbox, ts.label):
                results[i][im_index] = bbox.tolist() + [label]
        return results

    def _remove_short_tracks(self, all_tracks):
        """
        Remove short tracks with len < len thresh (default = 0)
        """
        tracks_new = dict()
        for k, v in all_tracks.items():
            if len(v) > self.tracker_cfg['length_thresh']:
                tracks_new[k] = v
            else:
                logger.info(len(v))
        logger.info(
            f"Removed {len(all_tracks) - len(tracks_new)} short tracks")
        return tracks_new

    def write_results(self, output_dir, seq_name):
        """Write the tracks in the format for MOT16/MOT17 sumbission
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

    def get_name(self):
        """
        Get parameters and make of experiment name
        """
        inact_avg = self.tracker_cfg['avg_inact']['proxy'] + str(
            self.tracker_cfg['avg_inact']['num']) \
            if self.tracker_cfg['avg_inact']['do'] else 'LastFrame'
        act_avg = self.tracker_cfg['avg_act']['proxy'] + str(
            self.tracker_cfg['avg_act']['num']) \
            if self.tracker_cfg['avg_act']['do'] else 'LastFrame'
        self.experiment = inact_avg + ':' + str(self.tracker_cfg['inact_reid_thresh']) + \
            ':' + act_avg + ':' + str(self.tracker_cfg['act_reid_thresh'])

        if self.tracker_cfg['random_patches_several_frames']:
            self.experiment += 'RandomPatchesSeveralFrames:1'
        if self.tracker_cfg['several_frames']:
            self.experiment += 'SeveralFrames:1'
        if self.tracker_cfg['random_patches']:
            self.experiment += 'RandomPatches:1'
        if self.tracker_cfg['random_patches_first']:
            self.experiment += 'RandomPatchesFirst:1'
        if self.tracker_cfg['running_mean_seq']:
            self.experiment += 'RunningMeanSeq:1'
        if self.tracker_cfg['first_batch']:
            self.experiment += 'FirstBatch:1'
        if self.tracker_cfg['running_mean_seq_reset']:
            self.experiment += 'RunningMeanSeqReset:1'
        if self.tracker_cfg['first_batch_reset']:
            self.experiment += 'FirstBatchReset:1'
        if self.tracker_cfg['every_frame_several_frames']:
            self.experiment += 'EveryFrameSeveralFrames:1'

        self.experiment += 'LenThresh:' + \
            str(self.tracker_cfg['length_thresh'])
        self.experiment += 'RemUnconf:' + str(self.tracker_cfg['unconfirmed'])
        self.experiment += 'LastNFrames:' + \
            str(self.motion_model_cfg['last_n_frames'])

        if self.tracker_cfg['use_bism']:
            self.experiment += 'Bism:1'

        if self.tracker_cfg['nan_first']:
            self.experiment += 'NanFirst:1'

        if self.motion_model_cfg['motion_compensation']:
            self.experiment += 'MCOM:1'  # Only Moving

        if self.motion_model_cfg['apply_motion_model']:
            self.experiment += 'MM:1'
            if self.tracker_cfg['kalman']:
                self.experiment += 'Kalman'
            else:
                self.experiment += str(self.motion_model_cfg['combi'])

        self.experiment = '_'.join([self.data[:-4],
                                    'OnTheFly:' +
                                    str(self.tracker_cfg['on_the_fly']),
                                    self.experiment])

        self.experiment += 'InactPat:' + \
            str(self.tracker_cfg['inact_patience'])
        self.experiment += 'DetConf:' + str(self.tracker_cfg['det_conf'])
        self.experiment += 'NewTrackConf:' + \
            str(self.tracker_cfg['new_track_conf'])

    def normalization_experiments(self, random_patches, frame, i, seq, k=10):
        '''
        Normalization experiments:
         * Reset bn stats and use random patches to update bn stats
         * Reset bn stats and use random patches of the first frame to 
            update bn stats
         * Use running mean of bbs as bn stats
         * Reset bn stats and use running mean of bbs as bn stats
         * Use bbs of the first batch
         * Reset bn stats and use bbs of the first batch
         * For every frame use stats of several frames
        '''
        # use random patches
        if self.tracker_cfg['random_patches'] or self.tracker_cfg['random_patches_first']:
            if i == 0:
                which_frame = \
                    'each frame' if self.tracker_cfg['random_patches'] else 'first frame'
                if self.log:
                    logger.info(
                        "Using random patches of {}...".format(which_frame))

            # either every frame of only first  frame
            if not self.tracker_cfg['random_patches_first'] or i == 0:
                self.encoder.train()
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        # means no running mean
                        m.momentum = 1
                # feed through network
                with torch.no_grad():
                    _, _ = self.encoder(
                        random_patches, output_option=self.output)
            # set to eval mode again
            self.encoder.eval()

        # take the running mean of sequence for stats
        elif self.tracker_cfg['running_mean_seq'] or \
                self.tracker_cfg['running_mean_seq_reset']:
            if i == 0 and self.log:
                logger.info("Using moving average of seq...")

            self.encoder.train()
            # if reset stats then reset in first frame
            if self.tracker_cfg['running_mean_seq_reset'] and i == 0:
                if self.log:
                    logger.info(
                        "Resetting BatchNorm statistics and use first batch as initial mean/std...")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        # start from first stats
                        m.momentum = 1
                        m.first_batch_mean = True
            # reset momentum
            elif self.tracker_cfg['running_mean_seq_reset'] and i == 1:
                if self.log:
                    logger.info("Setting mometum to 0.1 again...")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # update stats with momentum 0.1
                        m.momentum = 0.1
            # feed through network
            with torch.no_grad():
                _, _ = self.encoder(frame, output_option=self.output)

            # set to eval mode again
            self.encoder.eval()

        # take stats of first batch and reset
        elif i == 0 and (self.tracker_cfg['first_batch'] or
                         self.tracker_cfg['first_batch_reset']):
            if self.log:
                logger.info("Runing first batch in train mode...")

            self.encoder.train()
            if self.tracker_cfg['first_batch_reset']:
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        # means no running mean, i.e., uses first batch wo initialization
                        m.momentum = 1
                if self.log:
                    logger.info("Resetting BatchNorm statistics...")
            # feed through network
            with torch.no_grad():
                _, _ = self.encoder(frame, output_option=self.output)

            # set to eval mode again
            self.encoder.eval()

         # take stats of several frames around current frame and reset
        elif self.tracker_cfg['every_frame_several_frames']:
            if i == 0:
                if self.log:
                    logger.info(
                        f"Using {k} frames of bounding boxes before every frame to set BatchNorm statistics for each frame...")
            # reset stats
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.momentum = 1
                    m.first_batch_mean = True
            # set model in train mode
            self.encoder.train()

            # feed bounding boxes to adapt stats
            if i < k:
                idxs = list(range(k))
            if i+k > seq.num_frames:
                idxs = list(range(seq.num_frames-k, seq.num_frames))
            else:
                idxs = list(range(i, i+k))
            # feed k frames through encoder
            for n, idx in enumerate(idxs):
                if n == 1:
                    for m in self.encoder.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            # update stats with momentum 0.1
                            m.momentum = 0.1
                frame = seq._get(idx, just_frame=True)
                with torch.no_grad():
                    _, _ = self.encoder(frame, output_option=self.output)

            # set to eval mode again
            self.encoder.eval()

    def normalization_before(self, seq, k=15, first=False):
        '''
        * Run whole dataset in train mode to get statistics first and 
            then again in eval mode
        * Run the first k frames in train mode and use ranom patches 
            or bounding boxes of first k frames
        '''

        # feeding whole sequence data first --> reset stats
        if first:
            if self.log:
                logger.info('Resetting BatchNorm statistics...')
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()

        # if using random patches or several frames at the beginning
        if self.tracker_cfg['random_patches_several_frames'] or \
           self.tracker_cfg['several_frames']:
            what_input = 'bounding boxes' if self.tracker_cfg['several_frames'] else 'random patches'
            if self.log:
                logger.info(
                    "Using {} of the first {} frames to initialize BatchNorm statistics...".format(
                        k, what_input))
            self.encoder.train()

            # feed those frames through backbone
            for i, (frame, _, _, _, _, _, _, random_patches,
                    _, _, _, _) in enumerate(seq):

                # decide what to use
                if what_input == 'random patches':
                    inp = random_patches
                else:
                    inp = frame

                with torch.no_grad():
                    _, _ = self.encoder(inp, output_option=self.output)

                # only first k frames
                if i >= k:
                    break

            self.encoder.eval()
            seq.random_patches = False

    def setup_seq(self, seq=None, first=False):
        """
        Set up sequence:
        * set sequence name
        * initialize storage of features
        * if thresholds should be adapted automatically, initalize this
        * initialize track storage
        * set backbone into evaluation mode if wanted
        * set if random patches needed
        * get if sequences moving and frame rate
        """
        # set sequ name
        if "MOT" in seq:
            if len(seq.name.split('-')) > 2:
                # remove '-' from name
                self.seq = f"Sequence{int(seq.name.split('-')[1])}_" + \
                    seq.name.split('-')[2]
            else:
                self.seq = f"Sequence{int(seq.name.split('-')[1])}"

        else:
            self.seq = seq.name

        # add feature dict for seq
        if self.store_feats:
            self.init_feats()

        # automatic thresh update
        self.thresh_every = True if self.act_reid_thresh == "every" else False
        self.thresh_tbd = True if self.act_reid_thresh == "tbd" else False

        # initalize track storage
        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        self.mv_avg = dict()
        self.id = 0

        # set backbone into evaluation mode
        if not self.tracker_cfg['on_the_fly'] and not first:
            self.encoder.eval()
        elif first and self.log:
            logger.info('Feeding sequence data before tracking once')

        # if random patches should be sampled
        if seq is not None:
            seq.random_patches = self.tracker_cfg['random_patches'] or self.tracker_cfg[
                'random_patches_first'] or self.tracker_cfg['random_patches_several_frames']

        # get if sequence si moving and frame rate
        self.is_moving = is_moving(seq.name)
        self.frame_rate = frame_rate(seq.name)
        if self.log:
            logger.info("Frame rate: {}".format(self.frame_rate))

    def reset_threshs(self):
        self.act_reid_thresh = 'every' if self.thresh_every \
            else self.act_reid_thresh
        self.inact_reid_thresh = 'every' if self.thresh_every \
            else self.inact_reid_thresh
        self.act_reid_thresh = 'tbd' if self.thresh_tbd else \
            self.act_reid_thresh
        self.inact_reid_thresh = 'tbd' if self.thresh_tbd else \
            self.inact_reid_thresh

    def update_thresholds(self, dist, num_active, num_inactive):
        # compute threshold every frame or only in first frame if tbd
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

    def visualize(self, detections, tr_ids, path, seq, frame, do_text=False):
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

            d = copy.deepcopy(d)
            d['bbox'][0] = max(d['bbox'][0], 0)
            d['bbox'][1] = max(d['bbox'][1], 0)
            d['bbox'][2] = min(d['bbox'][2], img.shape[1])
            d['bbox'][3] = min(d['bbox'][3], img.shape[0])

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
            if do_text:
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

    def init_feats(self):
        self.features_[self.seq] = dict()

    def motion_compensation(self, whole_image, im_index):
        """
        compensate ego motion for bounding boxes
        """
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
                            pos = np.squeeze(
                                warp_pos(np.atleast_2d(pos), warp_matrix))
                            track.last_pos[i] = pos

        self.last_image = whole_image

    def motion_step(self, track):
        """
        Updates the given track's position by one step based on track.last_v
        """
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(
                *center_new,
                get_width(track.pos),
                get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self, approximate=False):
        """
        Applies a simple linear motion model that considers the 
        last n_steps steps.
        """
        if self.i == 0:
            logger.info(f"Appriximating motion {approximate}...")

        # update active tracks
        for track in self.tracks.values():
            if len(track.last_pos) > 1:
                last_pos = np.asarray(
                    track.last_pos[-self.motion_model_cfg['last_n_frames']:])
                frames = np.asarray(
                    track.past_frames[-self.motion_model_cfg['last_n_frames']:])

                # avg velocity between each pair of consecutive positions in
                # t.last_pos
                if self.motion_model_cfg['center_only']:
                    if approximate:
                        vs = np.stack([get_center(p2) - get_center(p1)
                                       for p1, p2 in zip(last_pos, last_pos[1:])])
                    else:
                        dt = [p2 - p1 for p1, p2 in zip(frames, frames[1:])]
                        vs = np.stack([(get_center(p2) - get_center(p1))/t
                                       for p1, p2, t in zip(last_pos, last_pos[1:], dt)])

                    track.update_v(vs.mean(axis=0))

                    # motion step
                    center_new = get_center(track.pos) + track.last_v
                    track.pos = make_pos(
                        *center_new,
                        get_width(track.pos),
                        get_height(track.pos))
                else:
                    if approximate:
                        vs = [p2 - p1 for p1,
                              p2 in zip(last_pos, last_pos[1:])]
                    else:
                        vs = (last_pos[1:, ]-last_pos[:-1, ])/np.repeat(
                            np.expand_dims((frames[1:]-frames[:-1]), 1), [4], axis=1)

                    # motion stepß
                    track.update_v(vs.mean(axis=0))
                    track.pos = track.pos + track.last_v

        # update for inactive tracks with last known dist
        for track in self.inactive_tracks.values():
            if len(track.last_pos) > 1:
                self.motion_step(track)

    def get_motion_dist(self, detections, curr_it):
        '''
        Compute motion distance using IoU distance of bounding boxes
        '''
        act_pos = [track.pos for track in self.tracks.values()]
        inact_pos = [track.pos for track in curr_it.values()]
        pos = torch.from_numpy(np.asarray(act_pos + inact_pos))
        det_pos = torch.from_numpy(np.asarray(
            [t['bbox'] for t in detections]))
        iou = bbox_overlaps(det_pos, pos)
        iou = 1 - iou
        return iou

    def combine_motion_appearance(self, iou, dist):
        """
        Combine appearance and motion distances
        """
        # init distances
        dist_emb = copy.deepcopy(dist)
        if type(iou) != np.ndarray:
            dist_iou = iou.cpu().numpy()
        else:
            dist_iou = iou

        # weighted of threshold
        if 'sum' in self.motion_model_cfg['combi']:
            alpha = float(self.motion_model_cfg['combi'].split('_')[-1])
            dist = ((1-alpha)*dist_emb + alpha*dist_iou)
        elif self.motion_model_cfg['combi'] == 'SeperateAssignment':
            dist = [dist_emb, dist_iou]

        return dist

    def add_feats_to_storage(self, detections):
        """
        Store features for anaylsis
        """
        detections_to_save = copy.deepcopy(detections)
        for d in detections_to_save:
            d['bbox'] = d['bbox'].tolist()
            d['feats'] = d['feats'].tolist()
        self.features_[self.seq][self.frame_id] = detections_to_save
