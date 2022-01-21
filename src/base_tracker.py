
from collections import defaultdict

import torch
#from tracking_wo_bnw.src.tracktor.utils import interpolate
import os
import numpy as np
import os.path as osp
import csv
import logging
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from src.tracking_utils import get_center, get_height, get_width, make_pos, warp_pos, bbox_overlaps
import cv2

logger = logging.getLogger('AllReIDTracker.BaseTracker')


class BaseTracker():
    def __init__(
            self,
            tracker_cfg,
            encoder,
            net_type='resnet50',
            output='plain',
            weight='No',
            data='tracktor_preprocessed_files.txt'):

        self.net_type = net_type
        self.encoder = encoder

        self.tracker_cfg = tracker_cfg
        self.motion_model_cfg = tracker_cfg['motion_config']
        self.output = output

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

    def make_results(self):
        results = defaultdict(dict)
        for i, ts in self.tracks.items():
            for t in ts:
                results[i][t['im_index']] = t['bbox']
        return results

    def _remove_short_tracks(self, all_tracks):
        tracks_new = dict()
        for k, v in all_tracks.items():
            if len(v) > self.tracker_cfg['length_thresh']:
                tracks_new[k] = v
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
            self.experiment += 'MC:1'
        
        if self.motion_model_cfg['apply_motion_model']:
            self.experiment += 'MM:1'

        self.experiment = '_'.join([self.data[:-4],
                                    weight,
                                    'evalBB:' + str(self.tracker_cfg['eval_bb']),
                                    self.experiment])
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
            if i == 0:
                logger.info("Using moving average of seq...")

            self.encoder.train()
            if self.tracker_cfg['running_mean_seq_reset'] and i == 0:
                logger.info(
                    "Resetting BatchNorm statistics and use first batch as initial mean/std...")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1
                        m.first_batch_mean = True
            elif self.tracker_cfg['running_mean_seq_reset'] and i == 1:
                logger.info("Setting mometum to 0.1 again...")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.momentum = 0.1
            with torch.no_grad():
                _, _ = self.encoder(frame, output_option=self.output)
            self.encoder.eval()

        elif i == 0 and (self.tracker_cfg['first_batch'] or self.tracker_cfg['first_batch_reset']):
            logger.info("Runing first batch in train mode...")

            self.encoder.train()
            if self.tracker_cfg['first_batch_reset']:
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1  # means no running mean, i.e., uses first batch wo initialization
                logger.info("Resetting BatchNorm statistics...")
            with torch.no_grad():
                _, _ = self.encoder(frame, output_option=self.output)
            self.encoder.eval()

    def normalization_before(self, seq, k=5, first=False):
        '''
        Run the first k frames in train mode
        '''
        if first:
            logger.info('Resetting BatchNorm statistics...')
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()

        if self.tracker_cfg['random_patches_several_frames'] or self.tracker_cfg['several_frames']:
            what_input = 'bounding boxes' if self.tracker_cfg['several_frames'] else 'random patches'
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

    def setup_seq(self, seq, first):
        self.seq = f"Sequence{int(seq.name.split('-')[1])}_" + \
            seq.name.split('-')[2]
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
        elif first:
            logger.info('Feeding sequence data before tracking once')

        # if random patches should be sampled
        seq.random_patches = self.tracker_cfg['random_patches'] or self.tracker_cfg[
            'random_patches_first'] or self.tracker_cfg['random_patches_several_frames']

        if self.save_embeddings_by_id:
            self.embeddings_by_id[seq.name] = defaultdict(list)

    def reset_threshs(self):
        self.act_reid_thresh = 'every' if self.thresh_every else self.act_reid_thresh
        self.inact_reid_thresh = 'every' if self.thresh_every else self.inact_reid_thresh
        self.act_reid_thresh = 'tbd' if self.thresh_tbd else self.act_reid_thresh
        self.inact_reid_thresh = 'tbd' if self.thresh_tbd else self.inact_reid_thresh

    def add_dist_to_storage(self, gt_n, gt_t, num_active, num_inactive, dist):
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
        self.distance_[self.seq]['active_inactive'].append(act.tolist())
        self.distance_[self.seq]['same_class_mat'].append(same_class.tolist())
        self.distance_[self.seq]['dist'].append(dist.tolist())

    def update_thresholds(self, dist, num_active, num_inactive):
        # update active threshold
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

    def visualize(self, tracks, path, seq, frame):
        if frame == 1:
            os.makedirs(
                osp.join('visualizations', self.experiment, seq),
                exist_ok=True)
        img = matplotlib.image.imread(path)
        figure, ax = plt.subplots(1)
        figure.set_size_inches(img.shape[1]/100, img.shape[0]/100)
        for t in tracks:
            if t['id'] not in self.id_to_col.keys():
                self.id_to_col[t['id']] = self.color_list[self.col]
                self.col += 1

            # add rectangle
            rect = matplotlib.patches.Rectangle(
                (t['bbox'][0], t['bbox'][1]),
                t['bbox'][2]-t['bbox'][0],
                t['bbox'][3]-t['bbox'][1],
                edgecolor=self.id_to_col[t['id']],
                facecolor="none",
                linewidth=2)
            ax.add_patch(rect)

            # add text with id, ioa and visibility
            text = ', '.join(
                [str(t['id']), self.round2(t['ioa']), self.round2(t['vis'])])
            plt.text(
                t['bbox'][0]-15,
                t['bbox'][1]+(t['bbox'][3]-t['bbox'][1])/2,
                text,
                va='center',
                rotation='vertical',
                c=self.id_to_col[t['id']],
                fontsize=8)

        ax.imshow(img)
        plt.axis('off')
        plt.savefig(
            osp.join('visualizations', self.experiment, seq, f"{frame:06d}.jpg"),
            bbox_inches='tight',
            pad_inches=0,
            dpi=100)
        plt.close()

    def init_vis(self):
        colors = mcolors.CSS4_COLORS
        color_list = list(colors.keys())
        self.color_list = color_list + color_list
        random.shuffle(color_list)
        self.id_to_col = dict()
        self.col = 0
        self.round2 = lambda x : str(round(100*x)/100)

    def init_debug(self):
        self.errors = defaultdict(int)
        self.event_dict = defaultdict(dict)
        self.round1 = lambda x : str(round(10*x)/10)

    @staticmethod
    def plot_single_image(imgs, name):
        import matplotlib.pyplot as plt

        for i, img in enumerate(imgs):
            img = img.permute(1, 2, 0)
            figure, ax = plt.subplots(1)
            figure.set_size_inches(img.shape[1]/100, img.shape[0]/100)
            ax.imshow(np.asarray(img))
            plt.savefig(str(i) + + name + '.png', dpi=100)
            plt.close()

    def motion_compensation(self, whole_image, im_index):
        # adapted from tracktor
        if im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(whole_image.cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.motion_model_cfg['num_iter_mc'],  self.motion_model_cfg['termination_eps_mc'])
            _, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode_dct[self.motion_model_cfg['warp_mode']], criteria, None, 15)
            warp_matrix = torch.from_numpy(warp_matrix)

            for tracks in [self.tracks, self.inactive_tracks]:
                for tid, t in tracks.items():
                    for d in t:
                        if type(d['pos']) == torch.Tensor:
                            d['pos'] = d['pos'].cpu().numpy()
                        d['pos'] = warp_pos(np.atleast_2d(d['pos']), warp_matrix)
                        # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data
        
        self.last_image = whole_image

    def motion_step(self, track):
        # adapted from tracktor
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track[-1]['pos']) + track[-1]['last_v']
            track[-1]['pos'] = make_pos(
                *center_new,
                get_width(track[-1]['pos']),
                get_height(track[-1]['pos']))
        else:
            track[-1]['pos'] = track[-1]['pos'] + track[-1]['last_v']
            print('whhat', type(track[-1]['pos']))

    def motion(self):
        # adapted from tracktor
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for tid, t in self.tracks.items():
            print([tr['last_pos'] for tr in t])
            last_pos = np.asarray([tr['last_pos'] for tr in t])
            print(last_pos.shape, 'Hellp')
            if len(last_pos) > 1:
                # avg velocity between each pair of consecutive positions in t.last_pos
                if self.motion_model_cfg['center_only']:
                    vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
                else:
                    vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

                t[-1]['last_v'] = torch.stack(vs).mean(dim=0)
                self.motion_step(t)
            else:
                t[-1]['last_v'] = 0
            t[-1]['last_pos'].append(t[-1]['pos'].cpu().numpy())
            print('appending:', t[-1]['pos'].cpu().numpy())

        for t in self.inactive_tracks:
            if t[-1]['last_v'] is not None:
                self.motion_step(t)

    def get_motion_dist(self, tracks):
        act_pos = [t[-1]['pos'] for tid, t in self.tracks.items()]
        inact_pos = [t[-1]['pos'] for tid, t in self.inactive_tracks.items()]
        pos = torch.cat(act_pos + inact_pos, 0)
        det_pos = torch.from_numpy(np.asarray([t['pos'] for t in tracks])).cuda()
        iou = bbox_overlaps(det_pos, pos)
        iou = 1 - iou
        return iou

    def combine_motion_appearance(self, iou, dist, tracks):
        ioa = torch.tensor([t['ioa'] for t in tracks])
        ioa = ioa.repeat(dist.shape[1], 1).T
        ioa_mask = ioa > 0.8
        dist_emb = dist
        dist_emb[ioa_mask] = 0
        dist_iou = iou
        dist_iou[~ioa_mask] = 0
        dist = dist_emb + dist_iou.cpu().numpy()
        return dist

        

