from io import IOBase
import pandas as pd
from collections import defaultdict, Counter

from pandas.core import frame
import torch
import sklearn.metrics
import scipy
#from tracking_wo_bnw.src.tracktor.utils import interpolate
import os
import numpy as np
import os.path as osp
import csv
import logging
from torchvision.ops import box_iou
import time
from lapsolver import solve_dense
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import average_precision_score
from statistics import mean
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.ops import box_iou
from math import floor

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker():
    def __init__(self, tracker_cfg, encoder, gnn=None, graph_gen=None, proxy_gen=None, 
                    dev=None, net_type='resnet50', test=0, sr_gan=None, output='plain', weight='No',
                    data='tracktor_preprocessed_files.txt'):
        self.net_type = net_type
        self.encoder = encoder
        self.gnn = gnn
        self.graph_gen = graph_gen

        self.tracker_cfg = tracker_cfg
        self.output = output
        self.device = dev

        self.inact_thresh = tracker_cfg['inact_thresh']
        self.act_reid_thresh = tracker_cfg['act_reid_thresh']
        self.inact_reid_thresh = tracker_cfg['inact_reid_thresh']
        self.output_dir = tracker_cfg['output_dir']

        self.data = data
        
        self.get_name(weight)
        print(self.experiment)

        os.makedirs(osp.join(self.output_dir, self.experiment), exist_ok=True)
        if self.tracker_cfg['eval_bb']:
            self.encoder.eval()

        self.distance_ = defaultdict(dict)
        self.interaction = defaultdict(list)
        self.occlusion = defaultdict(list)

        self.store_dist = True

    def track(self, seq, scale=1.0, first=False):
        # sclae LUP by 1000000
        # scale abd by 10
        # scale resnet triplet neck dist by 10
        self.scale = scale
        logger.info("scaling by {}".format(self.scale))
        self.seq = f"Sequence{int(seq.name.split('-')[1])}_" + seq.name.split('-')[2]
        self.thresh_every = True if self.act_reid_thresh == "every" else False
        self.thresh_tbd = True if self.act_reid_thresh == "tbd" else False

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        if self.store_dist:
            self.distance_[self.seq]['inact_dist_same'] = list()
            self.distance_[self.seq]['act_dist_same'] = list()
            self.distance_[self.seq]['inact_dist_diff'] = list()
            self.distance_[self.seq]['act_dist_diff'] = list()
            self.distance_[self.seq]['interaction_mat'] = list()
            self.distance_[self.seq]['occlusion_mat'] = list()
            self.distance_[self.seq]['active_inactive'] = list()
            self.distance_[self.seq]['same_class_mat'] = list()
            self.distance_[self.seq]['dist'] = list()

            self.distance_[self.seq]['visibility_count'] = {-1:0, 0:0, 0.1:0, 0.2:0, 0.3:0, \
                0.4:0, 0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0}

        if self.tracker_cfg['eval_bb'] and not first:
            self.encoder.eval()
        elif first:
            logger.info('Feeding sequence data before tracking once')

        self.mv_avg = dict()
        logger.info("Tracking sequence {} of lenght {}".format(seq.name, seq.num_frames))

        # if random patches should be sampled
        seq.random_patches = self.tracker_cfg['random_patches'] or self.tracker_cfg[
            'random_patches_first'] or self.tracker_cfg['random_patches_several_frames']
        
        i, self.id = 0, 0
        # batch norm experiemnts
        self.normalization_before(seq, first)

        for frame, g, path, boxes, tracktor_ids, gt_ids, vis, random_patches, img_for_det in seq:
            #self.normalization_experiments(random_patches, frame, i)
            ### GET BACKBONE
            frame_id = int(path.split(os.sep)[-1][:-4])
            tracks = list()
            with torch.no_grad():
                if self.net_type == 'resnet50_attention':
                    _, feats, feat_maps = self.encoder(frame, output_option=self.output)
                    if self.gnn:
                        feats = [[fe, fc] for fe, fc in zip(feat_maps, feats)]

                elif self.net_type == 'resnet50_analysis' or self.net_type == 'bot' or \
                    self.net_type == 'batch_drop_block' or self.net_type == 'LUP' or \
                        self.net_type == 'transreid' or self.net_type == 'os':
                    feats = self.encoder(frame)
                elif self.net_type == 'det_bb':
                    feats = self.encoder([img_for_det], [{'public_preds': None, 'boxes': torch.stack([torch.from_numpy(b) for b in boxes]).to(self.device), 'ids': None}])
                    feats = feats[0]['embeddings']
                elif self.net_type == 'abd':
                    feats = self.encoder(frame)[0]
                    #feats = F.normalize(feats, p=2, dim=1)
                elif 'fc_person.weight' in self.encoder.state_dict().keys():
                    # for train mode: if only one sample repeat this sample
                    added = False
                    if frame.shape[0] == 1:
                        added = True
                        frame = torch.cat([frame, frame])

                    _, feats, _ = self.encoder(frame, output_option=self.output)
                    
                    # remove repeated sample again
                    if added:
                        feats = feats[0].unsqueeze(dim=0)
                else:  
                    _, feats = self.encoder(frame, output_option=self.output)
            
            if first:
                continue

            ### iterate over bbs in current frame 
            for f, b, tr_id, gt_id, v in zip(feats, boxes, tracktor_ids, gt_ids, vis):
                if (b[3]-b[1])/(b[2]-b[0]) < self.tracker_cfg['h_w_thresh']:   
                    track = {'bbox': b, 'feats': f, 'im_index': frame_id, \
                        'tracktor_id': tr_id, 'id': gt_id, 'vis': v}           
                    tracks.append(track)

                    if self.store_dist:
                        if v == 1.0:
                            v = 0.999
                        self.distance_[self.seq]['visibility_count'][floor(v*10)/10] += 1

                    #self.tracks[tr_id].append(track)
            #continue

            self.add_ioa(tracks)
            
            ### track
            self._track(tracks, i, frame=frame)  
            i += 1
        
        if first:
            logger.info('Done with pre-tracking feed...')
            return

        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)

        # compute reid performance
        if self.tracker_cfg['get_reid_performance']:
            self.get_reid_performance()

        # get results
        results = self.make_results(seq.name)
        if self.tracker_cfg['interpolate']:
            logger.info("Interpolate")
            results = interpolate(results)
        self.write_results(results, self.output_dir, seq.name)

        if self.store_dist:
            print(self.experiment + 'distances.json')
            import json
            with open(self.experiment + 'distances.json', 'w') as jf:
                json.dump(self.distance_, jf)

        #self.act_reid_thresh = self.inact_reid_thresh = 'every' if self.thresh_every else self.act_reid_thresh
        #self.act_reid_thresh = self.inact_reid_thresh = 'tbd' if self.thresh_tbd else self.act_reid_thresh

    def _track(self, tracks, i, frame=None):
        if i == 0:
            # just add all bbs to self.tracks / intitialize
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
        elif i > 0:
            # get hungarian matching
            if not self.tracker_cfg['each_sample']:
                dist, row, col, ids, dist_l = self.get_hungarian(tracks, sep=self.tracker_cfg['assign_separately'], img_for_vis=frame)
            else:
                dist, row, col, ids, dist_l = self.get_hungarian_all_samps(tracks)

            if type(dist) == list:
                dist[0] = dist[0] * self.scale
            else:
                dist = dist * self.scale

            if dist is not None:
                # get bb assignment
                self.assign(tracks, dist, row, col, ids, dist_l, sep=self.tracker_cfg['assign_separately'])
    
    def box_area(self, boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by their
        (x1, y1, x2, y2) coordinates.

        Args:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format with
                ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Returns:
            Tensor[N]: the area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def _box_inter_area(self, boxes1, boxes2):
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        return inter.numpy(), area1.numpy(), area2.numpy()


    def add_ioa(self, tracks):
        bbs = torch.from_numpy(np.vstack([tr['bbox'] for tr in tracks]))
        inter, area, _ = self._box_inter_area(bbs, bbs)
        ioa = inter / np.atleast_2d(area).T
        ioa = ioa - np.eye(ioa.shape[0])

        # not taking foot position into account --> ioa_nofoot
        self.curr_interaction = ioa
        ioa_nf = np.sum(ioa, axis=1).tolist()
        self.interaction[self.seq] += ioa_nf
        
        for i, t in zip(ioa_nf, tracks):
            t['ioa_nf'] = i

        # taking foot position into account
        bot = np.atleast_2d(bbs[:, 3]).T < np.atleast_2d(bbs[:, 3])
        ioa[~bot] = 0
        self.curr_occlusion = ioa
        ioa = np.sum(ioa, axis=1).tolist()
        self.occlusion[self.seq] += ioa

        for i, t in zip(ioa, tracks):
            t['ioa'] = i

    def get_hungarian_all_samps(self, tracks):
        x = torch.stack([t['feats'] for t in tracks])
        gt_n = [v['id'] for v in tracks]
        gt_t = list()
        dist_all = list()
        ids = list()

        gt_t += [v[-1]['id'] for v in self.tracks.values()]
        if not self.tracker_cfg['avg_act']['do']:
            if len(tracks) > 0:
                y = torch.stack([t[-1]['feats'] for t in self.tracks.values()])
                ids.extend([i for i in self.tracks.keys()])
                dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine').T #'cosine')
                dist_all.extend([d for d in dist])
        else:
            for id, tr in self.tracks.items():
                y = torch.stack([t['feats'] for t in tr])
                ids.append(id)
                dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')#'cosine')
                if self.tracker_cfg['each_sample'] == 1:
                    dist_all.append(np.min(dist, axis=1))
                elif self.tracker_cfg['each_sample'] == 2:
                    dist_all.append(np.mean(dist, axis=1))
                elif self.tracker_cfg['each_sample'] == 3:
                    dist_all.append(np.max(dist, axis=1))
                elif self.tracker_cfg['each_sample'] == 4:
                    dist_all.append(np.max(dist, axis=1)+np.min(dist, axis=1))

        
        num_acite = len(ids)

        # get inactive tracklets (inacht thresh = 100000)
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh}
        gt_t += [v[-1]['id'] for v in curr_it.values()]
        for id, tr in curr_it.items():
            y = torch.stack([t['feats'] for t in tr])
            ids.append(id)
            dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')#'cosine')
            if self.tracker_cfg['each_sample'] == 1:
                dist_all.append(np.min(dist, axis=1))
            elif self.tracker_cfg['each_sample'] == 2:
                dist_all.append(np.mean(dist, axis=1))
            elif self.tracker_cfg['each_sample'] == 3:
                dist_all.append(np.max(dist, axis=1))
            elif self.tracker_cfg['each_sample'] == 4:
                dist_all.append((np.max(dist, axis=1)+np.min(dist, axis=1))/2)
        
        num_inacite = len([k for k, v in curr_it.items()])

        dist = np.vstack(dist_all).T
        row, col = solve_dense(dist)

        if self.store_dist:
            gt_n = np.atleast_2d(np.array(gt_n))
            gt_t = np.atleast_2d(np.array(gt_t))

            same_class = gt_t == gt_n.T
            act = np.atleast_2d(np.array([1] * num_acite + [0] * num_inacite)) == np.atleast_2d(np.ones(dist.shape[0])).T
            
            self.distance_[self.seq]['inact_dist_same'].extend(dist[same_class & ~act].tolist())
            self.distance_[self.seq]['act_dist_same'].extend(dist[same_class & act].tolist())
            self.distance_[self.seq]['inact_dist_diff'].extend(dist[~same_class & ~act].tolist())
            self.distance_[self.seq]['act_dist_diff'].extend(dist[~same_class & act].tolist())
            self.distance_[self.seq]['interaction_mat'].append(self.curr_interaction.tolist())
            self.distance_[self.seq]['occlusion_mat'].append(self.curr_occlusion.tolist())
            self.distance_[self.seq]['active_inactive'].append(act.tolist())
            self.distance_[self.seq]['same_class_mat'].append(same_class.tolist())
            self.distance_[self.seq]['dist'].append(dist.tolist())

        if self.act_reid_thresh == 'tbd' or self.thresh_every:
            print('AM I IN')
            quit()
            gt_n = np.atleast_2d(np.array(gt_n))
            gt_t = np.atleast_2d(np.array(gt_t))

            same_class = gt_t == gt_n.T
            act = np.atleast_2d(np.array([1] * num_acite + [0] * num_inacite)) == np.atleast_2d(np.ones(dist.shape[0])).T

            #self.act_reid_thresh = np.mean(dist[~same_class & act]) - np.std(dist[~same_class & act])
            if self.thresh_every:
                self.act_reid_thresh = np.mean(dist[act]) - 0 * np.std(dist[act])
            elif self.thresh_tbd:
                self.act_reid_thresh = np.mean(dist[act]) - 0.5 * np.std(dist[act])
        
        if (self.inact_reid_thresh == 'tbd' or self.thresh_every) and num_inacite > 0:
            print("I AM HERE")
            quit()
            gt_n = np.atleast_2d(np.array(gt_n))
            gt_t = np.atleast_2d(np.array(gt_t))

            same_class = gt_t == gt_n.T
            act = np.atleast_2d(np.array([1] * num_acite + [0] * num_inacite)) == np.atleast_2d(np.ones(dist.shape[0])).T
            #self.inact_reid_thresh = np.mean(dist[~same_class & ~act]) - np.std(dist[~same_class & ~act])
            if self.thresh_every:
                self.inact_reid_thresh = np.mean(dist[~act]) - 2 * np.std(dist[~act])
            elif self.thresh_tbd:
                self.inact_reid_thresh = np.mean(dist[~act]) - 1 * np.std(dist[~act])

        return dist, row, col, ids, None

    def get_hungarian_distribution(self, tracks):
        x = torch.stack([t['feats'] for t in tracks])
        dist_all = list()
        ids = list()
        for id, tr in self.tracks.items():
            y = torch.stack([t['feats'] for t in tr])
            ids.append(id)
            dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')#'cosine')
            mean = np.mean(dist, axis=1)
            std = np.std(dist, axis=1)
            dist_all.append(np.min(dist, axis=1))
        
        for id, tr in self.inactive_tracks.items():
            y = torch.stack([t['feats'] for t in tr])
            ids.append(id)
            dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')#'cosine')
            dist_all.append(np.max(dist, axis=1))
        
        dist = np.vstack(dist_all).T
        row, col = solve_dense(dist)

        return dist, row, col, ids, None

    def get_hungarian(self, tracks, sep=False, img_for_vis=None):
        if self.net_type == 'resnet50_attention' and self.gnn:
            x_1 = torch.stack([t['feats'][0] for t in tracks])
            x_2 = torch.stack([t['feats'][1] for t in tracks])
            x = [x_1, x_2] # feats, fc7
        else:
            x = torch.stack([t['feats'] for t in tracks])
        num_detects =len(tracks)
        gt_n = [v['id'] for v in tracks]
        ids, gt_t = list(), list()
        y_inactive, y = None, None

        # Get active tracklets
        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = self.avg_it(self.tracks, 'act')
            else:
                if self.net_type == 'resnet50_attention' and self.gnn:
                    y_1 = torch.stack([v[-1]['feats'][0] for v in self.tracks.values()])
                    y_2 = torch.stack([v[-1]['feats'][1] for v in self.tracks.values()])
                    y = [y_1, y_2] # feats, fc7
                else:
                    y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
            
            ids += list(self.tracks.keys())
            gt_t += [v[-1]['id'] for v in self.tracks.values()]
            num_acite = len(ids)

        # get inactive tracklets (inacht thresh = 100000)
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh}

        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = self.avg_it(curr_it, 'inact')
            else:
                if self.net_type == 'resnet50_attention' and self.gnn:
                    y_1 = torch.stack([v[-1]['feats'][0] for v in curr_it.values()])
                    y_2 = torch.stack([v[-1]['feats'][1] for v in curr_it.values()])
                    y_inactive = [y_1, y_2] # feats, fc7
                else:
                    y_inactive = torch.stack([v[-1]['feats'] for v in curr_it.values()])

            if len(self.tracks) > 0 and not sep:
                if self.net_type == 'resnet50_attention' and self.gnn:
                    y = [torch.cat([y[0], y_inactive[0]]), torch.cat([y[1], y_inactive[1]])]
                else:
                    y = torch.cat([y, y_inactive])

            elif not sep:
                y = y_inactive
                num_acite = 0
            
            ids += [k for k, v in curr_it.items()]
            gt_t += [v[-1]['id'] for v in curr_it.values()]
            num_inacite = len([k for k, v in curr_it.items()])

        elif len(curr_it) == 0 and len(self.tracks) == 0:
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
            return None, None, None, None
        
        else:
            num_inacite = 0

        # compute gnn features
        if self.gnn:
            x, y, fc7s = self.get_gnn_feats(x, y, num_detects, img_for_vis=img_for_vis, gt_n=gt_n, gt_t=gt_t)

        # compute distance
        dist_l = list()
        if type(y) == list: # mostly from gnn when several of the past frames are used
            for frame in y:
                d = sklearn.metrics.pairwise_distances(x.cpu().numpy(), frame.cpu().numpy(), metric='cosine')#'cosine')
                dist_l.append(d)
            dist = sum(dist_l)
        else:
            if not sep:
                if self.net_type == 'resnet50_attention' and self.gnn:
                    dist = np.asarray([sklearn.metrics.pairwise_distances(t.unsqueeze(0).cpu().numpy(), \
                        d.cpu().numpy(), metric='cosine') for t, d in zip(y, x)]).squeeze().T #'euclidean')#'cosine')
                    dist = np.atleast_2d(dist)

                    # add dist after bb to spatial dist
                    if self.tracker_cfg['gnn_and_bb']:
                        dist_fc7s = sklearn.metrics.pairwise_distances(fc7s[0].cpu().numpy(), \
                            fc7s[1].cpu().numpy(), metric='cosine')
                        dist_fc7s = np.atleast_2d(dist_fc7s)

                        dist = dist + dist_fc7s
                else:
                    if not self.tracker_cfg['use_bism']:
                        dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')#'cosine')
                    else:
                        dist = 1 - self.bisoftmax(x.cpu(), y.cpu())

                if self.store_dist:
                    gt_n = np.atleast_2d(np.array(gt_n))
                    gt_t = np.atleast_2d(np.array(gt_t))

                    same_class = gt_t == gt_n.T
                    act = np.atleast_2d(np.array([1] * num_acite + [0] * num_inacite)) == np.atleast_2d(np.ones(dist.shape[0])).T
                    
                    self.distance_[self.seq]['inact_dist_same'].extend(dist[same_class & ~act].tolist())
                    self.distance_[self.seq]['act_dist_same'].extend(dist[same_class & act].tolist())
                    #print(dist[same_class & act].tolist(), dist[same_class & ~act].tolist())
                    self.distance_[self.seq]['inact_dist_diff'].extend(dist[~same_class & ~act].tolist())
                    self.distance_[self.seq]['act_dist_diff'].extend(dist[~same_class & act].tolist())
                    self.distance_[self.seq]['interaction_mat'].append(self.curr_interaction.tolist())
                    self.distance_[self.seq]['occlusion_mat'].append(self.curr_occlusion.tolist())
                    self.distance_[self.seq]['active_inactive'].append(act.tolist())
                    self.distance_[self.seq]['same_class_mat'].append(same_class.tolist())
                    self.distance_[self.seq]['dist'].append(dist.tolist())

                if self.act_reid_thresh == 'tbd' or self.thresh_every:
                    print('AM I IN')
                    quit()
                    gt_n = np.atleast_2d(np.array(gt_n))
                    gt_t = np.atleast_2d(np.array(gt_t))

                    same_class = gt_t == gt_n.T
                    act = np.atleast_2d(np.array([1] * num_acite + [0] * num_inacite)) == np.atleast_2d(np.ones(dist.shape[0])).T

                    #self.act_reid_thresh = np.mean(dist[~same_class & act]) - np.std(dist[~same_class & act])
                    if self.thresh_every:
                        self.act_reid_thresh = np.mean(dist[act]) - 0 * np.std(dist[act])
                    elif self.thresh_tbd:
                        self.act_reid_thresh = np.mean(dist[act]) - 0.5 * np.std(dist[act])
                
                if (self.inact_reid_thresh == 'tbd' or self.thresh_every) and num_inacite > 0:
                    print("I AM HERE")
                    quit()
                    gt_n = np.atleast_2d(np.array(gt_n))
                    gt_t = np.atleast_2d(np.array(gt_t))

                    same_class = gt_t == gt_n.T
                    act = np.atleast_2d(np.array([1] * num_acite + [0] * num_inacite)) == np.atleast_2d(np.ones(dist.shape[0])).T
                    #self.inact_reid_thresh = np.mean(dist[~same_class & ~act]) - np.std(dist[~same_class & ~act])
                    if self.thresh_every:
                        self.inact_reid_thresh = np.mean(dist[~act]) - 2 * np.std(dist[~act])
                    elif self.thresh_tbd:
                        self.inact_reid_thresh = np.mean(dist[~act]) - 1 * np.std(dist[~act])
                    
                row, col = solve_dense(dist)
                
                '''print()
                print(np.round(dist, decimals=2))
                print(gt_n, gt_t)
                print(row, col)
                quit()'''

                '''# bi-softmax
                feats = torch.mm(x.cpu(), y.cpu().t())
                d2t_scores = F.softmax(feats, dim=1)
                t2d_scores = feats.softmax(dim=0)
                dist = 1 - ((d2t_scores + t2d_scores) / 2).numpy()
                non_zero = (np.expand_dims(np.array(gt_n), axis=0).T == np.expand_dims(np.array(gt_t), axis=0)).nonzero()
                print(dist)
                print(gt_n, gt_t)
                print(dist[non_zero[0], non_zero[1]])
                quit()'''
            else:
                dist_act = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine') #'euclidean')#'cosine')
                row, col = solve_dense(dist_act)
                if y_inactive is not None:
                    dist_inact = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y_inactive.cpu().numpy(), metric='cosine') #'euclidean')#'cosine')
                else:
                    dist_inact = None
                dist = [dist_act, dist_inact]
            

        # row represent current frame, col represents last frame + inactiva tracks
        #row, col = scipy.optimize.linear_sum_assignment(dist)

        return dist, row, col, ids, dist_l

    def bisoftmax(self, x, y):
        feats = torch.mm(x, y.t())
        d2t_scores = feats.softmax(dim=1)
        t2d_scores = feats.softmax(dim=0)
        scores = (d2t_scores + t2d_scores) / 2
        return scores.numpy()

    def avg_it(self, curr_it, mode='inact'):
        feats = list()
        avg = self.tracker_cfg['avg_' + mode]['num'] 
        proxy = self.tracker_cfg['avg_' + mode]['proxy']
        avg_spat = self.tracker_cfg['avg_' + mode]['num_spat'] 
        proxy_spat = self.tracker_cfg['avg_' + mode]['proxy_spat']

        if self.net_type == 'resnet50_attention' and self.gnn:
            it_1 = {i: [{'feats': t['feats'][0]} for t in it] for i, it in curr_it.items()}
            it_2 = {i: [{'feats': t['feats'][1]} for t in it] for i, it in curr_it.items()}
            curr_its = [it_1, it_2] # feature maps, fc7
            
            avgs = [0 if avg_spat == 'no' else avg_spat, 0 if avg == 'no' else avg]
            proxys = ['last' if proxy_spat == 'no' else proxy_spat, 'last' if proxy == 'no' else proxy]
        else:
            curr_its = [curr_it]
            avgs = [avg]
            proxys = [proxy]

        for avg, proxy, curr_it in zip(avgs, proxys, curr_its):
            if type(avg) != str:
                if proxy != 'mv_avg':
                    avg = int(avg)
                else:
                    avg = float(avg)
            feat = list()
            for i, it in curr_it.items():
                # take all bbs until now
                if proxy == 'last':
                    f = it[-1]['feats']
                elif proxy == 'min_ioa':
                    ioa = [t['ioa'] for t in it]
                    ioa.reverse()
                    f = [t['feats'] for t in it][-(ioa.index(min(ioa))+1)]

                elif avg == 'all' or (proxy != 'frames_gnn' and avg != 'first' and proxy != 'mv_avg' and len(it) < avg):
                    if proxy == 'mean':
                        f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                    elif proxy == 'median':
                        f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                    elif proxy == 'mode':
                        f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]

                # take the last avg
                elif proxy != 'frames_gnn' and avg != 'first' and proxy != 'mv_avg' and len(it) >= avg:
                    if proxy == 'mean':
                        f = torch.mean(torch.stack([t['feats'] for t in it[-avg:]]), dim=0)
                    elif proxy == 'median':
                        f = torch.median(torch.stack([t['feats'] for t in it[-avg:]]), dim=0)[0]
                    elif proxy == 'mode':
                        f = torch.mode(torch.stack([t['feats'] for t in it[-avg:]]), dim=0)[0]
                elif proxy == 'mv_avg':
                    if i not in self.mv_avg.keys():
                        f = it[-1]['feats']
                    else:
                        f = self.mv_avg[i] * avg + it[-1]['feats'] * (1-avg) 
                    self.mv_avg[i] = f
                # take the first only
                elif avg == 'first':
                    f = it[0]['feats']
                # for gnn 
                elif proxy == 'frames_gnn':
                    if type(avg) != int:
                        num = int(avg.split('_')[-1])
                        k = int((len(it)-1)/(num-1))
                        f = [it[i*k]['feats'] for i in range(num)]
                    else:
                        f = [t['feats'] for t in it][-avg:]
                        num = avg
                    c = 1
                    while len(f) < num:
                        f.append(f[-c])
                        c += 1 if c+1 < num else 0
                    f = torch.stack(f)
                feat.append(f)
            feats.append(feat)
        
        res = list()
        for feat in feats:
            if len(feat[0].shape) == 1:
                feat = torch.stack(feat)
            elif len(feat[0].shape) == 3:
                feat = torch.cat([f.unsqueeze(0) for f in feat], dim=0)
            elif len(feat) == 1:
                feat = feat
            else: 
                feat = torch.cat(feat, dim=0)
            res.append(feat)
        if self.net_type != 'resnet50_attention' or (self.net_type == 'resnet50_attention' and not self.gnn):
            return res[0]
        else:
            return res

    def assign_act_inact_same_time(self, row, col, dist, tracks, active_tracks, ids):
        assigned = list()
        for r, c in zip(row, col):
            # assign tracks if reid distance < thresh
            if ids[c] in self.tracks.keys() and dist[r, c] < self.act_reid_thresh:                
                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])
                assigned.append(r)  

            # match w inactive track
            elif ids[c] in self.inactive_tracks.keys() and dist[r, c] < self.inact_reid_thresh:
                # move inactive track to active
                self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                del self.inactive_tracks[ids[c]]
                for i in range(len(self.tracks[ids[c]])):
                    del self.tracks[ids[c]][i]['inact_count']

                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])
                assigned.append(r)  

            '''# new track bc distance too big
            else:
                active_tracks.append(self.id)
                self.tracks[self.id].append(tracks[r])
                self.id += 1'''
        
        return set(assigned)
    
    def assign_separatly(self, row, col, dist, tracks, active_tracks, ids):

        assigned = self.assign_act_inact_same_time(row, col, dist[0], tracks, active_tracks, \
            ids[:dist[0].shape[1]])
        if dist[1] is not None:
            unassigned = sorted(list(set(list(range(dist[0].shape[0]))) - assigned))

            if len(unassigned) != 0:
                dist[1] = dist[1][unassigned, :]

                row_inact, col_inact = solve_dense(dist[1])
                assigned_2 = self.assign_act_inact_same_time(row_inact, col_inact, dist[1], [ t for i, t in enumerate(\
                    tracks) if i in unassigned], active_tracks, ids[dist[0].shape[1]:])
                assigned_2 = set([u for i, u in enumerate(unassigned) if i in assigned_2])
                assigned.update(assigned_2)

        return assigned

    def assign(self, tracks, dist, row, col, ids, dist_list=None, sep=False): 
        # assign tracks from hungarian
        #num = int(self.tracker_cfg['avg_inact']['num']) if self.tracker_cfg['avg_inact']['proxy'] == 'frames_gnn' else 1
        active_tracks = list()
        debug = list()

        if not sep:
            row = self.assign_act_inact_same_time(row, col, dist, tracks, active_tracks, ids)
        else:
            row = self.assign_separatly(row, col, dist, tracks, active_tracks, ids)

        # move tracks not used to inactive tracks
        keys = list(self.tracks.keys())
        for k in keys:
            if k not in active_tracks:
                self.inactive_tracks[k] = self.tracks[k]
                del self.tracks[k]
                for i in range(len(self.inactive_tracks[k])):
                    self.inactive_tracks[k][i]['inact_count'] = 0
        
        # set inactive count one up
        for k in self.inactive_tracks.keys():
            for i in range(len(self.inactive_tracks[k])):    
                self.inactive_tracks[k][i]['inact_count'] += 1

        # tracks that have not been assigned by hungarian
        for i in range(len(tracks)):
            if i not in row:
                self.tracks[self.id].append(tracks[i])
                self.id += 1

    def get_gnn_feats(self, x, y, num_detects, visualize=True, img_for_vis=None, gt_n=None, gt_t=None):
        if self.tracker_cfg['avg_inact']['proxy'] == 'frames_gnn':
            if type(self.tracker_cfg['avg_inact']['num']) == int: 
                num_frames = self.tracker_cfg['avg_inact']['num'] 
            else:
                num_frames = int(self.tracker_cfg['avg_inact']['num'].split('_')[-1])
            feats = torch.cat([x, y])
            with torch.no_grad():
                self.gnn.eval()
                if self.tracker_cfg['gallery_att']:
                    edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats, num_dets=num_detects)
                else:    
                    edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
                _, feats = self.gnn(feats, edge_ind, edge_attr, 'plain')
            x = feats[-1][:num_detects, :]
            y = feats[-1][num_detects:, :]
            y = [torch.stack([y[j] for j in range(i, y.shape[0], num_frames)]) for i in range(num_frames)]
        else:
            # for gnn
            feats = torch.cat([x, y])
            with torch.no_grad():
                self.gnn.eval()
                if self.tracker_cfg['gallery_att']:
                    edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats, num_dets=num_detects)
                else:    
                    edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
                _, feats = self.gnn(feats, edge_ind, edge_attr, 'plain')
            x = feats[-1][:num_detects, :]
            y = feats[-1][num_detects:, :]

            '''
            # for spatial attention
            feats = torch.cat([y[0], x[0]])
            fc7 = y[1]
            fc7s = [x[1], y[1]]
            with torch.no_grad():
                self.gnn.eval()
                _, _, _, qs, gs, attended_feats, _, att_g = self.gnn(feats, num_query=y[0].shape[0])
            

            if visualize:
                att_g = att_g.view(y[0].shape[0], num_detects, att_g.shape[-2], att_g.shape[-1])
                # iterate over all querys
                for i in range(y[0].shape[0]):
                    print(att_g[i])
                    print(gt_t[i], gt_n)
                    quit()
                    visualize_att_map(att_g[i], gt_t[i], gt_n, img_for_vis, save_dir='visualization_attention_maps')
            # if self attention
            # x = attended_feats[y[0].shape[0]:, :].view(y[0].shape[0], num_detects, 2048)
            # y = attended_feats[:y[0].shape[0], :] 

            # if not self attention
            x = attended_feats.view(y[0].shape[0], num_detects, 2048)
            y = fc7'''
        
        return x, y, feats

    def make_results(self, seq_name):
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
        logger.info("Removed {} short tracks".format(len(all_tracks)-len(tracks_new)))
        return tracks_new
    
    def write_results(self, all_tracks, output_dir, seq_name):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
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
        self.experiment = inact_avg  + ':' + str(self.tracker_cfg['inact_reid_thresh']) + \
            ':' + act_avg + ':' + str(self.tracker_cfg['act_reid_thresh'])
        
        if self.tracker_cfg['use_bism']:
            self.experiment += 'bism:1'
        
        if self.gnn:
            self.experiment = 'gnn_' + self.experiment
        
        #import time        
        #self.experiment = '_'.join(['_'.join(time.asctime().split(' ')[1:])[:-5], weight, self.experiment])

        self.experiment = '_'.join([self.data[:-4], weight, 'evalBB:' + str(self.tracker_cfg['eval_bb']), self.experiment])

        logger.info(self.experiment)

    def normalization_experiments(self, random_patches, frame, i):
        ###### Normalization experiments ######
        if self.tracker_cfg['random_patches'] or self.tracker_cfg['random_patches_first'] :
            if i == 0:
                logger.info("Using random patches of current frame")
            if not self.tracker_cfg['random_patches_first'] or i == 0:
                #logger.info("only once")
                self.encoder.train()
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1 # means no running mean
                with torch.no_grad():
                    if 'fc_person.weight' in self.encoder.state_dict().keys():
                        _, feats, _ = self.encoder(random_patches, output_option='plain')
                    else:
                        _, feats = self.encoder(random_patches, output_option='plain')
                self.encoder.eval()
        
        elif self.tracker_cfg['running_mean_seq'] or self.tracker_cfg['running_mean_seq_reset']:
            if self.tracker_cfg['running_mean_seq_reset'] and i == 0:
                logger.info("Resetting BatchNorm statistics")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1
                        m.first_batch_mean = True
            elif  self.tracker_cfg['running_mean_seq_reset'] and i != 0:
                logger.info("Setting mometum to 0.1 again")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 0.1
            self.encoder.train()
            if i == 0:
                logger.info("Using moving average of seq")
            with torch.no_grad():
                if 'fc_person.weight' in self.encoder.state_dict().keys():
                    added = False
                    if frame.shape[0] == 1:
                        added = True
                        frame = torch.cat([frame, frame])

                    _, feats, _ = self.encoder(frame, output_option=self.output)
                    
                    # remove repeated sample again
                    if added:
                        feats = feats[0].unsqueeze(dim=0)
                else:
                    _, feats = self.encoder(frame, output_option='plain')
            self.encoder.eval()

        elif i == 0 and (self.tracker_cfg['first_batch'] or \
            self.tracker_cfg['first_batch_reset']):
            self.encoder.train()
            if self.tracker_cfg['first_batch_reset']:
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1 # means no running mean, i.e., uses first batch wo initialization
                logger.info("Resetting BatchNorm statistics")
            logger.info("Runing first batch in train mode")
            with torch.no_grad():
                if 'fc_person.weight' in self.encoder.state_dict().keys():
                    _, feats, _ = self.encoder(frame, output_option='plain')
                else:
                    _, feats = self.encoder(frame, output_option='plain')
            self.encoder.eval()

        ###################################


    def normalization_before(self, seq, k=5, first=False):
        if first:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()

        i = 0
        if self.tracker_cfg['random_patches_several_frames'] or self.tracker_cfg['several_frames']:
            print("Using {} samples".format(k))
            for frame, g, path, boxes, tracktor_ids, gt_ids, vis, random_patches, img_for_det in seq:
                if self.tracker_cfg['random_patches_several_frames']:
                    inp = random_patches
                else:
                    inp = frame
                self.encoder.train()
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1 # means no running mean
                with torch.no_grad():
                    if 'fc_person.weight' in self.encoder.state_dict().keys():
                        _, _, _ = self.encoder(inp, output_option='plain')
                    else:
                        _, _ = self.encoder(inp, output_option='plain')
                i += 1
                if i >= k:
                    break
            self.encoder.eval()
            seq.random_patches = False


    def get_reid_performance(self, topk=8, first_match_break=True):
        feats = torch.cat([t['feats'].cpu().unsqueeze(0) for k, v in self.tracks.items() for t in v if t['id'] != -1], 0)
        lab = np.array([t['id'] for k, v in self.tracks.items() for t in v if t['id'] != -1])
        dist = sklearn.metrics.pairwise_distances(feats.numpy(), metric='cosine')
        m, n = dist.shape

        # Sort and find correct matches
        indices = np.argsort(dist, axis=1)
        indices = indices[:, 1:]
        matches = (lab[indices] == lab[:, np.newaxis])

        # Compute CMC for each query
        ret = np.zeros(topk)
        num_valid_queries = 0

        for i in range(m):
            if not np.any(matches[i, :]): continue

            index = np.nonzero(matches[i, :])[0]
            delta = 1. / (len(index))
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
            num_valid_queries += 1

        cmc = ret.cumsum() / num_valid_queries
        logger.info("rank-1 {}, rank-5 {}, rank-8 {}".format(cmc[0], cmc[5], cmc[7]))

        aps = []
        for k in range(dist.shape[0]):
            # Filter out the same id and same camera
            y_true = matches[k, :]

            y_score = -dist[k][indices[k]]
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))

        logger.info("mAP {}".format(np.mean(aps)))

#import cv2
def visualize_att_map(attention_maps, qs, gs, img_for_vis, save_dir='visualization_attention_maps'):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import PIL.Image as Image
    # paths[0] = query path, rest gallery paths
    for g, gallery, attention_map in zip(gs, img_for_vis, attention_maps):
        gallery = torch.stack([(img * std[i]) + mean[i] for i, img in enumerate(gallery)])
        gallery = np.transpose(gallery.cpu().numpy(), (1, 2, 0))
        attention_map = cv2.resize(attention_map.squeeze().cpu().numpy(), (gallery.shape[1], gallery.shape[0]))

        cam = show_cam_on_image(gallery, attention_map)        
        
        fig = figure(figsize=(6, 7), dpi=80)

        fig.add_subplot(1,2,1)
        plt.imshow(cv2.cvtColor(gallery, cv2.COLOR_BGR2RGB))

        fig.add_subplot(1,2,2)
        plt.imshow(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
        
        plt.axis('off')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, \
            os.path.basename(str(qs) + '_' + str(g) + '.png')))
    quit()

def show_cam_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
