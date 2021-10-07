import pandas as pd
from collections import defaultdict, Counter
import torch
import sklearn.metrics
import scipy
from tracking_wo_bnw.src.tracktor.utils import interpolate
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
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker():
    def __init__(self, tracker_cfg, encoder, gnn=None, graph_gen=None, proxy_gen=None, 
                    dev=None, net_type='resnet50', test=0, sr_gan=None):
        self.counter_iou = 0
        self.net_type = net_type
        self.encoder = encoder
        self.gnn = gnn
        self.graph_gen = graph_gen
        self.proxy_gen = proxy_gen
        self.tracker_cfg = tracker_cfg
        self.device = dev
        self.hidden = dict()
        self.position_fails = list()

        self.inact_thresh = tracker_cfg['inact_thresh']
        self.act_reid_thresh = tracker_cfg['act_reid_thresh']
        self.inact_reid_thresh = tracker_cfg['inact_reid_thresh']
        self.output_dir = tracker_cfg['output_dir']
        
        self.get_name()
        os.makedirs(osp.join(self.output_dir, self.experiment), exist_ok=True)

        if self.tracker_cfg['use_sr_gan']:
            self.sr_gan = sr_gan.eval()
        
        if self.tracker_cfg['eval_bb']:
                self.encoder.eval()

    def track(self, seq):

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        logger.info("Tracking sequence {} of lenght {}".format(seq.name, seq.num_frames))
        seq.random_patches = self.tracker_cfg['random_patches']
        
        i, self.id = 0, 0
        for frame, g, path, boxes, tracktor_ids, gt_ids, vis, random_patches in seq:
            self.normalization_experiments(random_patches, frame, i)

            ### GET BACKBONE
            frame_id = int(path.split(os.sep)[-1][:-4])
            tracks = list()
            with torch.no_grad():
                if self.net_type == 'resnet50_attention':
                    _, feats, feat_maps = self.encoder(frame, output_option='plain') 
                    if self.gnn:
                        feats = feat_maps
                elif self.net_type == 'batch_drop_block':
                    feats = self.encoder(frame)
                elif self.net_type == 'bot':
                    feats = self.encoder(frame)
                elif self.net_type == 'resnet50_analysis':
                    feats = self.encoder(frame)
                else:   
                    _, feats = self.encoder(frame, output_option='plain')

            ### iterate over bbs in current frame 
            for f, b, tr_id, gt_id, v in zip(feats, boxes, tracktor_ids, gt_ids, vis):
                if (b[3]-b[1])/(b[2]-b[0]) < self.tracker_cfg['h_w_thresh']:   
                    track = {'bbox': b, 'feats': f, 'im_index': frame_id, \
                        'tracktor_id': tr_id, 'id': gt_id, 'vis': v}           
                    tracks.append(track)
            
            ### track
            self._track(tracks, i)            
            i += 1

        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)
        logger.info("Position fails {}".format(len(self.position_fails)))

        # compute reid performance
        if self.tracker_cfg['get_reid_performance']:
            self.get_reid_performance()

        # get results
        results = self.make_results(seq.name)
        if self.tracker_cfg['interpolate']:
            logger.info("Interpolate")
            results = interpolate(results)
        self.write_results(results, self.output_dir, seq.name)
        self.hidden = dict()

    def _track(self, tracks, i, writer):
        if i == 0:
            # just add all bbs to self.tracks / intitialize
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
        elif i > 0:
            # get hungarian matching
            dist, row, col, ids, dist_l = self.get_hungarian(tracks, writer)
            if dist is not None:
                # get bb assignment
                self.assign(tracks, dist, row, col, ids, dist_l)

    def get_hungarian(self, tracks):
        x = torch.stack([t['feats'] for t in tracks])
        num_detects = x.shape[0]
        gt_n = [v['id'] for v in tracks]
        ids, gt_t = list(), list()

        # Get active tracklets
        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = self.avg_it(self.tracks)
            else:
                y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
            
            ids += list(self.tracks.keys())
            gt_t += [v[-1]['id'] for v in self.tracks.values()]

        # get inactive tracklets
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh}
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = self.avg_it(curr_it)
            else:
                y_inactive = torch.stack([v[-1]['feats'] for v in curr_it.values()])

            if len(self.tracks) > 0:
                y = torch.cat([y, y_inactive])
            else:
                y = y_inactive
            
            ids += [k for k, v in curr_it.items()]
            gt_t += [v[-1]['id'] for v in curr_it.values()]

        elif len(curr_it) == 0 and len(self.tracks) == 0:
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
            return None, None, None, None
        
        # compute gnn features
        if self.gnn:
            x, y = self.get_gnn_feats(x, y, num_detects)

        # compute distance
        dist_l = list()
        if type(y) == list: # mostly from gnn when several of the past frames are used
            for frame in y:
                d = sklearn.metrics.pairwise_distances(x.cpu().numpy(), frame.cpu().numpy(), metric='cosine')#'cosine')
                dist_l.append(d)
            dist = sum(dist_l)
        else:
            dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine') #'euclidean')#'cosine')

        # row represent current frame, col represents last frame + inactiva tracks
        # row, col = scipy.optimize.linear_sum_assignment(dist)
        row, col = solve_dense(dist)

        return dist, row, col, ids, dist_l

    def avg_it(self, curr_it):
        feats = list()
        avg = self.tracker_cfg['avg_inact']['num'] 

        if type(avg) != str:
            avg = int(avg)

        for i, it in curr_it.items():
            # take all bbs until now
            if avg == 'all':
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            # take the last avg
            elif not self.tracker_cfg['avg_inact']['proxy'] == 'frames_gnn' and avg != 'first':
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            elif avg == 'moving_avg':
                pass
            # take the first only
            elif avg == 'first':
                f = it[0]['feats']
            # for gnn 
            elif self.tracker_cfg['avg_inact']['proxy'] == 'frames_gnn':
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
            
            feats.append(f)

        if len(feats[0].shape) == 1:
            return torch.stack(feats)
        if len(feats[0].shape) == 3:
            return torch.cat([f.unsqueeze(0) for f in feats], dim=0)
        else: 
            return torch.cat(feats, dim=0)

    def assign(self, tracks, dist, row, col, ids, dist_list=None): 
        # assign tracks from hungarian
        #num = int(self.tracker_cfg['avg_inact']['num']) if self.tracker_cfg['avg_inact']['proxy'] == 'frames_gnn' else 1
        active_tracks = list()
        debug = list()
        new_ids = list()

        for r, c in zip(row, col):
            # assign tracks if reid distance < thresh
            if ids[c] in self.tracks.keys() and dist[r, c] < self.act_reid_thresh:                
                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])      

            # match w inactive track
            elif ids[c] in self.inactive_tracks.keys() and dist[r, c] < self.inact_reid_thresh:                
                # move inactive track to active
                self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                del self.inactive_tracks[ids[c]]
                for i in range(len(self.tracks[ids[c]])):
                    del self.tracks[ids[c]][i]['inact_count']

                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])

            # new track bc distance too big
            else:
                active_tracks.append(self.id)
                self.tracks[self.id].append(tracks[r])
                new_ids.append([self.id, tracks[r]['id']])
                self.id += 1
                
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
                new_ids.append([self.id, tracks[i]['id']])
                self.id += 1
            
    def get_gnn_feats(self, x, y, num_detects):
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
            #print("just concat x and y")
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
        
        return x, y

    def make_results(self, seq_name):
        results = defaultdict(dict)
        cols = ['frame', 'id', 'my_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
        df = pd.DataFrame(columns=cols)
        for i, ts in self.tracks.items():
            #if len(ts) <= self.tracker_cfg['length_thresh']:
            #    print([{k: v for k, v in d.items() if k != 'feats'} for d in ts])
            showed = 0
            for t in ts:
                if t['id'] == -1 and showed == 0:
                    for d in ts:
                        df = df.append(pd.DataFrame([[d['im_index'], d['id'], i, d['bbox'][0]+1, d['bbox'][1]+1, d['bbox'][2]-d['bbox'][0],d['bbox'][3]-d['bbox'][1]]], columns=cols))
                    #print([{k: v for k, v in d.items() if k != 'feats'} for d in ts])
                    #print([d['id'] for d in ts])
                    showed = 1
                results[i][t['im_index']] = t['bbox']
        df.to_csv('checker_files/' + seq_name + 'checker.csv')
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


    def get_name(self):
        inact_avg = self.tracker_cfg['avg_inact']['proxy'] + '_' + str(
            self.tracker_cfg['avg_inact']['num']) if self.tracker_cfg['avg_inact']['do'] else 'last_frame'
        act_avg = self.tracker_cfg['avg_act']['proxy'] + '_' + str(
            self.tracker_cfg['avg_act']['num']) if self.tracker_cfg['avg_act']['do'] else 'last_frame'
        self.experiment = str(self.inact_thresh) + '_' + inact_avg  + '_' + str(
            self.tracker_cfg['inact_reid_thresh']) + '_' + act_avg + '_' + str(self.tracker_cfg['act_reid_thresh'])
        
        if self.gnn:
            self.experiment = 'gnn_' + self.experiment
        if self.tracker_cfg['init_tracktor']['do']:
            self.experiment = 'init_tracktor_' + self.experiment
        if self.tracker_cfg['interpolate']:
            self.experiment = 'interpolate_' + self.experiment
        if self.tracker_cfg['oracle_iou']:
            self.experiment = 'oracle_iou' + str(self.tracker_cfg['iou_thresh']) + self.experiment
        if self.tracker_cfg['oracle_size']:
            self.experiment = 'oracle_size' + str(self.tracker_cfg['size_thresh']) + self.experiment
        if self.tracker_cfg['oracle_frame_dist']:
            self.experiment = 'oracle_frame_dist' + str(self.tracker_cfg['frame_dist_thresh']) + self.experiment
        if self.tracker_cfg['oracle_size_diff']:
            self.experiment = 'oracle_size_diff' + str(self.tracker_cfg['size_diff_thresh']) + self.experiment
        
        import time        
        t = time.asctime().split(' ')[1:2] + time.asctime().split(' ')[3:]

        self.experiment = '_'.join(t) + self.experiment

        if debug_dist_consecutive_frames:
            self.avg_dists = dict()
            for mode in ['act', 'inact']:
                self.avg_dists['same_class_' + mode] = list()
                self.avg_dists['diff_class_' + mode] = list()
                self.avg_dists['min_same_class_' + mode] = list()
                self.avg_dists['max_same_class_' + mode] = list()
                self.avg_dists['min_diff_class_' + mode] = list()
                self.avg_dists['max_diff_class_' + mode] = list()

    def normalization_experiment(self, random_patches, frame, i):
        ###### Normalization experiments ######
        if self.tracker_cfg['random_patches']:
            if i == 0:
                logger.info("Using random patches of current frame")
            if not self.tracker_cfg['random_patches_first'] or i == 0:
                logger.info("only once")
                self.encoder.train()
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1
                with torch.no_grad():
                    _, feats = self.encoder(random_patches, output_option='plain')
                self.encoder.eval()
        
        elif self.tracker_cfg['running_mean_seq'] or self.tracker_cfg['running_mean_seq_reset']:
            if self.tracker_cfg['running_mean_seq_reset'] and i == 0:
                logger.info("Resetting BatchNorm statistics")
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.first_batch_mean = True
            self.encoder.train()
            if i == 0:
                logger.info("Using moving average of seq")
            with torch.no_grad():
                _, feats = self.encoder(frame, output_option='plain')
            self.encoder.eval()

        elif i == 0 and (self.tracker_cfg['first_batch'] or \
            self.tracker_cfg['first_batch_reset']):
            self.encoder.train()
            if self.tracker_cfg['first_batch_reset']:
                for m in self.encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = 1
                logger.info("Resetting BatchNorm statistics")
            logger.info("Runing first batch in train mode")
            with torch.no_grad():
                _, feats = self.encoder(frame, output_option='plain')
            self.encoder.eval()

        ###################################

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
