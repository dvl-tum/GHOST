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
from scipy.spatial.distance import cdist


logger = logging.getLogger('AllReIDTracker.Tracker')


class TrackerQG():
    def __init__(self, tracker_cfg, encoder, gnn=None, graph_gen=None, proxy_gen=None, dev=None, net_type='resnet50', test=0, sr_gan=None):
        self.counter_iou = 0
        self.net_type = net_type
        self.encoder = encoder
        self.gnn = gnn
        self.graph_gen = graph_gen
        self.proxy_gen = proxy_gen
        self.tracker_cfg = tracker_cfg
        self.device = dev
        self.hidden = dict()

        self.inact_thresh = tracker_cfg['inact_thresh']
        self.act_reid_thresh = tracker_cfg['act_reid_thresh']
        self.inact_reid_thresh = tracker_cfg['inact_reid_thresh']
        self.output_dir = tracker_cfg['output_dir']
        
        print(self.tracker_cfg['avg_inact']['proxy'], str(self.tracker_cfg['avg_inact']['num']))
        inact_avg = self.tracker_cfg['avg_inact']['proxy'] + '_' + str(self.tracker_cfg['avg_inact']['num']) if self.tracker_cfg['avg_inact']['do'] else 'last_frame'
        act_avg = self.tracker_cfg['avg_act']['proxy'] + '_' + str(self.tracker_cfg['avg_act']['num']) if self.tracker_cfg['avg_act']['do'] else 'last_frame'
        self.experiment = str(self.inact_thresh) + '_' + inact_avg  + '_' + str(self.tracker_cfg['inact_reid_thresh']) + '_' + act_avg + '_' + str(self.tracker_cfg['act_reid_thresh'])
        
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
        if test:
            self.experiment = 'test_' + self.experiment

        os.makedirs(osp.join(self.output_dir, self.experiment), exist_ok=True)

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)

        if self.tracker_cfg['use_sr_gan']:
            self.sr_gan = sr_gan.eval()

    def track(self, seq):
        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        logger.info("Tracking Sequence {} of length {}".format(seq.name, 
                        seq.num_frames))
        
        i, self.id = 0, 0
        gt = list()
        for frame, g, path, boxes, tracktor_ids, gt_ids in seq:
            logger.info("Frame {}".format(i))
            if self.tracker_cfg['use_sr_gan']:
                with torch.no_grad():
                    frame = self.sr_gan(frame)

            frame_id = int(path.split(os.sep)[-1][:-4])
            gt.append({'gt': g})
            tracks = list()
            self.encoder.eval()

            # feats = bounding box images
            for f, b, tr_id, gt_id in zip(frame, boxes, tracktor_ids, gt_ids):
                if (b[3]-b[1])/(b[2]-b[0]) < self.tracker_cfg['h_w_thresh']:
                    tracks.append({'bbox': b, 'feats': f, 'im_index': frame_id, 'tracktor_id': tr_id, 'id': gt_id})
            tracks = self._add_iou(tracks)
            print([t['id'] for t in tracks])

            if i > 0:
                feats = []
                ids = []
                gt_ids_last = []
                gt_ids = []
                #print(len(self.tracks), len(self.inactive_tracks))
                for trks in [self.tracks, self.inactive_tracks]:
                    for k, t in trks.items():
                        iou = np.argmin([f['max_iou'] for f in t])
                        idx = np.where(iou == np.amin(iou))[0][-1]
                        print(idx, len(t))
                        feats.append(t[idx]['feats'])
                        ids.append(k)
                        gt_ids_last.append(t[-1]['id'])
                        gt_ids.append(t[idx]['id'])
                feats = torch.cat([f.unsqueeze(0) for f in feats], dim=0)
                dets = [torch.cat([d.unsqueeze(0)] * feats.shape[0], dim=0) for d in frame]
                with torch.no_grad():
                    self.encoder.eval()
                    detections = list()
                    for d in dets:
                        detections.append(self.encoder(d, feats))
                    feats = self.encoder(feats, feats)
                    
                    #for d in frame:
                    #    dets = torch.cat([self.encoder(d.unsqueeze(0), f.unsqueeze(0)) for f in feats], dim=0)
                    #    detections.append(dets)
                    #feats = torch.cat([self.encoder(f.unsqueeze(0), f.unsqueeze(0)) for f in feats], dim=0)
                
                # detections: len = # detections, elements of detections = each feat as query once
                self.track_wo_tracktor(tracks, i, feats, detections, ids)
                print(gt_ids)
                print(gt_ids_last)
                #quit()
            else:
                for t in tracks:
                    self.tracks[self.id].append(t)
                    self.id += 1
            i += 1
        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)
        if self.tracker_cfg['get_reid_performance']:
            self.get_reid_performance()
        #self._compute_acc(seq.num_frames)

        # get results
        results = self.make_results(seq.name)
        if self.tracker_cfg['interpolate']:
            logger.info("Interpolate")
            results = interpolate(results)
        self.write_results(results, self.output_dir, seq.name)
        self.hidden = dict()

    def get_reid_performance(self, topk=8, first_match_break=True):
        feats = torch.cat([t['feats'].cpu().unsqueeze(0) for k, v in self.tracks.items() for t in v if t['id'] != -1], 0)
        lab = np.array([t['id'] for k, v in self.tracks.items() for t in v if t['id'] != -1])
        dist = sklearn.metrics.pairwise_distances(feats.numpy(), metric='euclidean')
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

    def _add_iou(self, tracks):
        feats = torch.tensor([t['bbox'] for t in tracks])
        iou_matrix = box_iou(torch.tensor(feats), torch.tensor(feats))
        iou_matrix -= torch.eye(iou_matrix.shape[0])
        max_iou = iou_matrix.max(dim=0)[0]

        for i in range(max_iou.shape[0]):
            tracks[i]['max_iou'] = max_iou[i].item()
        
        return tracks

    def track_wo_tracktor(self, tracks, i, feats, dets, ids):
        dist, row, col = self.get_hungarian(feats, dets)
        if dist is not None:
            self.assign(tracks, dist, row, col, ids)

    def get_hungarian(self, feats, dets):
        dists = list()
        for i in range(feats.shape[0]):
            det = np.array([d[i].cpu().numpy() for d in dets])
            dist = sklearn.metrics.pairwise_distances(det, feats[i].unsqueeze(0).cpu().numpy(), metric='cosine')
            dists.append(dist)
        dist = np.concatenate(dists, axis=1)
        #print(dist)
        # row represent current frame, col represents last frame + inactiva tracks
        # row, col = scipy.optimize.linear_sum_assignment(dist)
        row, col = solve_dense(dist)
        #print(row, col)

        return dist, row, col
    
    def assign(self, tracks, dist, row, col, ids): 
        # assign tracks from hungarian
        active_tracks = list()
        debug = list()
        new_ids = list()
        
        for r, c in zip(row, col):
            # assign tracks if reid distance < thresh
            # match w active track  
            if ids[c] in self.tracks.keys() and dist[r, c] < self.act_reid_thresh:
                if tracks[r]['id'] != self.tracks[ids[c]][-1]['id']:
                    debug.append(['act', tracks[r]['id'], self.tracks[ids[c]][-1]['id'], ids[c], dist[r, c], tracks[r]['max_iou'], self.tracks[ids[c]][-1]['max_iou'], tracks[r]['im_index'], self.tracks[ids[c]][-1]['im_index'], tracks[r]['bbox'][1], self.tracks[ids[c]][-1]['bbox'][1]])
                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])                
            # match w inactive track
            elif ids[c] in self.inactive_tracks.keys() and dist[r, c] < self.inact_reid_thresh:
                if tracks[r]['id'] != self.inactive_tracks[ids[c]][-1]['id']:
                    debug.append(['inact', tracks[r]['id'], self.inactive_tracks[ids[c]][-1]['id'], ids[c], dist[r, c], tracks[r]['max_iou'], self.inactive_tracks[ids[c]][-1]['max_iou'], tracks[r]['im_index'], self.inactive_tracks[ids[c]][-1]['im_index'], tracks[r]['bbox'][1], self.inactive_tracks[ids[c]][-1]['bbox'][1]])
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

        #if len(debug) > 0:
        #    print(debug)

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
        '''
        if len(new_ids) > 0:
            new_i = [j[0] for j in new_ids]
            real_i = [j[1] for j in new_ids]

            real_it = [v[-1]['id'] for k, v  in self.inactive_tracks.items() if v[-1]['id'] in real_i]
            it = [k for k, v  in self.inactive_tracks.items() if v[-1]['id'] in real_i]
            for j, k in zip(real_it, it):
                n = new_i[real_i.index(j)]
                print('inact', self.tracks[n][-1]['id'], self.inactive_tracks[k][-1]['id'], self.tracks[n][-1]['max_iou'], self.inactive_tracks[k][-1]['max_iou'], self.tracks[n][-1]['im_index'], self.inactive_tracks[k][-1]['im_index'], self.tracks[n][-1]['bbox'][1], self.inactive_tracks[k][-1]['bbox'][1])

            real_it = [v[-1]['id'] for k, v  in self.tracks.items() if v[-1]['id'] in real_i]
            it = [k for k, v  in self.tracks.items() if v[-1]['id'] in real_i]
            for j, k in zip(real_it, it):
                n = new_i[real_i.index(j)]
                if k != n:
                    print('act', self.tracks[n][-1]['id'], self.tracks[k][-1]['id'], self.tracks[n][-1]['max_iou'], self.tracks[k][-1]['max_iou'], self.tracks[n][-1]['im_index'], self.tracks[k][-1]['im_index'], self.tracks[n][-1]['bbox'][1], self.tracks[k][-1]['bbox'][1])
        '''        

    def assign_majority(self, tracks, dist_l, row_l, col_l, ids): 
        # assign tracks from hungarian
        votes = defaultdict(list)
        for r, tr in enumerate(tracks):
            for dist, row, col in zip(dist_l, row_l, col_l):
                if r in row:
                    c = int(col[np.where(row==r)[0]])
                    if ids[c] in self.tracks.keys() and dist[r, c] < self.act_reid_thresh:
                        votes[r].append(ids[c])
                    # match w inactive track
                    elif ids[c] in self.inactive_tracks.keys() and dist[r, c] < self.inact_reid_thresh:
                        votes[r].append(ids[c])
                    else:
                        votes[r].append(-1)
                else:
                    votes[r].append(-1)
        votes_maj = {r: max(set(v), key = v.count) for r, v in votes.items()}
        # label set from votes
        labs = set([v2 for k, v in votes.items() for v2 in v if v2 != -1])
        # label vote counter
        counters = {k: Counter(v) for k, v in votes.items() if votes_maj[k] != -1 }
        # distance computation k x labs
        
        active_tracks = list()
        if len(counters) != 0:
            dist = [[1 - c[l]/sum(c.values()) for l in labs] for k, c in counters.items()]
            dist = torch.tensor(dist)
            row, col = scipy.optimize.linear_sum_assignment(dist)

            active_tracks = list()
            for r, v in zip(row, col):
                r, v = list(counters.keys())[r], list(labs)[v]
                if v in self.tracks.keys():
                    self.tracks[v].append(tracks[r])
                    active_tracks.append(v)
                elif v in self.inactive_tracks.keys():
                    self.tracks[v] = self.inactive_tracks[v]
                    del self.inactive_tracks[v]
                    for i in range(len(self.tracks[v])):
                        del self.tracks[v][i]['inact_count']
                
                    self.tracks[v].append(tracks[r])
                    active_tracks.append(v)

        for k, v in votes_maj.items():
            if v == -1:
                self.tracks[self.id].append(tracks[k])
                self.id += 1
        # move tracks not used to inactive tracks
        keys = list(self.tracks.keys())
        for k in keys:
            if k not in active_tracks:
                self.inactive_tracks[k] = self.tracks[k]
                del self.tracks[k]
                #print(self.inactive_tracks.keys())
                for i in range(len(self.inactive_tracks[k])):
                    self.inactive_tracks[k][i]['inact_count'] = 0
        
        # set inactive count one up
        for k in self.inactive_tracks.keys():
            for i in range(len(self.inactive_tracks[k])): 
                self.inactive_tracks[k][i]['inact_count'] += 1

        # tracks that have not been assigned by hungarian
        for i in range(len(tracks)):
            if i not in votes_maj.keys():
                self.tracks[self.id].append(tracks[i])
                self.id += 1

    def make_results(self, seq_name):
        results = defaultdict(dict)
        cols = ['frame', 'id', 'my_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
        df = pd.DataFrame(columns=cols)
        for i, ts in self.tracks.items():
            showed = 0
            for t in ts:
                if t['id'] == -1 and showed == 0:
                    for d in ts:
                        df = df.append(pd.DataFrame([[d['im_index'], d['id'], i, d['bbox'][0]+1, d['bbox'][1]+1, d['bbox'][2]-d['bbox'][0],d['bbox'][3]-d['bbox'][1]]], columns=cols))
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

    