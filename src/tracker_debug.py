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


logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker():
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

    def track(self, seq):
        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        logger.info("Tracking Sequence {} of length {}".format(seq.name, 
                        seq.num_frames))
        
        i, self.id = 0, 0
        gt = list()
        for frame, g, path, boxes, tracktor_ids, gt_ids in seq:

            frame_id = int(path.split(os.sep)[-1][:-4])
            #logger.info("Tracking frame {} of {}".format(i, seq.name))
            gt.append({'gt': g})
            tracks = list()
            self.encoder.eval()

            with torch.no_grad():
                _, feats = self.encoder(frame, output_option='plain')
            if self.net_type == 'resnet50FPN':
                feats = torch.cat(feats, dim=1)

            for f, b, tr_id, gt_id in zip(feats, boxes, tracktor_ids, gt_ids):
                if (b[3]-b[1])/(b[2]-b[0]) < self.tracker_cfg['h_w_thresh']:
                    tracks.append({'bbox': b, 'feats': f, 'im_index': frame_id, 'tracktor_id': tr_id, 'id': gt_id})
            
            tracks = self._add_iou(tracks)

            if not self.tracker_cfg['init_tracktor']['do']:
                self.track_wo_tracktor(tracks, i)
            else:
                self.track_w_tracktor(tracks)
            
            i += 1
        
        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)

        # get results
        results = self.make_results(seq.name)
        if self.tracker_cfg['interpolate']:
            logger.info("Interpolate")
            results = interpolate(results)
        self.write_results(results, self.output_dir, seq.name)
        self.hidden = dict()

    def _add_iou(self, tracks):
        feats = torch.tensor([t['bbox'] for t in tracks])
        iou_matrix = box_iou(torch.tensor(feats), torch.tensor(feats))
        iou_matrix -= torch.eye(iou_matrix.shape[0])
        max_iou = iou_matrix.max(dim=0)[0]

        for i in range(max_iou.shape[0]):
            tracks[i]['max_iou'] = max_iou[i].item()
        
        return tracks

    def track_wo_tracktor(self, tracks, i):
        if i == 0:
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
        elif i > 0:
            dist, row, col, ids, dist_l = self.get_hungarian(tracks)
            if dist is not None:
                if type(dist) == list:
                    self.assign_majority(tracks, dist, row, col, ids)
                else:
                    self.assign(tracks, dist, row, col, ids, dist_l)

    def get_hungarian(self, tracks):
        x = torch.stack([t['feats'] for t in tracks])
        h_n = [v['bbox'][3] - v['bbox'][1] for v in tracks]
        iou_n = [v['max_iou'] for v in tracks]
        gt_n = [v['id'] for v in tracks]
        f_n = [v['im_index'] for v in tracks]
        num_detects = x.shape[0]

        h_t, iou_t, gt_t, f_t, ids = list(), list(), list(), list(), list()

        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = self.avg_it(self.tracks)
            else:
                y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
            ids += list(self.tracks.keys())

            h_t += [v[-1]['bbox'][3] - v[-1]['bbox'][1] for v in self.tracks.values()]
            iou_t += [v[-1]['max_iou'] for v in self.tracks.values()]
            gt_t += [v[-1]['id'] for v in self.tracks.values()]
            f_t += [v[-1]['im_index'] for v in self.tracks.values()]
        
        # get tracks to compare to
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh}
            
        num_inact = 0
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = self.avg_it(curr_it)
            else:
                y_inactive = torch.stack([v[-1]['feats'] for v in curr_it.values()])
            num_inact = y_inactive.shape
            if len(self.tracks) > 0:
                y = torch.cat([y, y_inactive])
            else:
                y = y_inactive
            
            ids += [k for k, v in curr_it.items()]
            h_t += [v[-1]['bbox'][3] - v[-1]['bbox'][1] for v in curr_it.values()]
            iou_t += [v[-1]['max_iou'] for v in curr_it.values()]
            gt_t += [v[-1]['id'] for v in curr_it.values()]
            f_t += [v[-1]['im_index'] for v in curr_it.values()]

        elif len(curr_it) == 0 and len(self.tracks) == 0:
            # logger.info("No active and no inactive tracks")
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
            return None, None, None, None
        # compute gnn features
        if self.gnn:
            x, y = self.get_gnn_feats(x, y, num_detects)
        dist_l = list()
        if type(y) == list:
            for frame in y:
                d = sklearn.metrics.pairwise_distances(x.cpu().numpy(), frame.cpu().numpy(), metric='cosine')
                #print(d)
                dist_l.append(d)
            
            dist = sum(dist_l)
            #row, col =  scipy.optimize.linear_sum_assignment(dist)
            row, col = solve_dense(dist)
        else:
            dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')
            oracle = self.tracker_cfg['oracle_iou'] or self.tracker_cfg['oracle_size'] or self.tracker_cfg['oracle_frame_dist'] or self.tracker_cfg['oracle_size_diff']
            if oracle:
                def iou_size(n):
                    if (iou_n[n] >= self.tracker_cfg['iou_thresh'] and self.tracker_cfg['oracle_iou']) or \
                        (h_n[n] <= self.tracker_cfg['size_thresh'] and self.tracker_cfg['oracle_size']):
                        self.counter_iou += 1
                        if gt_n[n] in gt_t and gt_n[n] != -1:
                            #self.counter_iou += 1
                            t = gt_t.index(gt_n[n])
                            dist[n, t] = 0
                        else:
                            for t in range(dist.shape[1]):
                                dist[n, t] = 1
                
                def frame_size_diff(n):
                    if self.tracker_cfg['oracle_frame_dist'] or self.tracker_cfg['oracle_size_diff']:
                        if gt_n[n] in gt_t and gt_n[n] != -1:
                            t = gt_t.index(gt_n[n])
                            if abs(h_n[n] - h_t[t]) >= self.tracker_cfg['size_diff_thresh'] and self.tracker_cfg['oracle_size_diff']:
                                self.counter_iou += 1
                                dist[n, t] = 0
                            if abs(f_n[n] - f_t[t]) >= self.tracker_cfg['frame_dist_thresh'] and self.tracker_cfg['oracle_frame_dist']:
                                self.counter_iou += 1
                                dist[n, t] = 0
                        else:
                            for t in range(dist.shape[1]):
                                dist[n, t] = 1

                for n in range(num_detects):
                    iou_size(n)
                    frame_size_diff(n)

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
            if avg == 'all':
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            elif avg == 'rnn':
                pass
            elif avg == 'iou':
                #idx = np.array([t['max_iou'] for t in it]).argmin()
                iou = [t['max_iou'] for t in it]
                idx = np.where(iou == np.amin(iou))[0][-1]
                f = it[idx]['feats']
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
            elif len(it) < avg:
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            #elif avg == 1:
            #    f = it[-1]['feats']
            else:
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it][-avg:]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it][-avg:]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it][-avg:]), dim=0)[0]
            feats.append(f)
            #print(f)
        if len(feats[0].shape) == 1:
            return torch.stack(feats)
        else: 
            return torch.cat(feats, dim=0)

    def get_gnn_feats(self, x, y, num_detects):
        if self.tracker_cfg['avg_inact']['proxy'] == 'frames_gnn':
            if type(self.tracker_cfg['avg_inact']['num']) == int: 
                num_frames = self.tracker_cfg['avg_inact']['num'] 
            else:
                num_frames = int(self.tracker_cfg['avg_inact']['num'].split('_')[-1])
            feats = torch.cat([x, y])
            with torch.no_grad():
                self.gnn.eval() 
                edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
                _, feats = self.gnn(feats, edge_ind, edge_attr, 'plain')
            x = feats[-1][:num_detects, :]
            y = feats[-1][num_detects:, :]
            y = [torch.stack([y[j] for j in range(i, y.shape[0], num_frames)]) for i in range(num_frames)]
        else:
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
    
    def assign(self, tracks, dist, row, col, ids, dist_list=None): 
        # assign tracks from hungarian
        active_tracks = list()
        new_ids = list()

        for r, c in zip(row, col):
            # assign tracks if reid distance < thresh
            
            if ids[c] in self.tracks.keys() and dist[r, c] < self.act_reid_thresh:
                self.tracks[ids[c]].append(tracks[r])
                active_tracks.append(ids[c])                
            # match w inactive track
            elif ids[c] in self.inactive_tracks.keys() and dist[r, c] < self.inact_reid_thresh:
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

    def track_w_tracktor(self, tracks):
        new_tracks_for_matching = list()
        
        # check if already enough elements for graph in current tracks, otherwise: add to tracks
        for tr in tracks:
            tr_id = tr['tracktor_id']
            if tr_id in self.tracks.keys() and len(self.tracks[tr_id]) >= self.tracker_cfg['avg_inact']['num']:
                    new_tracks_for_matching.append(tr)
            else:
                self.tracks[tr_id].append(tr)

        if len(new_tracks_for_matching) == 0:
            return

        x = torch.stack([t['feats'] for t in new_tracks_for_matching])
        num_detects = x.shape[0]

        # check if alreagy enough elements for graph in prior tracks
        tracks_for_matching = {k: v for k, v in self.tracks.items() 
                                if len(v) >= self.tracker_cfg['avg_inact']['num']}

        if len(tracks_for_matching) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y = self.avg_it(tracks_for_matching)
            else:
                y = torch.stack([v[-1]['feats'] for v in tracks_for_matching.values()])
            ids = [k for k in tracks_for_matching] # for j in range(self.tracker_cfg['avg_inact']['num'])]

        # check if already enough elements for graph in inactive tracks
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh 
                                and len(v) >= self.tracker_cfg['avg_inact']['num']}
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y_inactive = self.avg_it(curr_it)
            else:
                y_inactive = torch.stack([v[-1]['feats'] for v in curr_it.values()])
            if len(tracks_for_matching) > 0:
                y = torch.cat([y, y_inactive])
                ids += [k for k, v in curr_it.items()]# for j in range(self.tracker_cfg['avg_inact']['num']-1)]
            else:
                y = y_inactive
                ids = [k for k, v in curr_it.items()]# for j in range(self.tracker_cfg['avg_inact']['num']-1)]

        # compute gnn features for new detectons
        x, y = self.get_gnn_feats(x, y, num_detects)

        dist_l = list()
        for frame in y:
            d = sklearn.metrics.pairwise_distances(x.cpu().numpy(), frame.cpu().numpy(), metric='cosine')
            dist_l.append(d)
        
        #print(d[0], tracks[0]['id'], gt_t)
        dist = sum(dist_l)
        #row, col =  scipy.optimize.linear_sum_assignment(dist)
        row, col = solve_dense(dist)
        self.assign(new_tracks_for_matching, dist, row, col, ids, dist_l)

    def _compute_acc(self, num_frames):
        # tomorrow make df
        frames_dict = defaultdict(list)
        iou_size_id_dict = defaultdict(list)
        other_way_round = dict()
        height = list()
        width = list()
        iou = list()
        cols = ['frame', 'tr_id', 'iou', 'w', 'h']
        for i in range(num_frames):
            for tr_id, tr in self.tracks.items():
                for t in tr:
                    if t['im_index'] == i:
                        if t['id'] not in other_way_round.keys():
                            import pandas as pd
                            other_way_round[t['id']] = pd.DataFrame(columns=cols)
                        other_way_round[t['id']] = other_way_round[t['id']].append(pd.DataFrame([[i, tr_id, t['max_iou'], t['bbox'][2]-t['bbox'][0], t['bbox'][3]-t['bbox'][1]]], columns=cols))
                        height.append(t['bbox'][3]-t['bbox'][1])
                        width.append(t['bbox'][2]-t['bbox'][0])
                        iou.append(t['max_iou'])

        for k, v in other_way_round.items():
            print(v)
        print("Average height: {}".format(sum(height)/len(height)))
        print("Average width: {}".format(sum(width)/len(width)))
        print("Average iou: {}".format(sum(iou)/len(iou)))
        iou = list()
        width = list()
        height = list()
        frame_diff = list()
        import pandas as pd
        pd.set_option('display.max_columns', None)
        cols = ['gt_id', 'iou_l', 'iou_n', 'w_l', 'h_l', 'w_f', 'h_f', 'w_d', 'h_d', 'fr_d']
        df = pd.DataFrame(columns = cols)
        
        del other_way_round[-1]

        for k, v in other_way_round.items():
            for i in range(1, v.shape[0]):
                r_0 = v.iloc[i-1]
                r_1 = v.iloc[i]
                if r_0['tr_id'] != r_1['tr_id']:
                    iou.append([r_0['iou'], r_1['iou']])
                    width.append([r_0['w'], r_1['w']])
                    height.append([r_0['h'], r_1['h']])
                    frame_diff.append(r_0['frame'] - r_1['frame'])
                    df_n = pd.DataFrame([[k, r_0['iou'], r_1['iou'], r_0['w'], r_0['h'], r_1['w'], r_1['h'], r_0['w'] - r_1['w'], r_0['h'] - r_1['h'], r_1['frame'] - r_0['frame']]], columns=cols)
                    df = df.append(df_n)

        df = df[df['gt_id']!= -1]
        
        print(df[df['fr_d']<30])
        print()
        df_n = df[df['fr_d']< 100]
        print(df_n[df_n['fr_d'] > 30])
        print()
        print(df[df['fr_d']> 100])
        print()

        print(df.shape[0]/sum([v.shape[0] for v in other_way_round.values()]))
        print()
        print("Average frame dist {}".format(df['fr_d'].mean())) #sum(frame_diff)/len(frame_diff)))
        print("Average iou last frame {}".format(df['iou_l'].mean()))#sum([v[0] for v in iou])/len([v[0] for v in iou])))
        print("Average iou first frame {}".format(df['iou_n'].mean()))#sum([v[1] for v in iou])/len([v[1] for v in iou])))
        print("Average height last frame {}".format(df['h_l'].mean()))#sum([v[0] for v in height])/len([v[0] for v in height])))
        print("Average height first frame {}".format(df['h_f'].mean()))#sum([v[1]for v in height])/len([v[1] for v in height])))
        print("Average width last frame {}".format(df['w_l'].mean())) #sum([v[0] for v in width])/len([v[0] for v in width])))
        print("Average width first frame {}".format(df['w_f'].mean()))#sum([v[1] for v in width])/len([v[1] for v in width])))
        print()
        for tr_id, tr in self.tracks.items():
            track_dict = list()
            #print("NEW TRACK")
            for i in range(num_frames):
                id_in_frame = 0
                for t in tr:
                    if t['im_index'] == i:
                        iou_size_id_dict[tr_id].append([t['im_index'], t['id'], t['max_iou'], t['bbox'][2]-t['bbox'][0], t['bbox'][3]-t['bbox'][1]])
                        #print(t['id'], t['max_iou'], t['bbox'][2]-t['bbox'][0], t['bbox'][3]-t['bbox'][1], t['bbox'])
                        frames_dict[tr_id].append(t['id'])
                        id_in_frame = 1
                        break
                if not id_in_frame:
                    frames_dict[tr_id].append(0)
        
        #for k, v in iou_size_id_dict.items():
        #    for i in range(1, len(v)):
        #        if v[i-1][1] != v[i][1]:
        #            print(v[i][1], v[i-1][0], v[i-1][1], v[i][0], k)

        import json
        with open(osp.join('out', self.experiment, 'frames_ids.json'), 'w') as f:
            json.dump(frames_dict, f)
        
        a = lambda x: x!= 0
        id_dict = {k: list(filter(a, v)) for k, v in frames_dict.items()}
        
        id_counter_dict = dict()
        for k, v in id_dict.items():
            id_counter_dict[k] = Counter(v)

        ids_per_track = [len(c) for c in id_counter_dict.values()]
        logger.info("Average num of IDs per track {}".format(sum(ids_per_track)/len(ids_per_track)))
        #print(sorted(ids_per_track))
        with open(osp.join('out', self.experiment, 'num_ids_per_track.json'), 'w') as f:
            json.dump(id_counter_dict, f)

        id_switch_dict = dict()
        for k, v in id_dict.items():
            id_switches = 0
            for i in range(1, len(v)):
                if v[i-1] != v[i]:
                    id_switches += 1
            id_switch_dict[k] = id_switches
        logger.info("Average id switches per track {}".format(sum(id_switch_dict.values())/len(id_switch_dict.values())))
        #print(Counter(id_switch_dict.values()))
        with open(osp.join('out', self.experiment, 'id_switches_per_track.json'), 'w') as f:
            json.dump(id_switches, f)
        
        #accumulate 
        GT_ids = [k2 for k, v in id_counter_dict.items() for k2 in v.keys() if k2 != -1]
        gt_ids_counter = Counter(GT_ids)
        logger.info("Average number of tracks per GT ID {}".format(sum(gt_ids_counter.values())/len(gt_ids_counter)))
        #print(Counter(gt_ids_counter.values()))
        with open(osp.join('out', self.experiment, 'tracks_per_id.json'), 'w') as f:
            json.dump(gt_ids_counter, f)
        
        num_ids = len(set([i for ids in id_dict.values() for i in ids]))
        logger.info("{} Tracks generated for {} GT IDs".format(len(id_dict), num_ids))
        #quit()
