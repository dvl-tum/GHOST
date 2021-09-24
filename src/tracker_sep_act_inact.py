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

logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker():
    def __init__(self, tracker_cfg, encoder, gnn=None, graph_gen=None, proxy_gen=None, dev=None, net_type='resnet50'):
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
        t = time.asctime().split(' ')[1:3] + time.asctime().split(' ')[3:]

        self.experiment = '_'.join(t) + self.experiment
        print(self.experiment)

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
            gt.append({'gt': g})
            tracks = list()
            with torch.no_grad():
                self.encoder.eval()
                _, feats = self.encoder(frame, output_option='plain')
            if self.net_type == 'resnet50FPN':
                feats = torch.cat(feats, dim=1)

            for f, b, tr_id, gt_id in zip(feats, boxes, tracktor_ids, gt_ids):
                if (b[3]-b[1])/(b[2]-b[0]) < self.tracker_cfg['h_w_thresh']:
                    tracks.append({'bbox': b, 'feats': f, 'im_index': frame_id, 'tracktor_id': tr_id, 'id': gt_id})
            
            tracks = self._add_iou(tracks)

            self.track_wo_tracktor(tracks, i)
            
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
            act, inact = self.get_act_inact()
            if len(inact[2]) == 0 and len(self.tracks) == 0:
                for tr in tracks:
                    self.tracks[self.id].append(tr)
                    self.id += 1
                return
            # assign act
            if act[1]:
                dist, row, col, ids = self.get_hungarian(tracks, act[0], act[1], act[2], avg_dict=self.tracker_cfg['avg_act'])
                #print(dist)
                if dist is not None:
                    tracks, active_tracks = self.assign(tracks, dist, row, col, ids, avg_dict=self.tracker_cfg['avg_act'])
            else:
                active_tracks = list()

            if len(tracks) > 0 and inact[1]:
                dist, row, col, ids = self.get_hungarian(tracks, inact[0], inact[1], inact[3], inact=True, avg_dict=self.tracker_cfg['avg_inact'])
                if dist is not None:
                    tracks, active_tracks = self.assign(tracks, dist, row, col, ids, inact[2], active_tracks, avg_dict=self.tracker_cfg['avg_inact'])

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
                for j in range(len(self.inactive_tracks[k])):    
                    self.inactive_tracks[k][j]['inact_count'] += 1

            # tracks that have not been assigned by hungarian
            new_ids = list()
            for j in range(len(tracks)):
                self.tracks[self.id].append(tracks[j])
                new_ids.append([self.id, tracks[j]['id']])
                self.id += 1

            '''
            if len(new_ids) > 0:
                new_i = [j[0] for j in new_ids]
                real_i = [j[1] for j in new_ids]

                real_it = [v[-1]['id'] for k, v  in self.inactive_tracks.items() if v[-1]['id'] in real_i]
                it = [k for k, v  in self.inactive_tracks.items() if v[-1]['id'] in real_i]
                for j, k in zip(real_it, it):
                    n = new_i[real_i.index(j)]
                    print("frame {}".format(i))
                    print('inact', self.tracks[n][-1]['id'], self.inactive_tracks[k][-1]['id'], self.tracks[n][-1]['max_iou'], self.inactive_tracks[k][-1]['max_iou'], self.tracks[n][-1]['im_index'], self.inactive_tracks[k][-1]['im_index'], self.tracks[n][-1]['bbox'][1], self.inactive_tracks[k][-1]['bbox'][1])
                
                real_it = [v[-1]['id'] for k, v  in self.tracks.items() if v[-1]['id'] in real_i]
                it = [k for k, v  in self.tracks.items() if v[-1]['id'] in real_i]
                for j, k in zip(real_it, it):
                    n = new_i[real_i.index(j)]
                    if k != n:
                        print("frame {}".format(i))
                        print('act', self.tracks[n][-1]['id'], self.tracks[k][-1]['id'], self.tracks[n][-1]['max_iou'], self.tracks[k][-1]['max_iou'], self.tracks[n][-1]['im_index'], self.tracks[k][-1]['im_index'], self.tracks[n][-1]['bbox'][1], self.tracks[k][-1]['bbox'][1])
            
            new_i = [j[1] for j in new_ids]
            real_i = [j[1] for j in new_ids]
            it = [v[-1]['id'] for k, v  in self.inactive_tracks.items() if v[-1]['id'] in new_i]
            t = [v[-1]['id'] for k, v  in self.tracks.items() if v[-1]['id'] in new_i and k != real_i[new_i.index(v[-1]['id'])] != k]
            if len(it) > 0 or len(t) > 0:
                print("Problemo frame {}".format(i))
                print(new_ids, it, t)
            '''

    def get_act_inact(self):

        if len(self.tracks) > 0:
            if self.tracker_cfg['avg_act']['do']:
                y_act = self.avg_it(self.tracks, self.tracker_cfg['avg_act'])
            else:
                y_act = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
            
            ids_act = list(self.tracks.keys())

            h_ta = [v[-1]['bbox'][3] - v[-1]['bbox'][1] for v in self.tracks.values()]
            iou_ta = [v[-1]['max_iou'] for v in self.tracks.values()]
            gt_ta = [v[-1]['id'] for v in self.tracks.values()]
            f_ta = [v[-1]['im_index'] for v in self.tracks.values()]
        else:
            y_act = ids_act = h_ta = iou_ta = gt_ta = f_ta = None

        # get tracks to compare to
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh}
            
        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = self.avg_it(curr_it, self.tracker_cfg['avg_inact'])
            else:
                y_inactive = torch.stack([v[-1]['feats'] for v in curr_it.values()])
            
            ids_inact = [k for k, v in curr_it.items()]
            h_ti = [v[-1]['bbox'][3] - v[-1]['bbox'][1] for v in curr_it.values()]
            iou_ti = [v[-1]['max_iou'] for v in curr_it.values()]
            gt_ti= [v[-1]['id'] for v in curr_it.values()]
            f_ti = [v[-1]['im_index'] for v in curr_it.values()]
        else:
            y_inactive = ids_inact = h_ti = iou_ti = gt_ti = f_ti = None

        return (y_act, ids_act, (h_ta, iou_ta, gt_ta, f_ta)), (y_inactive, ids_inact, curr_it, (h_ti, iou_ti, gt_ti, f_ti))

    def get_hungarian(self, tracks, y, ids, debug_t=None, inact=False, avg_dict=None):
        from copy import deepcopy as dc
        x = torch.stack([t['feats'] for t in tracks])
        h_n = [v['bbox'][3] - v['bbox'][1] for v in tracks]
        iou_n = [v['max_iou'] for v in tracks]
        gt_n = [v['id'] for v in tracks]
        f_n = [v['im_index'] for v in tracks]
        debug_n = (h_n, iou_n,  gt_n, f_n)
        num_detects = x.shape[0]

        dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')
        # compute gnn features
        if self.gnn: # and not inact:
            x, y = self.get_gnn_feats(x, y, num_detects, avg_dict)
        dist_l = list()
        if type(y) == list:
            for frame in y:
                d = sklearn.metrics.pairwise_distances(x.cpu().numpy(), frame.cpu().numpy(), metric='cosine')
                dist_l.append(d)
            dist = sum(dist_l)
            #row, col =  scipy.optimize.linear_sum_assignment(dist)
            row, col = solve_dense(dist)
        else:
            dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy(), metric='cosine')
            oracle = self.tracker_cfg['oracle_iou'] or self.tracker_cfg['oracle_size'] or self.tracker_cfg['oracle_frame_dist'] or self.tracker_cfg['oracle_size_diff']
            if oracle:
                dist = self.go_oracle(debug_t, debug_n, dist, num_detects)
            # row represent current frame, col represents last frame + inactiva tracks
            # row, col = scipy.optimize.linear_sum_assignment(dist)
            row, col = solve_dense(dist)

        return dist, row, col, ids

    def go_oracle(self, debug_t, debug_n, dist, num_detects):
        h_n, iou_n,  gt_n, f_n = debug_n
        h_t, iou_t, gt_t, f_t = debug_t

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

        return dist

    def avg_it(self, curr_it, avg_dict):
        feats = list()
        avg = avg_dict['num'] 
        if type(avg) != str:
            avg = int(avg)

        for i, it in curr_it.items():
            if avg == 'all':
                if avg_dict['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif avg_dict['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif avg_dict['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            elif avg == 'rnn':
                pass
            elif avg == 'iou':
                idx = np.array([t['max_iou'] for t in it]).argmin()
                f = it[idx]['feats']
            elif avg_dict['proxy'] == 'frames_gnn':
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
                if avg_dict['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif avg_dict['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif avg_dict['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            #elif avg == 1:
            #    f = it[-1]['feats']
            else:
                if avg_dict['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it][-avg:]), dim=0)
                elif avg_dict['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it][-avg:]), dim=0)[0]
                elif avg_dict['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it][-avg:]), dim=0)[0]
            feats.append(f)
            #print(f)
        if len(feats[0].shape) == 1:
            return torch.stack(feats)
        else: 
            return torch.cat(feats, dim=0)

    def get_gnn_feats(self, x, y, num_detects, avg_dict):
        if avg_dict['proxy'] == 'frames_gnn':
            if type(self.tracker_cfg['avg_inact']['num']) == int: 
                num_frames = self.tracker_cfg['avg_act']['num'] 
            else:
                num_frames = int(self.tracker_cfg['avg_act']['num'].split('_')[-1])
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
                edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
                _, feats = self.gnn(feats, edge_ind, edge_attr, 'plain')
            x = feats[-1][:num_detects, :]
            y = feats[-1][num_detects:, :]    
        
        return x, y
    
    def assign(self, tracks, dist, row, col, ids, curr_it=None, active_tracks=None, avg_dict=None): 
        # assign tracks from hungarian
        #num = int(avg_dict['num']) if avg_dict['proxy'] == 'frames_gnn' else 1
        if not active_tracks:
            active_tracks = list()
        assigned = list()
        debug = list()
        for r, c in zip(row, col):
            # assign tracks if reid distance < thresh
            
            # match w active track  
            if not curr_it:
                if ids[c] in self.tracks.keys() and dist[r, c] < self.act_reid_thresh:
                    if tracks[r]['id'] != self.tracks[ids[c]][-1]['id']:
                        debug.append(['act', tracks[r]['id'], self.tracks[ids[c]][-1]['id'], ids[c], dist[r, c], tracks[r]['max_iou'], self.tracks[ids[c]][-1]['max_iou'], tracks[r]['im_index'], self.tracks[ids[c]][-1]['im_index'], tracks[r]['bbox'][1], self.tracks[ids[c]][-1]['bbox'][1]])
                    self.tracks[ids[c]].append(tracks[r])
                    active_tracks.append(ids[c])
                    assigned.append(r)
        
            # match w inactive track
            else:
                if ids[c] in self.inactive_tracks.keys() and dist[r, c] < self.inact_reid_thresh:
                    if tracks[r]['id'] != self.inactive_tracks[ids[c]][-1]['id']:
                        debug.append(['inact', tracks[r]['id'], self.inactive_tracks[ids[c]][-1]['id'], ids[c], dist[r, c], tracks[r]['max_iou'], self.inactive_tracks[ids[c]][-1]['max_iou'], tracks[r]['im_index'], self.inactive_tracks[ids[c]][-1]['im_index'], tracks[r]['bbox'][1], self.inactive_tracks[ids[c]][-1]['bbox'][1]])
                    
                    self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                    del self.inactive_tracks[ids[c]]
                    for i in range(len(self.tracks[ids[c]])):
                        del self.tracks[ids[c]][i]['inact_count']

                    self.tracks[ids[c]].append(tracks[r])
                    active_tracks.append(ids[c])
                    assigned.append(r)
                    
        tracks = [tracks[i] for i in range(len(tracks)) if i not in assigned]
        #if len(debug) != 0:
        #    print(debug)

        return tracks, active_tracks

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
