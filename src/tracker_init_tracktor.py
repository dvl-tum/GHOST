from collections import defaultdict
import torch
import sklearn.metrics
import scipy
from tracking_wo_bnw.src.tracktor.utils import interpolate
import os
import numpy as np
import os.path as osp
import csv
import logging
from nets.proxy_gen import ProxyGenRNN, ProxyGenMLP

logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker():
    def __init__(self, tracker_cfg, encoder, gnn=None, graph_gen=None):
        self.encoder = encoder
        self.gnn = gnn
        self.graph_gen = graph_gen
        self.tracker_cfg = tracker_cfg
        self.num_el_id = 0
        self.proxy_gen = ProxyGenRNN
        self.hidden = dict()

        self.inact_thresh = tracker_cfg['inact_thresh']
        self.reid_thresh = tracker_cfg['reid_thresh']
        self.output_dir = tracker_cfg['output_dir']
        
        avg = self.tracker_cfg['avg_inact']['num'] if self.tracker_cfg['avg_inact']['do'] else 'last_frame'
        self.experiment = 'mode' + '_' + avg  + '_' + str(self.inact_thresh)
        if self.gnn:
            self.experiment = 'gnn_' + self.experiment
        if self.tracker_cfg['init_tracktor']['do']:
            self.experiment = 'init_tracktor_' + self.experiment

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)

    def track(self, seq):
        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        logger.info("Tracking Sequence {} of length {}".format(seq.name, 
                        seq.num_frames))
        
        i, self.id = 0, 0
        gt = list()
        for frame, g, _, boxes, tracktor_ids in seq:
            print(i)
            gt.append({'gt': g})
            tracks = list()
            with torch.no_grad():
                _, feats = self.encoder(frame, output_option='plain')

            for f, b, tr_id in zip(feats, boxes, tracktor_ids):
                tracks.append({'bbox': b, 'feats': f, 'im_index': i, 'tracktor_id': tr_id})
            
            if not self.tracker_cfg['init_tracktor']['do']:
                self.track_wo_tracktor(tracks, i)
            else:
                self.track_w_tracktor(tracks)
            
            i += 1
        
        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)
        
        # get results
        results = self.make_results()
        results = interpolate(results)
        self.write_results(results, self.output_dir, seq.name)

    def track_wo_tracktor(self, tracks, i):
        if i == 0:
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
        elif i > 0:
            dist, row, col, ids = self.get_hungarian(tracks)
            if dist is not None:
                self.assign(tracks, dist, row, col, ids)
    
    def track_w_tracktor(self, tracks):
        new_tracks_for_matching = list()
        
        # check if already enough elements for graph in current tracks, otherwise: add to tracks
        for tr in tracks:
            tr_id = tr['tracktor_id']
            if tr_id in self.tracks.keys() and len(self.tracks[tr_id]) >= self.num_el_id - 1:
                    new_tracks_for_matching.append(tr)
            else:
                self.tracks[tr_id].append(tr)

        if len(new_tracks_for_matching) == 0:
            return

        x = torch.stack([t['feats'] for t in new_tracks_for_matching])
        num_detects = x.shape[0]

        # check if alreagy enough elements for graph in prior tracks
        tracks_for_matching = {k: v for k, v in self.tracks.items() 
                                if len(v) >= self.num_el_id - 1}

        if len(tracks_for_matching) > 0:
            y = torch.stack([v[-i]['feats'] for v in tracks_for_matching.values() for i in range(1, self.num_el_id)])
            ids = [k for k in tracks_for_matching for j in range(self.num_el_id-1)]

        # check if already enough elements for graph in inactive tracks
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh 
                                and len(v) >= self.num_el_id - 1}
        if len(curr_it) > 0:
            y_inactive = torch.stack([v[-i]['feats'] for v in curr_it.values() for i in range(1, self.num_el_id)])
            if len(tracks_for_matching) > 0:
                y = torch.cat([y, y_inactive])
                ids += [k for k, v in curr_it.items() for j in range(self.num_el_id-1)]
            else:
                y = y_inactive
                ids = [k for k, v in curr_it.items() for j in range(self.num_el_id-1)]

        # compute gnn features for new detectons
        x, y = self.get_gnn_feats(x, y, num_detects)

        dict_for_avg = defaultdict(list)
        for i, f in zip(ids, y):
            dict_for_avg[i].append({'feats': f})
        
        if self.tracker_cfg['avg_inact']['do']:
            y = self.avg_it(dict_for_avg)
        else:
            y = torch.stack([v[0]['feats'] for v in dict_for_avg.values()])
        
        ids = list(dict_for_avg.keys())
        
        dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy())
        # row represent current frame, col represents last frame + inactiva tracks
        row, col = scipy.optimize.linear_sum_assignment(dist)

        self.assign(new_tracks_for_matching, dist, row, col, ids)

    def get_hungarian(self, tracks):
        x = torch.stack([t['feats'] for t in tracks])
        num_detects = x.shape[0]

        if len(self.tracks) > 0:
            #y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
            y = self.avg_it(self.tracks)
            ids = list(self.tracks.keys())
        
        # get tracks to compare to
        curr_it = {k: v for k, v in self.inactive_tracks.items() 
                                if v[-1]['inact_count'] <= self.inact_thresh}
            
        print(len(curr_it), len(self.tracks))

        if len(curr_it) > 0:
            if self.tracker_cfg['avg_inact']['do']:
                y_inactive = self.avg_it(curr_it)
            else:
                y_inactive = torch.stack([v[-1]['feats'] for v in curr_it.values()])
            
            if len(self.tracks) > 0:
                y = torch.cat([y, y_inactive])
                ids += [k for k, v in curr_it.items()]
            else:
                y = y_inactive
                ids = [k for k, v in curr_it.items()]
        elif len(curr_it) == 0 and len(self.tracks) == 0:
            logger.info("No active and no inactive tracks")
            for tr in tracks:
                self.tracks[self.id].append(tr)
                self.id += 1
            return None, None, None, None

        # compute gnn features
        if self.gnn:
            x, y = self.get_gnn_feats(x, y, num_detects)
        
        dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy())
        # row represent current frame, col represents last frame + inactiva tracks
        row, col = scipy.optimize.linear_sum_assignment(dist)
        
        return dist, row, col, ids

    def avg_it(self, curr_it):
        feats = list()
        avg = self.tracker_cfg['avg_inact']['num'] 
        if avg != 'all':
            avg = int(avg)
        for i, it in curr_it.items():
            print(i, len(it))
            if avg == 'all' or len(it) < avg:
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it]), dim=0)[0]
            elif avg == 'rnn':
                f, h = self.proxy_gen(it[-1]['feats'], self.hidden[i])
                self.hidden[i] = h
            else:
                if self.tracker_cfg['avg_inact']['proxy'] == 'mean':
                    f = torch.mean(torch.stack([t['feats'] for t in it][-avg:]), dim=0)
                elif self.tracker_cfg['avg_inact']['proxy'] == 'median':
                    f = torch.median(torch.stack([t['feats'] for t in it][-avg:]), dim=0)[0]
                elif self.tracker_cfg['avg_inact']['proxy'] == 'mode':
                    f = torch.mode(torch.stack([t['feats'] for t in it][-avg:]), dim=0)[0]
            feats.append(f)

        return torch.stack(feats)


    def get_gnn_feats(self, x, y, num_detects):
        feats = torch.cat([x, y])
        with torch.no_grad():
            edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
            _, feats = self.gnn(feats, edge_ind, edge_attr, 'plain')
        x = feats[-1][:num_detects, :]
        y = feats[-1][num_detects:, :]
        
        return x, y
    
    def assign(self, tracks, dist, row, col, ids): 
        #print("inside assign")
        #print(tracks)
        #print(len(tracks))
        # assign tracks from hungarian
        active_tracks = list()
        for r, c in zip(row, col):
            # assign tracks if reid distance < thresh
            if dist[r, c] < self.reid_thresh:

                # match w active track  
                if ids[c] in self.tracks.keys():
                    self.tracks[ids[c]].append(tracks[r])
                    active_tracks.append(ids[c])

                # match w inactive track
                elif ids[c] in self.inactive_tracks.keys():
                    self.tracks[ids[c]] = self.inactive_tracks[ids[c]]
                    del self.inactive_tracks[ids[c]]
                    for i in range(len(self.tracks[ids[c]])):
                        del self.tracks[ids[c]][i]['inact_count']

                    self.tracks[ids[c]].append(tracks[r])
                    active_tracks.append(ids[c])

            # new track bc distance too big
            else:
                #print(tracks[r])
                self.tracks[self.id].append(tracks[r])
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
                self.id += 1

    def make_results(self):
        results = defaultdict(dict)
        for i, ts in self.tracks.items():
            for t in ts:
                results[i][t['im_index']] = t['bbox']
        return results
    
    def write_results(self, all_tracks, output_dir, seq_name):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

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
                        [frame + 1,
                         i + 1,
                         x1 + 1,
                         y1 + 1,
                         x2 - x1 + 1,
                         y2 - y1 + 1,
                         -1, -1, -1, -1])

