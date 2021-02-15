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

logger = logging.getLogger('AllReIDTracker.Tracker')


class Tracker():
    def __init__(self, tracker_cfg, encoder, gnn=None, graph_gen=None):
        self.encoder = encoder
        self.gnn = gnn
        self.graph_gen = graph_gen
        self.tracker_cfg = tracker_cfg

        self.inact_thresh = tracker_cfg['inact_thresh']
        self.reid_thresh = tracker_cfg['reid_thresh']
        self.output_dir = tracker_cfg['output_dir']
        
        avg = self.tracker_cfg['avg_inact']['num'] if self.tracker_cfg['avg_inact']['do'] else 'last_frame'
        self.experiment = 'mode' + '_' + avg  + '_' + str(self.inact_thresh)

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)

    def track(self, seq):
        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)
        logger.info("Tracking Sequence {} of length {}".format(seq.name, 
                        seq.num_frames))
       
        i, self.id = 0, 0
        gt = list()
        for frame, g, _, boxes in seq:
            print(i)
            gt.append({'gt': g})
            tracks = list()
            with torch.no_grad():
                _, feats = self.encoder(frame, output_option='plain')
            
            for f, b in zip(feats, boxes):
                tracks.append({'bbox': b, 'feats': f, 'im_index': i})
            
            if i == 0:
                for d in tracks:
                    self.tracks[self.id].append(d)
                    self.id += 1
            elif i > 0:
                dist, row, col, ids = self.get_hungarian(tracks)
                self.assign(tracks, dist, row, col, ids)
            
            i += 1
        
        # add inactive tracks to active tracks for evaluation
        self.tracks.update(self.inactive_tracks)
        
        # get results
        results = self.make_results()
        results = interpolate(results)
        self.write_results(results, self.output_dir, seq.name)

    def get_hungarian(self, tracks):
        x = torch.stack([t['feats'] for t in tracks])
        num_detects = x.shape[0]
        
        y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
        ids = list(self.tracks.keys())
        
        inactive_tracks = {k: v for k, v in self.inactive_tracks.items() if v[-1]['inact_count'] <= self.inact_thresh}
        if len(inactive_tracks) > 0:
            y_inactive = torch.stack([v[-1]['feats'] for v in inactive_tracks.values()])
            y = torch.cat([y, y_inactive])
            ids += [k for k, v in inactive_tracks.items()]
        if self.gnn:
            x, y = self.get_gnn_feats(x, y, num_detects)
        
        dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy())
        # row represent current frame, col represents last frame + inactiva tracks
        row, col = scipy.optimize.linear_sum_assignment(dist)
        
        return dist, row, col, ids

    def get_gnn_feats(self, x, y, num_detects):
        feats = torch.cat([x, y])
        
        with torch.no_grad():
            edge_attr, edge_ind, feats = self.graph_gen.get_graph(feats)
            _, feats = self.gnn(feats, edge_ind, edge_attr, 'plain')
        x = feats[-1][:num_detects, :]
        y = feats[-1][num_detects:, :]
        
        return x, y
    
    def assign(self, tracks, dist, row, col, ids): 
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

