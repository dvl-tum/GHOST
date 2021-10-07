from collections import defaultdict
from types import new_class
from main_analysis import analysis
import torch
import os
import os.path as osp
from .MOTDataset import MOTDataset
from .MOT17_parser import MOTLoader
import pandas as pd
import PIL.Image as Image
from .utils import ClassBalancedSampler
import numpy as np
from ReID.dataset.utils import make_transform_bot
import copy


def collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    visibility = [item[2] for item in batch]
    im_path = [item[3] for item in batch]
    dets = [item[4] for item in batch]

    return [data, target, visibility, im_path, dets]


class ReIDDataset(MOTDataset):
    def __init__(self, split, sequences, dataset_cfg, tracker_cfg, dir, data_type='query', datastorage='data'):
        self.vis_thresh = tracker_cfg['iou_thresh']
        self.jaccard_thresh = tracker_cfg['jaccard_thresh']
        self.occluder_thresh = tracker_cfg['occluder_thresh']
        self.size_thresh =  tracker_cfg['size_thresh']
        self.frame_dist_thresh =  tracker_cfg['frame_dist_thresh']
        self.size_diff_thresh = tracker_cfg['size_diff_thresh']
        self.gallery_vis_thresh = tracker_cfg['gallery_vis_thresh']
        self.rel_gallery_vis_thresh = tracker_cfg['rel_gallery_vis_thresh']
        self.only_next_frame = tracker_cfg['only_next_frame']
        self.mode = split.split('_')[-1]
        self.data_type = data_type
        self.gallery_mask = None
        self.data_unclipped = list()
        self.gt = list()

        # if dist was already computed for this seq
        self.dist_computed = False
        super(ReIDDataset, self).__init__(split, sequences, dataset_cfg, dir, datastorage, add_detector=False)

    def process(self):
        self.transform = make_transform_bot(is_train=False, sz_crop=[256, 128])
        print(self.transform)
        self.id_to_y = dict()

        for seq in self.sequences:
            #print(seq)
            seq_ids = dict()
            if not self.preprocessed_exists or self.dataset_cfg['prepro_again']:
                loader = MOTLoader([seq], self.dataset_cfg, self.dir)
                loader.get_seqs()
                
                dets = loader.dets
                os.makedirs(self.preprocessed_dir, exist_ok=True)
                dets.to_pickle(self.preprocessed_paths[seq])

                dets_unclipped = loader.dets_unclipped
                dets_unclipped.to_pickle(self.preprocessed_paths[seq][:-4] + '_unclipped.pkl')

                gt = loader.gt
                os.makedirs(self.preprocessed_dir, exist_ok=True)
                gt.to_pickle(self.preprocessed_gt_paths[seq])
            else:
                dets = pd.read_pickle(self.preprocessed_paths[seq])
                dets_unclipped = pd.read_pickle(self.preprocessed_paths[seq][:-4] + '_unclipped.pkl')

            dets['gt_id'] = copy.deepcopy(dets['id'].values)

            for i, row in dets.iterrows():
                if row['id'] not in seq_ids.keys():
                    seq_ids[row['id']] = self.id
                    self.id += 1
                dets.at[i, 'id'] = seq_ids[row['id']]
                dets_unclipped.at[i, 'id'] = seq_ids[row['id']]

            self.id_to_y[seq] = seq_ids

            #if 'vis' in dets and dets['vis'].unique() != [-1]:
            #    dets = dets[dets['vis'] > self.dataset_cfg['gt_training_min_vis']]

            self.data.append(dets)
            self.data_unclipped.append(dets_unclipped)
            self.gt.append(gt)

        self.id += 1
        self.seperate_seqs()

        #self.occurance_analysis()
        
        if self.mode == 'test':
            self.make_query_gal()

    def occurance_analysis(self):

        # analyse vis
        analysis = defaultdict(dict)
        
        vis = [[np.round(v-0.1, decimals=1), np.round(v, decimals=1)] for v in np.linspace(0.1, 1.0, num=10)]
        vis[-1][-1] = 1.1
        vis = [tuple(v) for v in vis]
        
        for v in vis: 
            samps = self.seq_all
            samps = samps[samps['vis'] >= v[0]]   #vis_thresh = [0.5, 0.6]
            samps = samps[samps['vis'] < v[1]]
            analysis['vis'][str(v)] = samps.shape[0]
        
        size = [(v, v+25) for v in np.linspace(25, 775, num=31)]
        for s in size: 
            samps = self.seq_all
            samps = samps[samps['bb_height'] >= s[0]]   #vis_thresh = [0.5, 0.6]
            samps = samps[samps['bb_height'] < s[1]]
            analysis['size'][str(s)] = samps.shape[0]

        
        frame_dist = [(v, v+1) for v in np.linspace(0, 40, num=41)]
        frame_rate = int(self.data[0].attrs['frameRate'])
        for f in frame_dist:
            samps = self.seq_all
            for i, s in samps.iterrows():
                same_id = samps[samps['id'] == s['id']]
                same_id = same_id[same_id['frame'] != s['frame']]
                frames = same_id['frame'].to_numpy()
                if frames[frames>s['frame']].shape[0] > 0:
                    next_frame = np.min(frames[frames>s['frame']])
                else: continue
                frame_dist = next_frame - s['frame']

                if next_frame >= s['frame'] + (f[0] * frame_rate) and \
                    next_frame < s['frame'] + (f[1] * frame_rate):
                    if str(f) in analysis['frame_dist'].keys():
                        analysis['frame_dist'][str(f)] += 1
                    else:
                        analysis['frame_dist'][str(f)] = 1
        
        rel_size = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 5.0)] #[(5.0, 2.0)] + [(np.round(v, decimals=1), np.round(v-0.1, decimals=1)) for v in np.linspace(2, -0.9, num=30)]
        for r in rel_size:
            samps = self.seq_all
            for i, s in samps.iterrows():
                same_id = samps[samps['id'] == s['id']]
                same_id = same_id[same_id['frame'] != s['frame']]
                frames = same_id['frame'].to_numpy()

                if frames[frames>s['frame']].shape[0] > 0:
                    next_frame = np.min(frames[frames>s['frame']])
                else: continue

                size = same_id[same_id['frame'] == next_frame]['bb_height'].to_numpy()[0]
                if size > ((1+r[0]) * s['bb_height']) and \
                    size <= ((1+r[1]) * s['bb_height']):
                    if str(r) in analysis['rel_size'].keys():
                        analysis['rel_size'][str(r)] += 1
                    else:
                        analysis['rel_size'][str(r)] = 1
        
        gall_vis = [[np.round(v-0.1, decimals=1), np.round(v, decimals=1)] for v in np.linspace(0.1, 1.0, num=10)]
        gall_vis[-1][-1] = 1.1
        gall_vis = [tuple(v) for v in gall_vis]

        for r in gall_vis:
            samps = self.seq_all
            for i, s in samps.iterrows():
                same_id = samps[samps['id'] == s['id']]
                same_id = same_id[same_id['frame'] != s['frame']]
                frames = same_id['frame'].to_numpy()
 
                if frames[frames>s['frame']].shape[0] > 0:
                    next_frame = np.min(frames[frames>s['frame']])
                else: continue

                gall_vis = same_id[same_id['frame'] == next_frame]['vis'].to_numpy()[0]
                if gall_vis >= r[0] and gall_vis < r[1]:
                    #q_vis 
                    for v in vis:
                        if s['vis'] >= v[0] and s['vis'] < v[1]:
                            q_vis = str(v)
                    name = "Q_" + str(q_vis) + "_G_" + str(r)
                    if name in analysis['gall_vis'].keys():
                        analysis['gall_vis'][name] += 1
                    else:
                        analysis['gall_vis'][name] = 1
        
        print()
        print("Sequence {}".format(self.sequences[0]))
        print('Ocurrances')
        print(analysis)
        print([sum(list(v.values())) for k, v in analysis.items()])
        print()

    def seperate_seqs(self):
        samples = list()
        ys = list()
        for i, seq in enumerate(self.data):
            if i == 0:
                seq_all = seq
                seq_all_unclipped = self.data_unclipped[i]
                gt_all = self.gt[i]
            else:
                seq_all = seq_all.append(seq)
                seq_all_unclipped = seq_all_unclipped.append(self.data_unclipped[i])
                gt_all = gt_all.append(self.gt[i])

        seq_all.reset_index(drop=True, inplace=True)
        seq_all_unclipped.reset_index(drop=True, inplace=True)

        indices = list(range(seq_all['frame'].count()))
        seq_all.reindex(indices)
        seq_all_unclipped.reindex(indices)

        self.seq_all = seq_all
        self.seq_all_unclipped = seq_all_unclipped
        self.gt_all = gt_all

    def get_ioa(self, frame_df, frame_df_gt=None, correspondance=None, seq=None, visualization=False, prev=False):
        np.set_printoptions(linewidth=np.inf)
        if frame_df_gt is None:
            frame_df_gt = frame_df
        
        if correspondance is None:
            correspondance = np.argwhere(np.eye(frame_df.shape[0], frame_df.shape[0]))
        
        if prev:
            cols = ['bb_left prev', 'bb_top prev', 'bb_width prev', 'bb_height prev']
        else:
            cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height']

        def get_intersections(bbs1l, bbs2l, pri=False):
            bbs1 = np.asarray([r for r in bbs1l for i in range(len(bbs2l))])
            bbs2 = np.asarray([r for i in range(len(bbs1l)) for r in bbs2l])
            
            # a = top left for intersecting
            ax = np.max(np.vstack([bbs1[:, 0], bbs2[:, 0]]), axis=0)
            ay = np.max(np.vstack([bbs1[:, 1], bbs2[:, 1]]), axis=0)
            # b = top right for intersecting
            bx = np.min(np.vstack([bbs1[:, 0] + bbs1[:, 2], bbs2[:, 0] + bbs2[:, 2]]), axis=0)
            by = np.max(np.vstack([bbs1[:, 1] , bbs2[:, 1]]), axis=0)
            # c = bottom left for intersecting
            cx = np.max(np.vstack([bbs1[:, 0], bbs2[:, 0]]), axis=0)
            cy = np.min(np.vstack([bbs1[:, 1] + bbs1[:, 3], bbs2[:, 1] + bbs2[:, 3]]), axis=0)
            # d = bottom right for intersecting
            dx = np.min(np.vstack([bbs1[:, 0] + bbs1[:, 2], bbs2[:, 0] + bbs2[:, 2]]), axis=0)
            dy = np.min(np.vstack([bbs1[:, 1] + bbs1[:, 3], bbs2[:, 1] + bbs2[:, 3]]), axis=0)

            # upper line for intersecting
            ab = bx - ax
            # left line for intersecting
            ac = cy - ay

            # intersection area for intersecting
            intersection = ab * ac

            return intersection, ab, ac, ax, ay, bx, by, cx, cy, dx, dy

        def update(to_update, argument_a, argument_b):
            # set values to zero if bbs are not intersecting, i.e., neg width or height
            import copy
            to_update = copy.deepcopy(to_update)
            # if argument_a (ab) or argument_b (ac) negative ---> bbs are not intersecting
            to_update = np.where(argument_a>0, to_update, 0)
            to_update = np.where(argument_b>0, to_update, 0)
            return np.round(to_update, decimals=2)
        
        bbs1l = frame_df[cols].values.tolist()
        bbs2l = frame_df_gt[cols].values.tolist()

        intersection, ab, ac, ax, ay, bx, by, cx, cy, dx, dy = get_intersections(bbs1l, bbs2l)

        # detections = rows --> gt_bbs = cols
        # bb_left, bb_top, bb_width, bb_height
        intersection_bbs = np.asarray([np.reshape(update(ax, ab, ac), (frame_df.shape[0], frame_df_gt.shape[0])), 
                            np.reshape(update(ay, ab, ac), (frame_df.shape[0], frame_df_gt.shape[0])), 
                            np.reshape(update(ab, ab, ac), (frame_df.shape[0], frame_df_gt.shape[0])), 
                            np.reshape(update(ac, ab, ac), (frame_df.shape[0], frame_df_gt.shape[0]))])

        # re-arange intersection values as matrix
        intersection = np.reshape(update(intersection, ab, ac), (frame_df.shape[0], frame_df_gt.shape[0]))

        # set to zero for intersection with own GT bb
        intersection[correspondance[:, 0], correspondance[:, 1]] = 0

        # set intersection to zero if bb_bottom larger than col and use gt for this
        bot = np.asarray(bbs2l)[:, 1] + np.asarray(bbs2l)[:, 3]
        # if bb is more up in img then bb occluded --> keep ioa if bb more up in image --> when upper == True
        upper = np.expand_dims(bot,axis=0).T < np.expand_dims(bot, axis=0) 

        # get rows 
        corresponding_upper = upper[correspondance[:, 1], :].astype(float)
        intersection = intersection * corresponding_upper

        ##### NEW #####
        for i in range(intersection_bbs.shape[1]):
            #those are the intersection bbs of each GT bb with the i-th detection bb --> dim 1 = num GT bbs 
            det_inters = intersection_bbs[:, i, :].T 

            if np.nonzero(det_inters[:, 2])[0].shape < np.nonzero(det_inters[:, 3])[0].shape:
                non_zeros = np.nonzero(det_inters[:, 2])[0].tolist()
            else:
                non_zeros = np.nonzero(det_inters[:, 3])[0].tolist()

            non_zeros.remove(correspondance[i, 1])
            non_zeros = np.asarray(non_zeros)

            if non_zeros.shape[0] == 0:
                continue
            all_inds = np.argsort(-1*bot[non_zeros])

            prev_inter = None
            intra_intersection_dict = dict()
            for j, ind in enumerate(np.argsort(-1*bot[non_zeros])[:-1]):

                comp = non_zeros[ind]
                others = non_zeros[all_inds[j+1:]]
                other_bbs = det_inters[others].tolist()

                _intersection, ab, ac, ax, ay, bx, by, cx, cy, dx, dy = get_intersections([det_inters[comp].tolist()], other_bbs)
                if not np.any(_intersection>0):
                    continue

                if j > 0 and prev_inter:
                    # get intersection between
                    # 1. intersection bbs of current comp with others
                    # 2. intersection bb of previous comp with current comp 
                    intra_intersection = get_intersections(np.asarray([ax, ay, ab, ac]).T.tolist(), [np.asarray([prev_inter[3][0], prev_inter[4][0], prev_inter[1][0], prev_inter[2][0]]).T.tolist()])

                    # if positive: subtract
                    for p, intra in enumerate(intra_intersection[0]):
                        if intra > 0:
                            _intersection[p] -= intra

                # others are more up than comp --> rm inter from intersection
                changed = False
                for inter, o in zip(_intersection, others):
                    if inter > 0 and upper[correspondance[i, 1], o]:
                        changed = True
                        intersection[i, o] = np.max([intersection[i, o] - inter, 0])

                if not changed:
                    continue

                prev_inter = [_intersection, ab, ac, ax, ay, bx, by, cx, cy, dx, dy]

        # area of detected bbs
        diag = np.asarray(bbs1l)[:, 2] * np.asarray(bbs1l)[:, 3]
        diag = np.tile(diag, (frame_df_gt.shape[0], 1)).T
        ioa = intersection / diag 

        # set ioa to zero where ID det == ID GT
        ioa = np.where(ioa > 0, ioa, 0)

        # get visibility, ioa sum and boader cases
        vis = frame_df['vis'].values
        person_occ = np.sum(ioa * np.tile(frame_df_gt['label'].isin([1, 2, 7, 8, 12]).values, (frame_df.shape[0], 1)), axis=1)
        object_occ = np.sum(ioa * ~np.tile(frame_df_gt['label'].isin([1, 2, 7, 8, 12]).values, (frame_df.shape[0], 1)), axis=1)

        # get entering/leaving the scene
        # top and bottom
        border_h1 = bot[correspondance[:, 1]]>=int(self.data[0].attrs['imHeight'])-1
        top = np.asarray(bbs2l)[:, 1]
        border_h2 = top[correspondance[:, 1]]<=0
        # right and left 
        left = np.asarray(bbs2l)[:, 0] + np.asarray(bbs2l)[:, 2]
        border_w1 = left[correspondance[:, 1]]>=int(self.data[0].attrs['imWidth'])-1
        right = np.asarray(bbs2l)[:, 0]
        border_w2 = right[correspondance[:, 1]]<=0
        # combine both
        border_case = border_w1 | border_w2 | border_h1 | border_h2

        # 2 == occluded by person, 1 == occluded by object, 0 == not occluded
        occlusion = np.zeros(ioa.shape[0])
        occlusion = np.where((vis<0.55) & (object_occ>=0.45), 1, occlusion) 
        occlusion = np.where((person_occ>=0.45) & (vis<0.55), 2, occlusion)
        occlusion = np.where(border_case & (vis<=0.98), 3, occlusion)
        occlusion = np.where(~border_case & (person_occ<0.45) & (object_occ<0.45) & (vis<0.55), 4, occlusion)
                
        return occlusion, ioa

    def make_query_gal(self):
        self.different_gallery_set = False
        first_gallery_mask = False

        self.queries = self.seq_all
        self.query_indices = list()
        self.galleries = self.seq_all
        
        if self.jaccard_thresh != 0:
            ioa_dict = dict()
            id_dict = dict()
            gt_ids = dict()
            for frame in self.queries['frame'].unique():
                frame_df = self.queries[self.queries['frame']==int(frame)]
                frame_df = frame_df[frame_df['vis'] != -1]
                frame_gt_all = self.gt_all[self.gt_all['frame']==int(frame)]

                if frame_df.shape[0] == 0:
                    continue

                correspondance = np.argwhere((np.expand_dims(frame_df['gt_id'].values.astype(float), axis=0).T == np.expand_dims(frame_gt_all['id'].values, axis=0)).astype(float))
                _, ioa = self.get_ioa(frame_df, frame_gt_all, correspondance=correspondance)

                ioa_dict[frame] = ioa
                id_dict[frame] = frame_df['gt_id'].values.tolist()
                gt_ids[frame] = frame_gt_all['id'].values.tolist()

            # NEED TO CHANGE THIS
            self.gallery_mask = list()
            for ind, q in self.queries.iterrows():
                pid = q['gt_id']
                if pid == -1:
                    self.gallery_mask.append(np.array([False]*self.galleries.shape[0]))
                    continue
                q_map = self.galleries['gt_id'] != pid

                pid_df = self.galleries[self.galleries['gt_id'] == pid]
                for ind2, g in pid_df.iterrows():
                    if ind2 == ind:
                        continue
                    # prev frame
                    r_1 = q
                    # curr frame
                    r_2 = g

                    ioa_1 = ioa_dict[r_1['frame']]
                    ioa_2 = ioa_dict[r_2['frame']]
                    
                    id_1i = id_dict[r_1['frame']]
                    id_2i = id_dict[r_2['frame']]

                    id_1i = id_1i.index(pid)
                    id_2i = id_2i.index(pid)

                    ioa_1 = ioa_1[id_1i]
                    ioa_2 = ioa_2[id_2i]

                    id_1 = gt_ids[r_1['frame']]
                    id_2 = gt_ids[r_2['frame']]

                    correspondance = np.array([[m, n] for m in range(len(id_1)) for n in range(len(id_2)) if id_1[m] == id_2[n]])
                    #mins = np.sum(np.minimum(ioa_1[correspondance[:, 0]], ioa_2[correspondance[:, 1]]))
                    #maxs = np.sum(np.maximum(ioa_1[correspondance[:, 0]], ioa_2[correspondance[:, 1]]))

                    val_1 = np.sum(np.abs(ioa_1[correspondance[:, 0]] - ioa_2[correspondance[:, 1]]))

                    # use standard weighted jaccard and pretend zeros for non-matched entries
                    not_in_2 = [m for m in range(len(id_1)) if id_1[m] not in id_2]
                    not_in_1 = [m for m in range(len(id_2)) if id_2[m] not in id_1]
                    for ni1 in not_in_1:
                        val_1 += ioa_2[ni1]
                        #maxs += ioa_2[ni1]
                    for ni2 in not_in_2:
                        val_1 += ioa_1[ni2]
                        #maxs += ioa_1[ni2]
                    
                    '''if maxs == 0:
                        weighted_jacc = np.array(1.0)
                    else:
                        weighted_jacc = np.divide(mins, maxs)
                        weighted_jacc = np.nan_to_num(weighted_jacc, nan=1)'''

                    weighted_jacc = val_1/2

                    if weighted_jacc > self.jaccard_thresh[0] and weighted_jacc < self.jaccard_thresh[1]:
                        q_map[ind2] = True

                #pd.set_option('display.max_rows', q_map.shape[0]+1)
                self.gallery_mask.append(q_map)
            
            self.gallery_mask = np.asarray(self.gallery_mask)
            first_gallery_mask = True

        if self.occluder_thresh != 0:
            self.different_gallery_set = True
            self.queries['occluder'] = np.ones(self.queries.shape[0]) * -1
            for frame in self.queries['frame'].unique():
                frame_df = self.queries[self.queries['frame']==int(frame)]
                frame_df = frame_df[frame_df['vis'] != -1]
                frame_gt_all = self.gt_all[self.gt_all['frame']==int(frame)]

                if frame_df.shape[0] == 0:
                    continue
                    
                #occluder
                correspondance = np.argwhere((np.expand_dims(frame_df['gt_id'].values, axis=0).T == np.expand_dims(frame_gt_all['id'].values, axis=0)).astype(float))
 
                #print(frame, seq, frame_df['gt_id'].values, frame_gt['id'].values)
                occlusion, _ = self.get_ioa(frame_df, frame_gt_all, correspondance=correspondance)
                self.queries.loc[frame_df.index, 'occluder'] = occlusion
            self.queries = self.queries[self.queries['occluder'] >= self.occluder_thresh[0]]   #vis_thresh = [0.5, 0.6]
            self.queries = self.queries[self.queries['occluder'] < self.occluder_thresh[1]] 

        if self.vis_thresh != 0:
            self.different_gallery_set = True
            if type(self.vis_thresh) != tuple:
                self.queries = self.queries[self.queries['vis'] == self.vis_thresh]
            else:
                self.queries = self.queries[self.queries['vis'] >= self.vis_thresh[0]]   #vis_thresh = [0.5, 0.6]
                self.queries = self.queries[self.queries['vis'] < self.vis_thresh[1]] 

        if self.size_thresh != 0:
            self.different_gallery_set = True
            if type(self.size_thresh) != tuple:
                self.queries = self.queries[self.queries['bb_height'] >= self.size_thresh]
            else:
                self.queries = self.queries[self.queries['bb_height'] >= self.size_thresh[0]]   #vis_thresh = [50, 75]
                self.queries = self.queries[self.queries['bb_height'] < self.size_thresh[1]] 
        
        if self.frame_dist_thresh != 0:
            frame_rate = int(self.data[0].attrs['frameRate'])
            if not first_gallery_mask:
                self.gallery_mask = list()
            i = 0
            for ind, q in self.queries.iterrows():
                q_map11 = self.galleries['frame'] >= q['frame'] + (self.frame_dist_thresh[0] * frame_rate)
                q_map12 = self.galleries['frame'] <= q['frame'] - (self.frame_dist_thresh[0] * frame_rate)
                q_map21 = self.galleries['frame'] < q['frame'] + (self.frame_dist_thresh[1] * frame_rate)
                q_map22 = self.galleries['frame'] > q['frame'] - (self.frame_dist_thresh[1] * frame_rate)
                q_map11, q_map12, q_map21, q_map22 = np.asarray(q_map11), np.asarray(q_map12), np.asarray(q_map21), np.asarray(q_map22)
                q_map = (q_map11 & q_map21) | (q_map12 & q_map22)

                q_map[ind] = False
                if first_gallery_mask:
                    self.gallery_mask[i] = self.gallery_mask[i] & q_map
                else:
                    self.gallery_mask.append(q_map)
                i += 1
            self.gallery_mask = np.asarray(self.gallery_mask)
            first_gallery_mask = True

        if self.size_diff_thresh != 0:
            if not first_gallery_mask:
                self.gallery_mask = list()
            i = 0
            for ind, q in self.queries.iterrows():
                q_map1 = self.galleries['bb_height'] > abs((1+self.size_diff_thresh[0]) * q['bb_height'])
                q_map2 = self.galleries['bb_height'] <= abs((1+self.size_diff_thresh[1]) * q['bb_height'])
                q_map1, q_map2 = np.asarray(q_map1), np.asarray(q_map2)
                q_map = q_map1 & q_map2

                q_map[ind] = False
                if first_gallery_mask:
                    self.gallery_mask[i] = self.gallery_mask[i] & q_map
                else:
                    self.gallery_mask.append(q_map)
                i += 1
            self.gallery_mask = np.asarray(self.gallery_mask)
            first_gallery_mask = True

        if self.gallery_vis_thresh != 0:
            if not first_gallery_mask:
                self.gallery_mask = list()
            i = 0
            for ind, q in self.queries.iterrows():
                q_map1 = self.galleries['vis'] >= self.gallery_vis_thresh[0]
                q_map2 = self.galleries['vis'] < self.gallery_vis_thresh[1]
                q_map1, q_map2 = np.asarray(q_map1), np.asarray(q_map2)
                q_map = q_map1 & q_map2
                q_map[ind] = False
                if first_gallery_mask:
                    self.gallery_mask[i] = self.gallery_mask[i] & q_map
                else:
                    self.gallery_mask.append(q_map)
                i += 1
            self.gallery_mask = np.asarray(self.gallery_mask)
            first_gallery_mask = True

        if self.rel_gallery_vis_thresh != 0:
            if not first_gallery_mask:
                self.gallery_mask = list()
            i = 0
            for ind, q in self.queries.iterrows():
                q_map1 = self.galleries['vis'] >= (1+self.rel_gallery_vis_thresh[0]) * q['vis']
                q_map2 = self.galleries['vis'] < (1+self.rel_gallery_vis_thresh[1]) * q['vis']
                q_map1, q_map2 = np.asarray(q_map1), np.asarray(q_map2)
                q_map = q_map1 & q_map2
                q_map[ind] = False
                if first_gallery_mask:
                    self.gallery_mask[i] = self.gallery_mask[i] & q_map
                else:
                    self.gallery_mask.append(q_map)
                i += 1
            self.gallery_mask = np.asarray(self.gallery_mask)
            first_gallery_mask = True
        
        if self.only_next_frame != 0:
            if not first_gallery_mask:
                self.gallery_mask = list()
            i = 0
            for ind, q in self.queries.iterrows():
                frame = q['frame']
                same_id = self.galleries[self.galleries['id'] == q['id']]
                same_id = same_id[same_id['frame'] != q['frame']]
                frames = same_id['frame'].to_numpy()
                if frames[frames>q['frame']].shape[0] > 0:
                    next_frame = np.min(frames[frames>q['frame']])
                else:
                    next_frame = -1000
                q_map = self.galleries['frame'] == next_frame
                
                if first_gallery_mask:
                    self.gallery_mask[i] = self.gallery_mask[i] & q_map
                else:
                    self.gallery_mask.append(q_map)
                i += 1 
            self.gallery_mask = np.asarray(self.gallery_mask)
            first_gallery_mask = True

        self.query_indices = self.queries.index.to_numpy()
        
        if self.data_type == 'query':
            self.data = self.queries
            self.data_unclipped = self.seq_all_unclipped.loc[self.data.index.values.tolist()]
        elif self.data_type == 'gallery':
            self.data = self.galleries 
            self.data_unclipped = self.seq_all_unclipped.loc[self.data.index.values.tolist()]

    def get_bounding_boxe(self, row, row_unclipped, zero_pad=False):
        # tracktor resize (256,128))

        img = self.to_tensor(Image.open(row['frame_path']).convert("RGB"))
        im = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]

        #pad image with zeros
        if zero_pad:
            left_pad = abs(int(row_unclipped['bb_left'])) if int(row_unclipped['bb_left']) < 0 else 0
            right_pad = abs(int(row_unclipped['bb_right']) - img.shape[2]) if int(row_unclipped['bb_right']) > img.shape[2] else 0
            top_pad = abs(int(row_unclipped['bb_top'])) if int(row_unclipped['bb_top']) < 0 else 0
            bot_pad = abs(int(row_unclipped['bb_bot']) - img.shape[1]) if int(row_unclipped['bb_bot']) > img.shape[1] else 0
            m = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bot_pad))
            im = m(im)

        im = self.to_pil(im)
        im = self.transform(im)

        return im
 
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        row_unclipped = self.data_unclipped.loc[row.name]

        #get index for precomputed matrix
        if self.data_type == 'query':
            index = self.query_indices[idx]
        else:
            index = idx

        # if dist was already computed, image not needed
        if self.dist_computed:
            return self.data.iloc[idx]['id'], idx
        else:
            y = row['id']
            img = self.get_bounding_boxe(row, row_unclipped)
            return img, index, y