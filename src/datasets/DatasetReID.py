import torch
import os
import os.path as osp
from .MOTDataset import MOTDataset
from .MOT17_parser import MOTLoader
import pandas as pd
import PIL.Image as Image
from .utils import ClassBalancedSampler
import numpy as np


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
        self.size_thresh =  tracker_cfg['size_thresh']
        self.frame_dist_thresh =  tracker_cfg['frame_dist_thresh']
        self.size_diff_thresh = tracker_cfg['size_diff_thresh']
        self.mode = split.split('_')[-1]
        self.data_type = data_type
        self.gallery_mask = None
        super(ReIDDataset, self).__init__(split, sequences, dataset_cfg, dir, datastorage, )

    def process(self):
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

                gt = loader.gt
                os.makedirs(self.preprocessed_dir, exist_ok=True)
                gt.to_pickle(self.preprocessed_gt_paths[seq])
            else:
                dets = pd.read_pickle(self.preprocessed_paths[seq])
            
            for i, row in dets.iterrows():
                if row['id'] not in seq_ids.keys():
                    seq_ids[row['id']] = self.id
                    self.id += 1
                dets.at[i, 'id'] = seq_ids[row['id']]

            self.id_to_y[seq] = seq_ids

            #if 'vis' in dets and dets['vis'].unique() != [-1]:
            #    dets = dets[dets['vis'] > self.dataset_cfg['gt_training_min_vis']]

            self.data.append(dets)

        self.id += 1
        self.seperate_seqs()
        
        if self.mode == 'test':
            self.make_query_gal()

    def seperate_seqs(self):
        samples = list()
        ys = list()
        for i, seq in enumerate(self.data):
            if i == 0:
                seq_all = seq
            else:
                seq_all = seq_all.append(seq)

        seq_all.reset_index(drop=True, inplace=True)
        indices = list(range(seq_all['frame'].count()))
        seq_all.reindex(indices)
        self.seq_all = seq_all

    def make_query_gal(self):
        self.different_gallery_set = False
        self.queries = self.seq_all
        self.query_indices = list()
        self.galleries = self.seq_all
        if self.vis_thresh != 0:
            self.different_gallery_set = True
            if type(self.vis_thresh) != tuple:
                self.queries = self.queries[self.queries['vis'] == self.vis_thresh]
            else:
                print(self.queries)
                self.queries = self.queries[self.queries['vis'] >= self.vis_thresh[0]]   #vis_thresh = [0.5, 0.6]
                self.queries = self.queries[self.queries['vis'] < self.vis_thresh[1]] 
                self.query_indices = self.queries.index.to_numpy()

        if self.size_thresh != 0:
            self.different_gallery_set = True
            if type(self.size_thresh) != tuple:
                self.queries = self.queries[self.queries['bb_height'] >= self.size_thresh]
            else:
                self.queries = self.queries[self.queries['bb_height'] >= self.size_thresh[0]]   #vis_thresh = [50, 75]
                self.queries = self.queries[self.queries['bb_height'] < self.size_thresh[1]] 
        
        if self.frame_dist_thresh != 0:
            frame_rate = int(self.data[0].attrs['frameRate'])
            self.gallery_mask = list()
            for ind, q in self.queries.iterrows():
                q_map11 = self.galleries['frame'] >= q['frame'] + (self.frame_dist_thresh[0] * frame_rate)
                q_map12 = self.galleries['frame'] <= q['frame'] - (self.frame_dist_thresh[0] * frame_rate)
                q_map21 = self.galleries['frame'] < q['frame'] + (self.frame_dist_thresh[1] * frame_rate)
                q_map22 = self.galleries['frame'] > q['frame'] - (self.frame_dist_thresh[1] * frame_rate)
                q_map11, q_map12, q_map21, q_map22 = np.asarray(q_map11), np.asarray(q_map12), np.asarray(q_map21), np.asarray(q_map22)
                q_map = (q_map11 & q_map21) | (q_map12 & q_map22)
                self.gallery_mask.append(q_map)
            self.gallery_mask = np.asarray(self.gallery_mask)

        if self.size_diff_thresh != 0:
            self.gallery_mask = list()
            for ind, q in self.queries.iterrows():
                q_map1 = self.galleries['bb_height'] <= (1+self.size_diff_thresh[0]) * q['bb_height']
                q_map2 = self.galleries['bb_height'] > (1+self.size_diff_thresh[1]) * q['bb_height']
                q_map1, q_map2 = np.asarray(q_map1), np.asarray(q_map2)
                q_map = q_map1 & q_map2
                self.gallery_mask.append(q_map)
            self.gallery_mask = np.asarray(self.gallery_mask)

        if self.data_type == 'query':
            print(self.galleries.shape, self.queries.shape)
            self.data = self.queries
        elif self.data_type == 'gallery':
            self.data = self.galleries

    def get_bounding_boxe(self, row):
        # tracktor resize (256,128))

        img = self.to_tensor(Image.open(row['frame_path']).convert("RGB"))
        img = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]
        img = self.to_pil(img)
        img = self.transform(img)

        return img

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        y = row['id']
        index = self.query_indices[idx]
        img = self.get_bounding_boxe(row)
        return img, index, y