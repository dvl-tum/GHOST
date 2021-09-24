import os
import torch
from .MOTDataset import MOTDataset
from .MOT17_parser import MOTLoader
import pandas as pd 
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from collections import defaultdict


class ProxyDataset(MOTDataset):
    def __init__(self, split, sequences, dataset_cfg, dir, datastorage='data'):
        super(ProxyDataset, self).__init__(split, sequences, dataset_cfg, dir, datastorage)

    def process(self):
        self.seq_ids_to_id = defaultdict(dict)
        self.id = 0
        self.data = list()
        for seq in self.sequences:
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
                gt = pd.read_pickle(self.preprocessed_gt_paths[seq])

            if 'vis' in dets and dets['vis'].unique() != [-1]:
                dets = dets[dets['vis'] > self.dataset_cfg['gt_training_min_vis']]
            
            self.data.extend(self._get_id_seqs(name=seq, dets=dets))
            #self.data.append({'name': seq, 'dets': dets, 'gt': gt})
    
    def _get_id_seqs(self, name, dets):
        id_seqs = list()
        for i in dets['id'].unique():
            if not i in self.seq_ids_to_id[name].keys():
                self.seq_ids_to_id[name][i] = self.id
            id_rows = dets[dets['id'] == i]
            id_rows.loc[:, 'id'] = self.seq_ids_to_id[name][i]
            id_seqs.append(id_rows)
            self.id += 1

        return id_seqs

    def _get_images(self, id_seq):
        res = list()
        ys = list()
        for i, row in id_seq.iterrows():
            path = row['frame_path']
            img = self.to_tensor(Image.open(path).convert("RGB"))
            img = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]
            img = self.to_pil(img)
            img = self.transform(img)
            res.append(img)
            ys.append(row['id'])
    
        res = torch.stack(res, 0)
        res = res.cuda()
        ys = torch.tensor(ys).cuda()
    
        return res, ys

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        id_seq = self.data[idx]
        imgs, ys = self._get_images(id_seq)

        return imgs, ys