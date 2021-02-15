import torch
import os
import os.path as osp
from .MOTDataset import MOTDataset
from .MOT17_parser import MOTLoader
import pandas as pd
import PIL.Image as Image
from .utils import ClassBalancedSampler


def collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    visibility = [item[2] for item in batch]
    im_path = [item[3] for item in batch]
    dets = [item[4] for item in batch]

    return [data, target, visibility, im_path, dets]


class ReIDDataset(MOTDataset):
    def __init__(self, split, sequences, dataset_cfg, dir, datastorage='data'):
        super(ReIDDataset, self).__init__(split, sequences, dataset_cfg, dir, datastorage)
    
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

            if 'vis' in dets and dets['vis'].unique() != [-1]:
                dets = dets[dets['vis'] > self.dataset_cfg['gt_training_min_vis']]

            self.data.append(dets)

        self.id += 1
        self.seperate_seqs()

    def seperate_seqs(self):
        samples = list()
        ys = list()
        for seq in self.data:
            for index, row in seq.iterrows():
                samples.append(row)
                ys.append(row['id'])
        self.data = samples
        self.ys = ys

    def get_bounding_boxe(self, row):
        # tracktor resize (256,128))

        img = self.to_tensor(Image.open(row['frame_path']).convert("RGB"))
        img = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]
        img = self.to_pil(img)
        img = self.transform(img)

        return img

    def __getitem__(self, idx):
        row, y = self.data[idx], self.ys[idx]
        img = self.get_bounding_boxe(row)

        return img, y