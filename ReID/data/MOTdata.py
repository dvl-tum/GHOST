from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp

import pandas as pd
import numpy as np
import json
import math
import random

import os
from data.ReIDdata import ReIDDataset, pil_loader
from zipfile import ZipFile


split_dict = {'split_1': {'train': (2, 5, 9, 10, 13), 'test': (4, 11)},
              'split_2': {'train': (2, 4, 11, 10, 13), 'test': (5, 9)},
              'split_3': {'train': (4, 5, 9, 11), 'test': (2, 10, 13)},
              'split_S2': {'train': (4, 5, 9, 10, 11, 13), 'test': [2]},
              'split_S4': {'train': (2, 5, 9, 10, 11, 13), 'test': [4]},
              'split_S5': {'train': (2, 4, 9, 10, 11, 13), 'test': [5]},
              'split_S9': {'train': (2, 4, 5, 10, 11, 13), 'test': [9]},
              'split_S10': {'train': (2, 4, 5, 9, 11, 13), 'test': [10]},
              'split_S11': {'train': (2, 4, 5, 9, 10, 13), 'test': [11]},
              'split_S13': {'train': (2, 4, 5, 9, 10, 11), 'test': [13]}}


class MOTReIDDataset(ReIDDataset):
    
    def __init__(
            self,
            root: str,
            split: str,
            min_samples=5,
            seq_names=None,
            rand_scales=None,
            min_vis=0.3,
            min_h=50,
            min_w=25,
            mode='train',
            trans=None,
            sz_crop=[128, 64],
            eval_reid=False):
        
        # check if zip file extracted
        self.extract_zip(root)
        
        self.img_dir = os.path.join(root, 'images')
        if seq_names is None:
            self.seq_names = [
                "MOT17-02",
                "MOT17-04",
                "MOT17-05",
                "MOT17-09",
                "MOT17-10",
                "MOT17-11",
                "MOT17-13"]
        else:
            self.seq_names = seq_names
        ann_files = [os.path.join(root, 'annotations', f'MOT17_{seq_name}.json')\
             for seq_name in self.seq_names]

        print("Using split {}".format(split))
        self.__name__ = split
        self.eval_reid = eval_reid
        self.transform = self.get_transform(trans, sz_crop=sz_crop)
        self.no_imgs = None
        self.distractor_idx = None
        self.rand_scales = rand_scales

        # get sequences
        df_all = self.get_sequencess(ann_files, min_vis, min_h, min_w, min_samples)

        # relabel ids over different seq from 0 - num ids
        df = self.relabel_ids(df_all)

        # split into train and test 
        train_df, query_df, gallery_df = self.split_data(split, df)

        # set gallery cam_id to different id than rest 
        # --> need to be different in evaluation code
        gallery_df['cam_id'] = 1

        # make data lists for iteration over dataset
        self.make_data_list(train_df, query_df, gallery_df, mode)

        print("Done!")

    def get_sequencess(self, ann_files, min_vis, min_h, min_w, min_samples):
        np.random.seed(0)        
        for i, (ann_file, seq_name) in enumerate(zip(ann_files, self.seq_names)):

            # Create a Pandas DataFrame out of json annotations file
            with open(ann_file) as json_file:
                anns = json.load(json_file)
            
            df = self.anns2df_motcha(anns, self.img_dir)
            df = self.clean_rows(
                df,
                min_vis,
                min_h=min_h,
                min_w=min_w,
                min_samples=min_samples,
                seq_name=seq_name)

            if i == 0:
                df_all = df
            else:
                df_all = df_all.append(df)

        return df_all
    
    def split_data(self, split, df):
        if '50-50' not in split:
            test_seqs = ["MOT17-{:02d}".format(seq) for seq in split_dict[split]['test']]
            train_df = df[~df['Sequence'].isin(test_seqs)]
            test = df[df['Sequence'].isin(test_seqs)]
    
        else:
            test_id = df.groupby('Sequence')['reid_id'].apply(lambda x:
                np.random.choice(
                    x.unique(),
                    size=math.ceil(x.unique().shape[0]*0.5)))
            test_id = [i for ids in test_id.values for i in ids]
            if split == '50-50-1':
                train_df = df[~df['reid_id'].isin(test_id)]
                test = df[df['reid_id'].isin(test_id)]
            else:
                train_df = df[df['reid_id'].isin(test_id)]
                test = df[~df['reid_id'].isin(test_id)]
        
        # get query and test dfs by randomly sampling one frame per id
        test['index'] = test.index.values
        query_per_id = test.groupby('reid_id')['index'].agg(
            lambda x: np.random.choice(list(x.unique())))
        query_df = test.loc[query_per_id.values].copy()
        gallery_df = test.drop(query_per_id).copy()
        
        return train_df, query_df, gallery_df
    
    def make_data_list(self, train_df, query_df, gallery_df, mode):
        # convert dfs to tuples ['path', 'reid_id', 'cam_id']
        train = self.to_tuple_list(train_df)
        query = self.to_tuple_list(query_df)
        gallery =self. to_tuple_list(gallery_df)
        
        if mode == 'test':
            self.gallery_paths = gallery_df['path'].tolist()
            self.query_paths = query_df['path'].tolist()

            self.gallery_ys = gallery_df['reid_id'].tolist()
            self.query_ys = query_df['reid_id'].tolist()

            self.data = gallery + query
            self.im_paths = self.gallery_paths + self.query_paths
            self.ys = self.gallery_ys + self.query_ys
        elif mode == 'train':
            self.data = train
            self.im_paths = train_df['path']
            self.ys = train_df['reid_id']
            
    @staticmethod
    def to_tuple_list(df): 
        return list(df[['path', 'reid_id', 'cam_id']].itertuples(
                index=False, name=None))

    @staticmethod
    def clean_rows(df, min_vis, min_h, min_w, min_samples, seq_name):
        # Filter by size and occlusion
        keep = (df['vis'] >= min_vis) & (
            df['height'] >= min_h) & (df['width'] >= min_w) & (df['iscrowd'] == 0)
        clean_df = df[keep]
        
        # Keep only ids with at least MIN_SAMPLES appearances
        clean_df['samples_per_id'] = clean_df.groupby(
            'reid_id')['height'].transform('count').values
        clean_df = clean_df[clean_df['samples_per_id'] >= min_samples]
        clean_df['Sequence'] = seq_name
        clean_df['cam_id'] = 0

        return clean_df

    @staticmethod
    def anns2df_motcha(anns, img_dir):
        # Build DF from anns
        rows = []
        for ann in anns['annotations']:
            row = {
                'path': f"{osp.join(img_dir)}/{ann['id']}.png",
                'ped_id': int(ann['ped_id']),
                'height': int(ann['bbox'][-1]),
                'width': int(ann['bbox'][-2]),
                'iscrowd': int(ann['iscrowd']),
                'vis': float(ann['vis']),
                'frame_n': int(ann['frame_n'])}
            rows.append(row)

        #  rename ped_id to reid_id
        df = pd.DataFrame(rows)
        df['reid_id'] = df['ped_id']

        return df
    
    @staticmethod
    def relabel_ids(df):
        df.rename(columns = {'reid_id': 'reid_id_old'}, inplace=True)
        df['old_id_seq'] = [seq + "_" + str(i) for seq, i in zip(df['Sequence'].values, df['reid_id_old'].values)]

        # Relabel Ids from 0 to N-1
        ids_df = df[['old_id_seq']].drop_duplicates()
        ids_df['reid_id'] = np.arange(ids_df.shape[0])
        df = df.merge(ids_df)

        return df
    
    @staticmethod
    def extract_zip(root):
        # check if root directory alredy extracted
        if not osp.isdir(root):
            # check if zip file there
            check_zip = root + '.zip'
            if not os.path.isfile(check_zip) and not os.path.isdir(root):
                path = 'https://vision.in.tum.de/webshare/u/seidensc/MOT17_ReID.zip'
                assert False, f'Please download dataset from {path} into dataset/...'

            # extract zip file
            print("Extracting zip file...")
            with ZipFile(check_zip) as z:
                z.extractall(os.path.dirname(check_zip))

    def __getitem__(self, index):
        # 'path', 'reid_id', 'cam_id'
        img_path, pid, camid = self.data[index]
        if self.no_imgs:
            return pid, index, (img_path, camid)
        im = pil_loader(img_path)

        if not self.eval_reid and self.rand_scales:
            if random.randint(0, 1):
                r = random.randint(2, 4)
                im = im.resize((int(im.size[0]/r), int(im.size[1]/r)))
        im = self.transform(im)

        return im, pid, index, (img_path, camid)

