from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp

from torchreid.data import ImageDataset
import tqdm

import pandas as pd
import numpy as np
import json
import math
import random

from . import utils
import PIL.Image


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def anns2df(anns, img_dir):
    # Build DF from anns
    to_kps = lambda x: np.array(x['keypoints']).reshape(-1, 3)
    rows = []
    for ann in tqdm.tqdm(anns['annotations']):
        row = {
            'path': f"{img_dir}/{ann['id']}.png",
            'model_id': int(ann['model_id']),
            'height': int(ann['bbox'][-1]),
            'width': int(ann['bbox'][-2]),
            'iscrowd': int(ann['iscrowd']),
            'isnight': int(ann['is_night']),
            'vis': (to_kps(ann)[..., 2] == 2).mean(),
            'frame_n': int(ann['frame_n']),
            **{f'attr_{i}': int(attr_val) for i, attr_val in enumerate(
                ann['attributes'])}}
        rows.append(row)

    return pd.DataFrame(rows)


def anns2df_motcha(anns, img_dir):
    # Build DF from anns
    rows = []
    for ann in tqdm.tqdm(anns['annotations']):
        row = {
            'path': f"{osp.join(img_dir)}/{ann['id']}.png",
            'ped_id': int(ann['ped_id']),
            'height': int(ann['bbox'][-1]),
            'width': int(ann['bbox'][-2]),
            'iscrowd': int(ann['iscrowd']),
            'vis': float(ann['vis']),
            'frame_n': int(ann['frame_n'])}
        rows.append(row)

    return pd.DataFrame(rows)


def assign_ids(df, night_id=True, attr_indices=[0, 2, 3, 4, 7, 8, 9, 10]):
    id_cols = ['model_id'] + [
        f'attr_{i}' for i in attr_indices if f'attr{i}' in df.columns] 
    if night_id and 'isnight' in df.columns:
        id_cols += ['isnight']

    unique_ids_df = df[id_cols].drop_duplicates()
    unique_ids_df['reid_id'] = np.arange(unique_ids_df.shape[0])

    return df.merge(unique_ids_df)


def clean_rows(df, min_vis, min_h, min_w, min_samples):
    # Filter by size and occlusion
    keep = (df['vis'] >= min_vis) & (
        df['height'] >= min_h) & (df['width'] >= min_w) & (df['iscrowd'] == 0)
    clean_df = df[keep]
    # Keep only ids with at least MIN_SAMPLES appearances
    clean_df['samples_per_id'] = clean_df.groupby(
        'reid_id')['height'].transform('count').values
    clean_df = clean_df[clean_df['samples_per_id'] >= min_samples]

    return clean_df

def relabel_ids(df):
    df.rename(columns = {'reid_id': 'reid_id_old'}, inplace=True)
    df['old_id_seq'] = [seq + "_" + str(i) for seq, i in zip(df['Sequence'].values, df['reid_id_old'].values)]

    # Relabel Ids from 0 to N-1
    ids_df = df[['old_id_seq']].drop_duplicates()
    ids_df['reid_id'] = np.arange(ids_df.shape[0])
    df = df.merge(ids_df)

    return df


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


class MOTSeqDataset(ImageDataset):
    def __init__(self, ann_files, img_dir, min_vis=0.3, min_h=50, min_w=25,
            min_samples=15, night_id=True, motcha=False, split='split_1',
            seq_names=None, mode='train', **kwargs):
        np.random.seed(0)

        split = split.split('+')
        if len(split)> 1:
            split_seq = split[1]
        else:
            split_seq = None
        split = split[0]
        
        for i, (ann_file, seq_name) in enumerate(zip(ann_files, seq_names)):
            # Create a Pandas DataFrame out of json annotations file
            print("Reading json...")
            anns = read_json(ann_file)
            print("Done!")
            
            print("Preparing dataset...")
            if motcha:
                df = anns2df_motcha(anns, img_dir)
                df['reid_id'] = df['ped_id']

            else:
                df = anns2df(anns, img_dir)
                df = assign_ids(
                    df,
                    night_id=True,
                    attr_indices=[0, 2, 3, 4, 7, 8, 9, 10])

            df = clean_rows(
                df,
                min_vis,
                min_h=min_h,
                min_w=min_w,
                min_samples=min_samples)

            df['Sequence'] = seq_name
            if i == 0:
                df_all = df
            else:
                df_all = df_all.append(df)

        df = relabel_ids(df_all)

        # For testing, choose one apperance randomly for every track and 
        # put in the gallery
        to_tuple_list = lambda df: list(
            df[['path', 'reid_id', 'cam_id']].itertuples(
                index=False, name=None))

        df['cam_id'] = 0

        if '50-50' not in split:
            print(split_dict[split]['test'], split_dict[split]['train'])
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

        test['index'] = test.index.values

        query_per_id = test.groupby('reid_id')['index'].agg(
            lambda x: np.random.choice(list(x.unique())))
        query_df = test.loc[query_per_id.values].copy()
        gallery_df = test.drop(query_per_id).copy()

        if split_seq:
            print('splitting for 50-50 per seq evaluation')
            test_seqs = ["MOT17-{:02d}".format(seq) for seq in split_dict[
                split_seq]['test']]
            train_df = train_df[~train_df['Sequence'].isin(test_seqs)]
            query_df = query_df[query_df['Sequence'].isin(test_seqs)]
            gallery_df = gallery_df[gallery_df['Sequence'].isin(test_seqs)]


        train = to_tuple_list(train_df)
        # IMPORTANT: For testing, torchreid only compares gallery and query
        # images from different cam_ids
        # therefore we just assign them ids 0 and 1 respectively
        gallery_df['cam_id'] = 1
        
        query = to_tuple_list(query_df)
        gallery = to_tuple_list(gallery_df)
        
        if mode == 'test':
            self.gallery_paths = gallery_df['path'].tolist()
            self.query_paths = query_df['path'].tolist()

            self.gallery_ys = gallery_df['reid_id'].tolist()
            self.query_ys = query_df['reid_id'].tolist()

            data = gallery + query
            self.im_paths = self.gallery_paths + self.query_paths
            self.ys = self.gallery_ys + self.query_ys
        elif mode == 'train':
            data = train
            self.im_paths = train_df['path']
            self.ys = train_df['reid_id']

        print("Done!")
        super(MOTSeqDataset, self).__init__(train, query, gallery, **kwargs)
        self.data = data

def get_sequence_class(seq_names=None, split='split_1', eval_reid=False):
    if seq_names is None:
        seq_names = [
            "MOT17-02",
            "MOT17-04",
            "MOT17-05",
            "MOT17-09",
            "MOT17-10",
            "MOT17-11",
            "MOT17-13"]

    # raise RuntimeError("MODify this path to wherever you store json annotations and reid data!!")
    ann_files = [f'/storage/remote/atcremers82/mot_neural_solver/sanity_check_data/red_annotations/MOT17_{seq_name}.json' for seq_name in seq_names]
    img_dir = '/storage/remote/atcremers82/mot_neural_solver/sanity_check_data/reid'
    min_samples = 5
    motcha=True

    class MOTSpecificSeq(MOTSeqDataset):
        def __init__(self, **kwargs):
            super(MOTSpecificSeq, self).__init__(ann_files=ann_files, img_dir=img_dir,
                min_samples=min_samples, motcha=motcha, split=split, seq_names=seq_names,
                rand_scales=None, **kwargs)
            print("Using split {}".format(split))
            self.eval_reid = eval_reid
            self.transform = utils.make_transform(is_train=not self.eval_reid)
            self.no_imgs = None
            self.distractor_idx = None
            self.rand_scales = None

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

    MOTSpecificSeq.__name__ = split

    return MOTSpecificSeq


if __name__ == '__main__':
    dataset = get_sequence_class(split='50-50-1')
    dataset = dataset()
