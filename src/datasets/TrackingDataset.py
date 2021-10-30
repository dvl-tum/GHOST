import os

from numpy.core.fromnumeric import shape
import torch
from .MOTDataset import MOTDataset
from .MOT17_parser import MOTLoader
import pandas as pd 
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import csv
from torchvision.ops.boxes import clip_boxes_to_image
import random 
import torch.nn.functional as F


class TrackingDataset(MOTDataset):
    def __init__(self, split, sequences, dataset_cfg, dir, datastorage='data', dev=None):
        self.device = dev
        print(sequences)
        super(TrackingDataset, self).__init__(split, sequences, dataset_cfg, dir, datastorage)
        print(sequences)

    def _save_oracle(self, seq, dets):
        """
        Dets of format ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis', '?']

        Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        os.makedirs('gt_out', exist_ok=True)
         
        with open(os.path.join('gt_out', seq), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, row in dets.iterrows():
                writer.writerow(
                    [row['frame'],
                        row['id'],
                        row['bb_left'],
                        row['bb_top'],
                        row['bb_width'],
                        row['bb_height'],
                        -1, -1, -1, -1])

    def process(self):
        self.data = list()
        for seq in self.sequences:
            if not self.preprocessed_exists or self.dataset_cfg['prepro_again']:
                loader = MOTLoader([seq], self.dataset_cfg, self.dir)
                exist_gt = loader.get_seqs()
                
                dets = loader.dets
                dets_unclipped = loader.dets_unclipped
                pd.set_option('display.max_columns', None)
                os.makedirs(self.preprocessed_dir, exist_ok=True)
                dets.to_pickle(self.preprocessed_paths[seq])
                dets_unclipped.to_pickle(self.preprocessed_paths[seq][:-4] + '_unclipped.pkl')
                if exist_gt:
                    gt = loader.gt
                    corresponding_gt = loader.corresponding_gt
                    os.makedirs(self.preprocessed_dir, exist_ok=True)
                    gt.to_pickle(self.preprocessed_gt_paths[seq])
                    corresponding_gt.to_pickle(self.preprocessed_paths[seq][:-4] + '_corresponding.pkl')
                else:
                    corresponding_gt = None
                    gt = None
            else:
                dets = pd.read_pickle(self.preprocessed_paths[seq])
                dets_unclipped = pd.read_pickle(self.preprocessed_paths[seq][:-4] + '_unclipped.pkl')
                if exist_gt:
                    gt = pd.read_pickle(self.preprocessed_gt_paths[seq])
                    corresponding_gt = pd.read_pickle(self.preprocessed_paths[seq][:-4] + '_corresponding.pkl')
                else:
                    gt = None
                    corresponding_gt = None

            if self.dataset_cfg['save_oracle']:
                self._save_oracle(seq, dets)

            #if 'vis' in dets and dets['vis'].unique() != [-1]:
            #    dets = dets[dets['vis'] > self.dataset_cfg['gt_training_min_vis']]
            #    dets_unclipped = dets_unclipped[dets['vis'] > self.dataset_cfg['gt_training_min_vis']]

            self.data.append(Sequence(name=seq, dets=dets, gt=gt, 
                                        to_pil=self.to_pil, 
                                        to_tensor=self.to_tensor, 
                                        transform=self.transform,
                                        dev=self.device,
                                        dets_unclipped=dets_unclipped,
                                        corresponding_gt=corresponding_gt,
                                        transform_det=self.transform_det))
            #self.data.append({'name': seq, 'dets': dets, 'gt': gt})
        
        self.id += 1

    def _get_images(self, path, dets_frame):
        img = self.to_tensor(Image.open(path).convert("RGB"))
        res = list()
        dets = list()
        for ind, row in dets_frame.iterrows():
            im = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]
            im = self.to_pil(im)
            im = self.transform(im)
            res.append(im)
            dets.append(np.array([row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']], dtype=np.float32))
    
        res = torch.stack(res, 0)
        res = res.to(self.device)
    
        return res, dets

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        seq = self.data[idx]

        return [seq]


class Sequence():
    def __init__(self, name, dets, gt, to_pil, to_tensor, transform, dev=None, 
            dets_unclipped=None, corresponding_gt=None, zero_pad=True, transform_det=None,
            use_unclipped_for_eval=True):
        self.dets = dets
        self.dets_unclipped = dets_unclipped
        self.gt = gt
        self.corresponding_gt = corresponding_gt
        self.name = name
        self.to_pil = to_pil
        self.to_tensor = to_tensor
        self.transform = transform
        self.device = dev
        self.num_frames = len(self.dets['frame'].unique())
        self.zero_pad = zero_pad
        self.random_patches = False
        self.transform_det = transform_det
        self.use_unclipped_for_eval = use_unclipped_for_eval
        print("Zero padding of images {}".format(self.zero_pad))
        print("Using unclipped detections for evaluation {}".format(use_unclipped_for_eval))

    def _get_random_patches(self, img, height_max: int = 256, 
                height_min: int = 64, width_max: int = 256, 
                width_min: int =64, num_patches: int =  128, 
                frame_size: tuple = (560, 1024)):
        frame_size = (img.shape[1], img.shape[2])
        patches = list()
        heights = random.choices(range(height_min, height_max), k=num_patches)
        widths = random.choices(range(width_min, width_max), k=num_patches)
        for h, w in zip(heights, widths):
            x_pos = random.choice(range(1+int(w/2), frame_size[1]-int(w/2)))
            y_pos = random.choice(range(1+int(h/2), frame_size[0]-int(h/2)))

            im = img[:, y_pos:y_pos+h, x_pos:x_pos+w]
            im = self.to_pil(im)
            im = self.transform(im)
            patches.append(im)

        return torch.stack(patches, 0).to(self.device)

    def _get_images(self, path, dets_frame, dets_uncl_frame):
        img = self.to_tensor(Image.open(path).convert("RGB"))
        import copy
        img_for_det = copy.deepcopy(img)
        res = list()
        dets = list()
        tracktor_ids = list()
        ids = list()
        vis = list()

        if self.random_patches:
            random_patches = self._get_random_patches(img)
        else:
            random_patches = None
            
        for ind, row in dets_frame.iterrows():
            row_unclipped = dets_uncl_frame.loc[ind]

            im = img[:, int(row['bb_top']):int(row['bb_bot']), int(
                row['bb_left']):int(row['bb_right'])]
                        
            # pad if part of bb outside of image
            if self.zero_pad:
                left_pad = abs(int(row_unclipped['bb_left'])) if int(row_unclipped['bb_left']) < 0 else 0
                right_pad = abs(int(row_unclipped['bb_right']) - img.shape[2]) if int(row_unclipped['bb_right']) > img.shape[2] else 0
                top_pad = abs(int(row_unclipped['bb_top'])) if int(row_unclipped['bb_top']) < 0 else 0
                bot_pad = abs(int(row_unclipped['bb_bot']) - img.shape[1]) if int(row_unclipped['bb_bot']) > img.shape[1] else 0
                
                # zero padding
                m = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bot_pad))
                im = m(im)

                # image mean
                # im = F.pad(im, (left_pad, right_pad, top_pad, bot_pad), "constant", im.mean())

                # channel-wise mean
                # im = torch.stack([F.pad(im[i, :, :], (left_pad, right_pad, top_pad, bot_pad), "constant", im.mean(dim=1).mean(dim=1)[i]) for i in range(im.shape[0])])
                
                # others
                # left_pad = left_pad if left_pad < im.shape[2] else im.shape[2] - 1
                # right_pad = right_pad if right_pad < im.shape[2] else im.shape[2] - 1
                # top_pad = top_pad if top_pad < im.shape[1] else im.shape[1] - 1
                # bot_pad = bot_pad if bot_pad < im.shape[1] else im.shape[1] - 1
                # im = F.pad(im.unsqueeze(0), (left_pad, right_pad, top_pad, bot_pad), "circular").squeeze()

            im = self.to_pil(im)
            im = self.transform(im)
            res.append(im)

            if self.zero_pad or self.use_unclipped_for_eval:
                dets.append(np.array([row_unclipped['bb_left'], row_unclipped['bb_top'], row_unclipped[
                    'bb_right'], row_unclipped['bb_bot']], dtype=np.float32))
            else:
                dets.append(np.array([row['bb_left'], row['bb_top'], row[
                    'bb_right'], row['bb_bot']], dtype=np.float32))
            
            tracktor_ids.append(row['tracktor_id'])
            ids.append(row['id'])
            vis.append(row['vis'])
    
        res = torch.stack(res, 0)
        res = res.to(self.device)
    
        return res, dets, tracktor_ids, ids, vis, random_patches, img_for_det.to(self.device)
    
    def __iter__(self):
        self.frames = self.dets['frame'].unique()
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < self.num_frames:
            frame = self.frames[self.i]
            dets_frame = self.dets[self.dets['frame']==frame]
            dets_uncl_frame = self.dets_unclipped[self.dets_unclipped['frame']==frame]
            #print(dets_frame)
            #print(dets_uncl_frame)
            #quit()
            assert len(dets_frame['frame_path'].unique()) == 1
            
            img, dets_f, tracktor_ids, ids, vis, random_patches, img_for_det = self._get_images(dets_frame['frame_path'].unique()[0], dets_frame, dets_uncl_frame)

            if self.gt is not None:
                gt_frame = self.gt[self.gt['frame']==frame]
                gt_f = {row['id']: np.array([row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']], dtype=np.float32) for i, row in gt_frame.iterrows()}
            else: 
                gt_f = None

            self.i += 1
            return img, gt_f, dets_frame['frame_path'].unique()[0], dets_f, tracktor_ids, ids, vis, random_patches, img_for_det
        else:
            raise StopIteration


class Sequence2(Dataset):
    def __init__(self, name, dets, gt, to_pil, to_tensor, transform):
        self.dets = dets
        self.gt = gt
        self.name = name
        self.to_pil = to_pil
        self.to_tensor = to_tensor
        self.transform = transform

    def _get_images(self, path, dets_frame):
        img = self.to_tensor(Image.open(path).convert("RGB"))
        res = list()
        dets = list()
        for ind, row in dets_frame.iterrows():

            im = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]
            im = self.to_pil(im)
            im = self.transform(im)
            res.append(im)
            dets.append(np.array([row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']], dtype=np.float32))
    
        res = torch.stack(res, 0)
        res = res.cuda()
    
        return res, dets

    def __getitem__(self, idx):
        frame = self.frames[idx]
        gt_frame = self.gt[self.gt['frame']==frame]
        dets_frame = self.dets[self.dets['frame']==frame]

        assert len(dets_frame['frame_path'].unique()) == 1
        
        img, dets_f = self._get_images(dets_frame['frame_path'].unique()[0], dets_frame)

        gt_f = {row['id']: np.array([row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']], dtype=np.float32) for i, row in gt_frame.iterrows()}

        return img, gt_f, dets_frame['frame_path'].unique()[0], dets_f
