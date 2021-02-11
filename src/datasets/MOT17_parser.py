import os.path as osp
import os
import pandas as pd
from torchvision.ops import box_iou 
import scipy
#from torch_geometric.data import Dataset, Data
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import torch
from lapsolver import solve_dense
from torchvision.transforms import ToTensor
from torchvision import transforms
from ReID.dataset.utils import make_transform_bot


class MOTLoader():
    def __init__(self, sequence, dataset_cfg, dir):
        self.dataset_cfg = dataset_cfg
        self.sequence = sequence

        self.mot_dir = osp.join(dataset_cfg['mot_dir'], dir)
        self.det_dir = osp.join(dataset_cfg['det_dir'], dir)
        self.det_file = dataset_cfg['det_file']

    def get_seqs(self):

        for s in self.sequence:
            gt_file = osp.join(self.mot_dir, s, 'gt', 'gt.txt')
            det_file = osp.join(self.det_dir, s, 'det', self.det_file)
            seq_file = osp.join(self.mot_dir, s, 'seqinfo.ini')
            
            self.get_seq_info(seq_file, gt_file, det_file)
            self.get_dets(det_file, s)
            self.get_gt(gt_file)
            
            self.clip_boxes_to_image(df=self.dets)
            self.dets.sort_values(by = 'frame', inplace = True)
            self.dets['detection_id'] = np.arange(self.dets.shape[0])
            self.assign_gt()
            self.dets.attrs.update(self.seq_info)

    def get_gt(self, gt_file):
        if osp.exists(gt_file):
            self.gt = pd.read_csv(gt_file, names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis'])
            self.gt['bb_left'] -= 1  # Coordinates are 1 based
            self.gt['bb_top'] -= 1
            self.gt['bb_bot'] = (self.gt['bb_top'] + self.gt['bb_height']).values
            self.gt['bb_right'] = (self.gt['bb_left'] + self.gt['bb_width']).values

            self.gt = self.gt[self.gt['label'].isin([1, 2, 7, 8, 12])].copy()
        else:
            self.gt = None
            
    def get_dets(self, det_file, s):
        img_dir = osp.join(self.mot_dir, s, 'img1')

        if osp.exists(det_file):
            self.dets = pd.read_csv(det_file, names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis', '?'])
            self.dets['bb_left'] -= 1 # Coordinates are 1 based
            self.dets['bb_top'] -= 1
            self.dets['bb_right'] = (self.dets['bb_left'] + self.dets['bb_width']).values
            self.dets['bb_bot'] = (self.dets['bb_top'] + self.dets['bb_height']).values

            if len(self.dets['id'].unique()) > 1:
                self.dets['tracktor_id'] = self.dets['id']
             
            # add frame path
            add_frame_path = lambda i: osp.join(osp.join(img_dir, f"{i:06d}.jpg"))
            self.dets['frame_path'] = self.dets['frame'].apply(add_frame_path)

    def get_seq_info(self, seq_file, gt_file, det_file):
        seq_info = pd.read_csv(seq_file, sep='=').to_dict()['[Sequence]']
        seq_info['path'] = osp.dirname(seq_file)
        seq_info['has_gt'] = osp.isfile(gt_file)
        seq_info['is_gt'] = det_file == gt_file
        self.seq_info = seq_info

    def clip_boxes_to_image(self, df):
        img_height, img_width = self.seq_info['imHeight'], self.seq_info['imWidth']
        img_height = np.array([int(img_height)]* df.shape[0])
        img_width = np.array([int(img_width)]* df.shape[0])

        # top and left 
        initial_bb_top = df['bb_top'].values.copy()
        initial_bb_left = df['bb_left'].values.copy()
        df['bb_top'] = np.maximum(df['bb_top'].values, 0).astype(int)
        df['bb_left'] = np.maximum(df['bb_left'].values, 0).astype(int)

        # bottom and right
        initial_bb_bot = df['bb_bot'].values.copy()
        initial_bb_right = df['bb_right'].values.copy()
        df['bb_bot'] = np.minimum(img_height, df['bb_bot']).astype(int)
        df['bb_right'] = np.minimum(img_width, df['bb_right']).astype(int)

        # width and height
        bb_top_diff = df['bb_top'].values - initial_bb_top
        bb_left_diff = df['bb_left'].values - initial_bb_left
        df['bb_height'] -= bb_top_diff
        df['bb_width'] -= bb_left_diff
        df['bb_height'] = np.minimum(img_height - df['bb_top'], df['bb_height']).astype(int)
        df['bb_width'] = np.minimum(img_width - df['bb_left'], df['bb_width']).astype(int)

        # double check
        conds = (df['bb_width'] > 0) & (df['bb_height'] > 0)
        conds = conds & (df['bb_right'] > 0) & (df['bb_bot'] > 0)
        conds  =  conds & (df['bb_left'] < img_width) & (df['bb_top'] < img_height)
        df = df[conds].copy()
        
        return df

    def assign_gt(self):
        if self.seq_info['has_gt']:
            for frame in self.dets['frame'].unique():
                # get df entries of current frame 
                frame_detects = self.dets[self.dets.frame == frame]
                frame_gt = self.gt[self.gt.frame == frame]

                # Compute IoU for each pair of detected / GT bounding box
                iou_matrix = box_iou(torch.tensor(frame_detects[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values),
                                 torch.tensor(frame_gt[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values))
                iou_matrix[iou_matrix < self.dataset_cfg['gt_assign_min_iou']] = np.nan
                dist = 1 - iou_matrix
                assigned_detects, corresponding_gt = solve_dense(dist)
                unassigned_detect = np.array(list(set(range(frame_detects.shape[0])) - set(assigned_detects)))
                
                # get indices of assigned and unassigned frames
                assigned_detect_index = frame_detects.iloc[assigned_detects].index
                unassigned_detect_index = frame_detects.iloc[unassigned_detect].index
                
                # get IDs of assigned gt
                corresponding_gt = frame_gt.iloc[corresponding_gt]['id'].values
                
                # set IDs of assigned and unassigned
                self.dets.loc[assigned_detect_index, 'id'] = corresponding_gt
                self.dets.loc[unassigned_detect_index, 'id'] = -1  # False Positives
        

class MOTDataset(Dataset):
    def __init__(self, split, sequences, dataset_cfg, dir, datastorage='data'):
        super(MOTDataset, self).__init__()
        self.split = split
        self.sequences = self.add_detector(sequences, dataset_cfg['detector'])
        self.dataset_cfg = dataset_cfg
        self.dir = dir
        self.datastorage = datastorage
        self.data = list()
        self.id = 0
        
        self.to_tensor = ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.transform = make_transform_bot(is_train=False)

        self.process()
    
    def add_detector(self, sequence, detector):
        if detector == 'all':
            dets = ('DPM', 'FRCNN', 'SDP')
            sequence = ['-'.join([s, d]) for s in sequence for d in dets]
        elif detector == '':
            pass
        else:
            sequence = ['-'.join([s, detector]) for s in sequence]

        return sequence
    
    @property
    def preprocessed_paths(self):
        return {s: osp.join(self.preprocessed_dir, s+'.pkl') for s in self.sequences}

    @property
    def preprocessed_gt_paths(self):
        return {s: osp.join(self.preprocessed_dir, s+'_gt.pkl') for s in self.sequences}
    
    @property
    def preprocessed_dir(self):
        return osp.join(self.datastorage, self.split)

    @property
    def preprocessed_exists(self):
        for p in self.preprocessed_paths.values():
            if not osp.isdir(p):
                return False
        return True

    def process(self):
        self.seqs_by_names = dict()
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
            
            self.seqs_by_names[seq] = dets
            self.get_n_frame_graphs(dets)
        
        self.id += 1

    def get_n_frame_graphs(self, dets, n=2):
        #print(dets)
        for i in dets['frame'].unique():
            self.data.append((dets.attrs['name'], [i + j for j in range(n)]))
        
    def get_bounding_boxes(self, nodes):
        # tracktor resize (256,128)
        res = list()
        ids = list()
        
        for i in nodes['frame'].unique():
            frame = nodes[nodes['frame'] == i]
            assert len(frame['frame_path'].unique()) == 1
            img = self.to_tensor(Image.open(frame['frame_path'].unique()[0]).convert("RGB"))
            for ind, row in frame.iterrows():
                im = img[:, row['bb_top']:row['bb_bot'], row['bb_left']:row['bb_right']]
                im = self.to_pil(im)
                im = self.transform(im)
                res.append(im)
                ids.append(row['id'])
        
        res = torch.stack(res, 0)

        return res, ids

    def __len__(self):
        return len(self.data)

    def _get_num_ids(self):
        return self.id

    def __getitem__(self, idx):
        seq_name, frames = self.data[idx][0], self.data[idx][1]
        dets = self.seqs_by_names[seq_name]
        
        nodes = pd.concat([dets[dets['frame'] == i] for i in frames])
        bounding_boxes, ids = self.get_bounding_boxes(nodes)
        
        return bounding_boxes, torch.tensor(ids)


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


class TrackingDataset(MOTDataset):
    def __init__(self, split, sequences, dataset_cfg, dir, datastorage='data'):
        super(TrackingDataset, self).__init__(split, sequences, dataset_cfg, dir, datastorage)

    def process(self):
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
            
            self.data.append(Sequence(name=seq, dets=dets, gt=gt, 
                                        to_pil=self.to_pil, 
                                        to_tensor=self.to_tensor, 
                                        transform=self.transform))
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
        res = res.cuda()
    
        return res, dets

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        seq = self.data[idx]
        '''
        # seq_dict.keys() = [name, dets, dt]
        seq_dict = self.data[idx]
        dets, gt = seq_dict['dets'], seq_dict['gt']

        seq_imgs = list()
        seq_gt = list()
        seq_dets = list()
        seq_paths = list()
        
        for frame in dets['frame'].unique():
            gt_frame = gt[gt['frame']==frame]
            dets_frame = dets[dets['frame']==frame]
            print(dets_frame)
            quit()
            assert len(dets_frame['frame_path'].unique()) == 1
            
            img, dets_f = self._get_images(dets_frame['frame_path'].unique()[0], dets_frame)
            seq_imgs.append(img)
            seq_dets.append(dets_f)

            gt_f = {row['id']: np.array([row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']], dtype=np.float32) for i, row in gt_frame.iterrows()}
            seq_gt.append(gt_f)

            seq_paths.append(dets_frame['frame_path'].unique()[0])
        '''

        return [seq] #seq_imgs, seq_gt, seq_paths, seq_dets


class Sequence():
    def __init__(self, name, dets, gt, to_pil, to_tensor, transform):
        self.dets = dets
        self.gt = gt
        self.name = name
        self.to_pil = to_pil
        self.to_tensor = to_tensor
        self.transform = transform
        self.num_frames = len(self.dets['frame'].unique())

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
    
    def __iter__(self):
        self.frames = self.dets['frame'].unique()
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < self.num_frames:
            frame = self.frames[self.i]
            print(frame)
            gt_frame = self.gt[self.gt['frame']==frame]
            dets_frame = self.dets[self.dets['frame']==frame]

            assert len(dets_frame['frame_path'].unique()) == 1
            
            img, dets_f = self._get_images(dets_frame['frame_path'].unique()[0], dets_frame)

            gt_f = {row['id']: np.array([row['bb_left'], row['bb_top'], row['bb_right'], row['bb_bot']], dtype=np.float32) for i, row in gt_frame.iterrows()}

            self.i += 1
            return img, gt_f, dets_frame['frame_path'].unique()[0], dets_f
        else:
            raise StopIteration


def collate_train(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    return [data, target]


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
  