import os
import os.path as osp
import PIL.Image as Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from ReID.dataset.utils import make_transform_bot
from torch.utils.data import Dataset
import pandas as pd
import torch
from .MOT17_parser import MOTLoader


class MOTDataset(Dataset):
    def __init__(self, split, sequences, dataset_cfg, dir, datastorage='data', add_detector=True):
        super(MOTDataset, self).__init__()
        self.split = split
        if add_detector:
            self.sequences = self.add_detector(sequences, dataset_cfg['detector'])
        else:
            self.sequences = sequences
        
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