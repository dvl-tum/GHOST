from tracking_wo_bnw.src.tracktor.datasets import *
from ReID import net
import os.path as osp
import os
import shutil
from torch.utils.data import DataLoader


class Tracker():
    def __init__(self, config):
        self.config = config

        #load ReID net
        self.encoder, _ = net.load_net(
            self.config['reid_net']['trained_on']['name'],
            self.config['reid_net']['trained_on']['num_classes'], 'test',
            **self.config['reid_net']['encoder_params'])

        self.tracks = dict()

        if self.config['dataset']['tracktor_pre']['use']:
            self.use_preprocessed()

        self.dataset = factory.Datasets(self.config['dataset'])

        self.undo_preprocessed()

    def track(self):
        for seq in self.dataset:
            print(f"Tracking: {seq}")
            data_loader = DataLoader(seq, batch_size=1, shuffle=False)
            for i, frame in enumerate(data_loader):
                print(frame)


    def use_preprocessed(self):
        track_paths = self.config['dataset']['tracktor_pre']['path']
        dataset =  self.config['dataset']['name'].split('_')[0].upper()
        seq = self.config['dataset']['name'].split('_')[1]
        detector = self.config['dataset']['name'].split('_')[2]

        if dataset == 'MOT-17':
            train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                               'MOT17-10', 'MOT17-11', 'MOT17-13']
            test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
                              'MOT17-08', 'MOT17-12', 'MOT17-14']

        if seq == 'train':
            seq = train_sequences
            dir = 'train'
        elif seq == 'test':
            seq = test_sequences
            dir = 'test'
        else:
            seq = [f"{dataset}-{seq}"]
            dir = 'test' if seq in test_sequences else 'train'

        for s in seq:
            s_det = f"{s}-{detector}"
            self.s_dir = osp.join(self.dataset['path'], dataset, dir, s)
            self.d_dir = osp.join(self.dataset['path'], dataset, dir, s_det)
            t_dir = osp.join(self.dataset['path'], track_paths, dir, s_det)
            self.t_dir = ops.join(t_dir, 'tracktor_prepr_det.txt')

            shutil.move(osp.join(self.d_dir, 'det.txt'),
                        osp.join(self.d_dir, 'd_o.txt'))
            shutil.mode(self.t_dir, osp.join(self.d_dir, 'det.txt'))

    def undo_preprocessed(self):
        shutil.mode(osp.join(self.d_dir, 'det.txt'), self.t_dir)
        shutil.move(osp.join(self.d_dir, 'd_o.txt'),
                    osp.join(self.d_dir, 'det.txt'))




            

