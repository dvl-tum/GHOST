from tracking_wo_bnw.src.tracktor.datasets import factory
from ReID import net
import os.path as osp
import os
import shutil
from torch.utils.data import DataLoader
import ReID
from torchvision.ops.boxes import clip_boxes_to_image, nms
from ReID.dataset.utils import make_transform_bot
import PIL.Image
from torchvision import transforms
import torch
import sklearn.metrics
import scipy
from collections import defaultdict
import numpy as np
from tracking_wo_bnw.src.tracktor.utils import interpolate, get_mot_accum, \
    evaluate_mot_accums
from src.datasets.MOT import MOT17, collate
from data.splits import _SPLITS
from src.nets.proxy_gen import ProxyGenMLP, ProxyGenRNN



class Manager():
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, tracker_cfg):
        self.device = device

        #load ReID net
        self.encoder, sz_embed = ReID.net.load_net(
            reid_net_cfg['trained_on']['name'],
            reid_net_cfg['trained_on']['num_classes'], 'test',
            **reid_net_cfg['encoder_params'])
        
        self.encoder = self.encoder.to(self.device)
        self.proxy_gen = ProxyGenMLP(sz_embed)

        self.tracker(tracker_cfg)

        self.loaders = self.get_data_loaders(dataset_cfg)

    def get_data_loaders(self, dataset_cfg):
        loaders = dict()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            print(mode)
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            dir = _SPLITS[dataset_cfg['splits']][mode]['dir']
            dataset = MOT17(seqs, dataset_cfg, dir)

            loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=True,
                                       collate_fn=collate)
        return loaders

    def train(self):
        for data in self.loaders['train']:
            data, target, visibility = data
            

    def track(self):
        mot_accums = list()
        self.encoder.eval()
        for seq in self.dataset:
            self.tracks = defaultdict(list)
            self.inactive_tracks = defaultdict(list)
            print(f"Tracking: {seq}")
            data_loader = DataLoader(seq, batch_size=1, shuffle=False)
            id = 0
            for i, frame in enumerate(data_loader):
               tracks = list()
               # extract persons
               print(len(frame['dets'].shape), frame['dets'].shape)
               dets = frame['dets'].squeeze(0)
               print(dets.shape)
               img = frame['img']
               boxes = clip_boxes_to_image(dets, img.shape[-2:])
               imgs = self.build_crops(img, boxes)
               with torch.no_grad():
                   _, feats = self.encoder(imgs, output_option='plain')
               for f, b, p in zip(feats, boxes, frame['img_path']):
                   tracks.append({'bbox': b, 'feats': f, 'im_path': p, 'im_index': i})
               
               if i > 0:
                    x = torch.stack([t['feats'] for t in tracks])
                    num_detects = x.shape[0]
                    y = torch.stack([v[-1]['feats'] for v in self.tracks.values()])
                    ids = list(self.tracks.keys())
                    if len(self.inactive_tracks) > 0:
                        y_inactive = torch.stack([v[-1]['feats'] for v in self.inactive_tracks.values()])
                        y = torch.stack([y, y_inactive])
                        ids += [self.inactive_tracks.keys()]
                    dist = sklearn.metrics.pairwise_distances(x.cpu().numpy(), y.cpu().numpy())
                    '''if dist.shape[0] != dist.shape[1]:
                        if dist.shape[0] > dist.shape[1]:
                            padding=((0,0),(0,dist.shape[0]-dist.shape[1]))
                        else:
                            padding=((0,dist.shape[1]-dist.shape[0]),(0,0))
                        dist = np.pad(dist,padding,mode='constant',constant_values=0)
                    ''' 
                    # row represent current frame, col represents last frame + inactiva tracks
                    row, col = scipy.optimize.linear_sum_assignment(dist)
               
               if i == 0:
                    for d in tracks:
                        self.tracks[id].append(d)
                        id += 1
               else:
                   for r in row:
                        if col[r] < len(ids):
                            if r < num_detects:
                                if ids[col[r]] in self.tracks.keys():
                                    if dist[r, col[r]] < self.reid_thresh:
                                        self.tracks[ids[col[r]]].append(tracks[r])
                                    else:
                                        self.tracks[id].append(tracks[r])
                                        id += 1
                                elif ids[col[r]] in self.inactive_tracks.keys():
                                    if dist[r, col[r]] < self.reid_thresh:
                                        self.tracks[ids[col[r]]] = self.inactive_tracls[ids[col[r]]]
                                        del self.inactive_tracks[ids[col[r]]]
                                    else:
                                        self.tracks[id].append(tracks[r])
                                        id += 1
                                elif r < num_detects:
                                    self.tracks[id].append(tracks[r])
                                    id += 1
                            else:
                                # padded zeros
                                pass
            
            results = self.make_results()
            
            results = interpolate(results)

            seq.write_results(results, self.config['tracker']['output_dir'])

            if seq.no_gt:
                print(f"No GT data for evaluation available.")
            else:
                mot_accums.append(get_mot_accum(results, seq))
        if mot_accums:
            print(f"Evaluation:")
            evaluate_mot_accums(mot_accums,
                            [str(s) for s in self.dataset if not s.no_gt],
                            generate_overall=True)
            

    def make_results(self):
        results = defaultdict(dict)
        for id, ts in self.tracks.items():
            for t in ts:
                results[id][t['im_index']] = np.concatenate([t['bbox'].numpy(), np.array([-1])])

        return results
