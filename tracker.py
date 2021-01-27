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
from tracking_wo_bnw.src.tracktor.utils import interpolate, get_mot_accum, evaluate_mot_accums


class Tracker():
    def __init__(self, config, device, timer):
        self.config = config
        self.device = device

        #load ReID net
        self.encoder, _ = ReID.net.load_net(
            self.config['reid_net']['trained_on']['name'],
            self.config['reid_net']['trained_on']['num_classes'], 'test',
            **self.config['reid_net']['encoder_params'])
        
        self.encoder = self.encoder.cuda(self.device)
        self.reid_thresh = self.config['tracker']['reid_thresh']

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)

        self.dataset = factory.Datasets(self.config['dataset']['name'])
        
        '''
        if self.config['dataset']['tracktor_pre']['use']:
            self.use_preprocessed()
        try:
            self.dataset = factory.Datasets(self.config['dataset'])
        except Exception as e:
            print(e)
            print("HELLO")
            self.undo_preprocessed()
            undone = 1
            quit()
        
        if undone == 0:
            self.undo_preprocessed()
        '''
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

    def build_crops(self, image, rois):
        res = []
        #trans = Compose([ToPILImage(), Resize((256,128)), ToTensor()])
        to_pil = transforms.ToPILImage()
        trans = make_transform_bot(is_train=False)
        for r in rois:
            print(r)
            x0 = int(r[0])
            y0 = int(r[1])
            x1 = int(r[2])
            y1 = int(r[3])
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            im = image[0,:,y0:y1,x0:x1]
            im = to_pil(im)
            im = trans(im)
            res.append(im)
        res = torch.stack(res, 0)
        res = res.cuda()
        return res

    def use_preprocessed(self):
        track_paths = self.config['dataset']['tracktor_pre']['path']
        dataset =  self.config['dataset']['name'].split('_')[0].upper()
        seq = self.config['dataset']['name'].split('_')[1]
        detector = self.config['dataset']['name'].split('_')[2]
        
        if dataset == 'MOT17':
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
            self.s_dir = osp.join(self.config['dataset']['path'], dataset, dir, s)
            self.d_dir = osp.join(self.config['dataset']['path'], dataset, dir, s_det)
            t_dir = osp.join(self.config['dataset']['path'], track_paths, dir, s_det)
            self.t_dir = osp.join(t_dir, 'det', 'tracktor_prepr_det.txt')

            shutil.move(osp.join(self.d_dir, 'det', 'det.txt'),
                        osp.join(self.d_dir, 'det', 'd_o.txt'))
            shutil.copyfile(self.t_dir, osp.join(self.d_dir, 'det', 'det.txt'))

    def undo_preprocessed(self):
        shutil.move(osp.join(self.d_dir, 'det', 'det.txt'), self.t_dir)
        shutil.move(osp.join(self.d_dir, 'det', 'd_o.txt'),
                    osp.join(self.d_dir, 'det', 'det.txt'))
            

    def make_results(self):
        results = defaultdict(dict)
        for id, ts in self.tracks.items():
            for t in ts:
                results[id][t['im_index']] = np.concatenate([t['bbox'].numpy(), np.array([-1])])

        return results
