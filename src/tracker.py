import numpy as np
from collections import defaultdict


class Tracker():
    def __init__(self, tracker_cfg):
        self.reid_thresh = tracker_cfg['reid_thresh']

        self.tracks = defaultdict(list)
        self.inactive_tracks = defaultdict(list)

    def track(self, seq):
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

