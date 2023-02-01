import os
import pandas as pd


p1 = 'out/with_our_reid'
p1 = 'out/byte_original'
p1 = 'out/with_our_reid_thresh'
p2 = '/storage/slurm/seidensc/datasets/MOT/MOT17/train' #MOT17-02-FRCNN/img1'
p1 = 'track_results'
p2 = '/storage/user/seidensc/datasets/DanceTrack/val'


for seq in os.listdir(p1):
    if seq[:-4] not in os.listdir(p2):
        continue
    num_frames = max([int(f[:-4]) for f in os.listdir(os.path.join(p2, seq[:-4], 'img1')) if f[-3:] == 'jpg'])
    df = pd.read_csv(os.path.join(p1, seq), names=['frame', 'id', 't', 'l', 'h', 'w', 'conf', '?', '!', '.'])
    df['frame'] = df['frame'].values + (num_frames - df['frame'].values.max())
    df.to_csv(seq[:-4], index=False, header=False)
