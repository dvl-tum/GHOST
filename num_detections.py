import os
import pandas as pd
import numpy as np


path = 'Mar_9_12:02:06_2021mean_60_10000_0.65_0.5'
path =  '../../datasets/MOT/MOT17Labels/train'
seqs = os.listdir(path)

# frame, id, left, top, width, height, x, y, z, k 
w = list()
h = list()
num_samps = list()
for s in seqs:
    print(os.path.join(path, s, 'det/tracktor_prepr_det.txt'))
    df = pd.read_csv(os.path.join(path, s, 'det/tracktor_prepr_det.txt'))
    for ind, row in df.iterrows():
        w.append(row[4])
        h.append(row[5])
    print(df.shape)
    num_samps.append(df.shape[0])

w = np.array(w)
h = np.array(h)

print(np.max(w), np.min(w), np.mean(w))
print(np.max(h), np.min(h), np.mean(h))
print(sum(num_samps))
