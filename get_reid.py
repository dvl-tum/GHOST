p = "slurm-356708.out" #'slurm-356440.out'
with open(p, 'r') as f:
    content = f.readlines()

content = [x.strip() for x in content] 

ranks = list()
mAP = list()
for l in content:
    if l[:4] == 'rank':
        l = l.split(' ')
        l = [float(l[1][:-1]), float(l[3][:-1]), float(l[5])]
        ranks.append(l)
    if l[:3] == 'mAP':
        l = float(l.split(' ')[-1])
        mAP.append(l)
print(ranks)
print(mAP)

import numpy as np
ranks = np.array(ranks)
ranks = np.mean(ranks, axis=0)

mAP = np.array(mAP)
mAP = np.mean(mAP)

print(ranks)
print(mAP)
