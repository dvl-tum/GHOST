import os
from collections import defaultdict
_SPLITS = defaultdict(dict)

#################
# MOT17
#################
dets = ('DPM', 'FRCNN', 'SDP')
train_seq_nums=  (2, 4, 5, 9, 10, 11, 13)
test_seq_nums=  (1, 3, 6, 7, 8, 12, 14)

# MOT17 train and test  sequences:
_SPLITS['mot17_test']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in test_seq_nums], 'dir': 'test'}
_SPLITS['mot17_train']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['debug_mot17_train']['test'] = {'seq': ['MOT17-05'], 'dir': 'train'}

# Cross Validation splits
_SPLITS['mot17_split_1']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 5, 9, 10, 13)], 'dir': 'train'}
_SPLITS['mot17_split_1']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (4, 11)], 'dir': 'train'}

_SPLITS['mot17_split_2']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 4, 11, 10, 13)], 'dir': 'train'}
_SPLITS['mot17_split_2']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (5, 9)], 'dir': 'train'}

_SPLITS['mot17_split_3']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (4, 5, 9, 11)], 'dir': 'train'}
_SPLITS['mot17_split_3']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 10, 13)], 'dir': 'train'}

# Remove IDs later
_SPLITS['50-50-1']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['50-50-2']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}


#################
# MOT20
#################

train_seq_nums=  (1, 2, 3, 5)
test_seq_nums = (4, 6, 7, 8)
_SPLITS['debug_mot20_train']['test'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in [5]], 'dir': 'train'}
_SPLITS['mot20_train']['test'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['mot20_test']['test'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in test_seq_nums], 'dir': 'test'}


#################
# BDD100k
#################

_SPLITS['bdd100k_val']['test'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/BDD100/bdd100k/images/track/val'), 'dir': 'val'}
_SPLITS['bdd100k_test']['test'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/BDD100/bdd100k/images/track/test'), 'dir': 'test'}
_SPLITS['debug_bdd100k_val']['test'] = {'seq': ['b23c9e00-b425de1b'], 'dir': 'val'}


#################
# DanceTrack
#################

_SPLITS['dance_test']['test'] = {'seq': os.listdir('/storage/user/seidensc/datasets/DanceTrack/test'), 'dir': 'test'}
_SPLITS['dance_val']['test'] = {'seq': os.listdir('/storage/user/seidensc/datasets/DanceTrack/val'), 'dir': 'val'}
_SPLITS['debug_dance_val']['test'] = {'seq':['dancetrack0004'], 'dir': 'val'}
