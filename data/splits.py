import os
from collections import defaultdict
_SPLITS = defaultdict(dict)

#################
# MOT17
#################
dets = ('DPM', 'FRCNN', 'SDP')

# Train sequences:
train_seq_nums=  (2, 4, 5, 9, 10, 11, 13)
_SPLITS['mot17_train']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['mot17_train_test']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
# _SPLITS['mot17_train_test']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
# _SPLITS['mot17_train_test']['val'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}


# Cross Validation splits
_SPLITS['mot17_split_1']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 5, 9, 10, 13)], 'dir': 'train'}
_SPLITS['mot17_split_1']['val'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (4, 11)], 'dir': 'train'}
_SPLITS['mot17_split_1']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (4, 11)], 'dir': 'train'}

#_SPLITS['mot17_split_1'] = {'train_gt': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 5, 9, 10, 13)]}
#_SPLITS['mot17_split_1'] = {'val': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 11) for det in dets]}

_SPLITS['mot17_split_2']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 4, 11, 10, 13)], 'dir': 'train'}
_SPLITS['mot17_split_2']['val'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (5, 9)], 'dir': 'train'}
_SPLITS['mot17_split_2']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (5, 9)], 'dir': 'train'}

_SPLITS['mot17_split_3']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (4, 5, 9, 11)], 'dir': 'train'}
_SPLITS['mot17_split_3']['val'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 10, 13)], 'dir': 'train'}
_SPLITS['mot17_split_3']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 10, 13)], 'dir': 'train'}

# Cross Validation splits withing sequence
_SPLITS['mot17_split_along']['train'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 4, 5, 9, 10, 11, 13)], 'dir': 'train'}
_SPLITS['mot17_split_along']['val'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in (2, 4, 5, 9, 10, 11, 13)], 'dir': 'train'}

#_SPLITS['debug_train']['train'] = {'seq': ['MOT17-02'], 'dir': 'train'}
_SPLITS['debug_train']['train'] = {'seq': ['MOT17-04'], 'dir': 'train'}
_SPLITS['debug_train']['val'] = {'seq': ['MOT17-04'], 'dir': 'train'}
#_SPLITS['debug_train']['test'] = {'seq': ['MOT17-04'], 'dir': 'train'}
_SPLITS['debug_train']['test'] = {'seq': ['MOT17-13'], 'dir': 'train'}

# Remove IDs later
_SPLITS['50-50-1']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['50-50-2']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}

# Test sequences
test_seq_nums=  (1, 3, 6, 7, 8, 12, 14)
_SPLITS['mot17_test']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in test_seq_nums], 'dir': 'test'}


#mot20 split
train_seq_nums=  (1, 2, 3, 5)
debug = [1]
test_seq_nums = (4, 6, 7, 8)
_SPLITS['mot20_train']['train'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['mot20_train_debug']['test'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in debug], 'dir': 'train'}
_SPLITS['mot20_train_test']['test'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}
_SPLITS['mot20_test']['test'] = {'seq': [f'MOT20-{seq_num:02}' for seq_num in test_seq_nums], 'dir': 'test'}


#bdd100k
_SPLITS['bdd100k']['test'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/BDD100/bdd100k/images/track/val'), 'dir': 'val'}
_SPLITS['bdd100k_debug']['test'] = {'seq': ['b23c9e00-b425de1b'], 'dir': 'val'}


#DANCE
_SPLITS['dance']['train'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/DanceTrack/train'), 'dir': 'train'}
_SPLITS['dance']['val'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/DanceTrack/val'), 'dir': 'val'}
_SPLITS['dance']['test'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/DanceTrack/test'), 'dir': 'test'}

_SPLITS['dance_val']['test'] = {'seq': os.listdir('/storage/slurm/seidensc/datasets/DanceTrack/val'), 'dir': 'val'}
_SPLITS['dance_val_debug']['test'] = {'seq':['dancetrack0004'], 'dir': 'val'}
