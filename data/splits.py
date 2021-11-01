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
#_SPLITS['debug_train']['val'] = {'seq': ['MOT17-04'], 'dir': 'train'}
_SPLITS['debug_train']['test'] = {'seq': ['MOT17-02'], 'dir': 'train'}
#_SPLITS['debug_test']['test'] = {'seq': ['MOT17-13'], 'dir': 'train'}

# Remove IDs later
_SPLITS['50-50']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in train_seq_nums], 'dir': 'train'}


# Test sequences
test_seq_nums=  (1, 3, 6, 7, 8, 12, 14)
_SPLITS['mot17_test']['test'] = {'seq': [f'MOT17-{seq_num:02}' for seq_num in test_seq_nums], 'dir': 'test'}
