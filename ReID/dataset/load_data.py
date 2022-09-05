import os
import json
from .extract_market import marketlike

'''
Usage: 
bounding_box_train: used for training
bounding_box_test: used as gallery for testing
query: used as query images for testing
'''


def load_data(root: str = None, add_distractors=True):
    image_dir = os.path.join(root, 'images')

    # check if json file already exists --> if not: generate image folders
    if (not os.path.isfile(os.path.join(root, 'info.json')) or \
            not os.path.isfile(os.path.join(root, 'labels.json')) or \
            not os.path.isdir(os.path.join(root, 'images'))):

        # names of zip files
        if os.path.basename(root) == 'Market-1501-v15.09.15':
            check_zip = os.path.join(os.path.dirname(root),
                                     'Market-1501-v15.09.15.zip')
            # check if zip file or extracted directory exists
            if not os.path.isfile(check_zip) and not os.path.isdir(root):
                if os.path.basename(root) == 'Market-1501-v15.09.15':
                    path = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'

                    assert False, f'Please download dataset from {path}'
        else:
            assert False, f'no dataset for given root: {root}'

        # generate image folders for dataset and splits
        marketlike(root=root, image_dir=image_dir, check_zip=check_zip)

    # load image paths and labels for splits
    with open(os.path.join(root, 'info.json'), 'r') as file:
        data = json.load(file)

    with open(os.path.join(root, 'labels.json'), 'r') as file:
        labels = json.load(file)

    for key, value in labels.items():
        # -1 == junk, -2 == distractors
        if not add_distractors:
            labels[key] = [v for i, v in enumerate(value) if value[i] != -1 and value[i] != -2]
            data[key] = [v for i, v in enumerate(data[key]) if value[i] != -1 and value[i] != -2]
        else:
            labels[key] = [v for i, v in enumerate(value) if value[i] != -1]
            data[key] = [v for i, v in enumerate(data[key]) if value[i] != -1]

    # make list if not, for cuhk03 classic split is list
    if type(data) != list:
        data, labels = [data], [labels]

    # check if same number of identities in splits
    for split, split_paths in zip(labels, data):
        for t in ['bounding_box_train', 'bounding_box_test', 'query']:
            assert len(split_paths[t]) == len(split[t])

    return labels, data
