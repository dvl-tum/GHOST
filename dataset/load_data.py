import os
import json
from .extract_market import marketlike
from .extract_cuhk import cuhk03


'''
Naming conventions:
In bbox "0001_c1s1_001051_00.jpg", "c1" is the first camera
(there are totally 6 cameras).

"s1" is sequence 1 of camera 1. Here, a sequence was defined automatically by
the camera. We suppose that the camera cannot store a whole video that is quite
large, so it splits the video into equally large sequences. Two sequences,
namely, "c1s1" and "c2s1" do not happen exactly at the same time. This is
mainly because the starting time of the 6 cameras are not exactly the same (it
takes time to turn on them). But, "c1s1" and "c2s1" are roughly at the same
time period.

"001051" is the 1051th frame in the sequence "c1s1". The frame rate is 25
frames per sec.

As with the last two digits, remember we use the DPM detector. Then, for
identity "0001", there may be multiple detected bounding boxes in the frame
"c1s1_001051". In other words, a pedestrian in the image may have several
bboxes by DPM. So, "00" means that this bounding box is the first one among
the several.

Usage: 
bounding_box_train: used for training
bounding_box_test: used as gallery for testing
query: used as query images for testing
'''


def load_data(root: str = None):
    image_dir = os.path.join(root, 'images')

    # check if json file already exists --> if not: generate image folders
    if not os.path.isfile(os.path.join(root, 'info.json')):

        # names of zip files
        if os.path.basename(os.path.dirname(root)) == 'cuhk03':
            check_zip = os.path.join(os.path.dirname(os.path.dirname(root)), 'cuhk03_release.zip')
        elif os.path.basename(os.path.dirname(root)) == 'cuhk03-np':
            check_zip = os.path.join(os.path.dirname(os.path.dirname(root)), 'cuhk03-np.zip')
        elif os.path.basename(root) == 'Market-1501-v15.09.15':
            check_zip = os.path.join(os.path.dirname(root), 'Market-1501-v15.09.15.zip')

        # check if zip file or extracted directory exists
        if not os.path.isfile(check_zip) and not os.path.isdir(root):
            if os.path.basename(root) == 'cuhk03-np':
                path = 'https://drive.google.com/file/d/1pBCIAGSZ81pgvqjC-lUHtl0OYV1icgkz/view'
            elif os.path.basename(os.path.dirname(root)) == 'cuhk03':
                path = 'https://drive.google.com/uc?id=0BxJeH3p7Ln48djNVVVJtUXh6bXc&export=download'
            elif os.path.basename(root) == 'Market-1501-v15.09.15':
                path = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
            print('Please download dataset from ' + path)
            quit()

        # generate image folders for dataset and splits
        if os.path.basename(os.path.dirname(root)) == 'cuhk03':
            cuhk03(root=root, check_zip=check_zip)
        else:
            marketlike(root=root, image_dir=image_dir, check_zip=check_zip)

    # load image paths and labels for splits
    with open(os.path.join(root, 'info.json'), 'r') as file:
        data = json.load(file)

    with open(os.path.join(root, 'labels.json'), 'r') as file:
        labels = json.load(file)

    # make list if not, for cuhk03 classic split is list
    if type(data) != list:
        data, labels = [data], [labels]

    # check if same number of identities in splits
    for split, split_paths in zip(labels, data):
        for t in ['bounding_box_train', 'bounding_box_test', 'query']:
            assert len(split_paths[t]) == len(split[t])

    return labels, data


if __name__ == '__main__':
    # test

    lab, data = load_data(root='../../../datasets/cuhk03-np/detected')
    print(len(lab))
    lab, data = load_data(root='../../../datasets/cuhk03-np/labeled')
    print(len(lab))
    lab, data = load_data(root='../../../datasets/Market-1501-v15.09.15')
    print(len(lab))
    lab, data = load_data(root='../../../datasets/cuhk03/detected')
    print(len(lab))
    lab, data = load_data(root='../../../datasets/cuhk03/labeled')
    print(len(lab))
