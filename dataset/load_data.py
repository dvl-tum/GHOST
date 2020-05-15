import os
from collections import defaultdict
import json
import h5py
from scipy.io import loadmat


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

    # convert downloaded data to pytorch image folder
    if not os.path.isfile(os.path.join(root, 'info.json')):
        if not os.path.isdir(
                os.path.join(root, 'bounding_box_test')) or not os.path.isdir(
                os.path.join(root, 'bounding_box_train')) or not os.path.isdir(
                os.path.join(root, 'query')):
            if os.path.basename(root) == 'cuhk03-np':
                path = 'https://drive.google.com/file/d/1pBCIAGSZ81pgvqjC-lUHtl0OYV1icgkz/view'
            else:
                path = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
            print('Please download dataset from ' + path)
            quit()
        files = defaultdict(list)

        # iterate over bbtrain, bbtest und query and move single images to
        # class directories
        for data in ['bounding_box_train', 'bounding_box_test', 'query']:
            directory = os.path.join(root, data)
            for img in os.listdir(directory):
                pid = int(img.split('_')[0])
                dir_name = '{:05d}'.format(pid)
                if not os.path.isdir(os.path.join(image_dir, dir_name)):
                    os.makedirs(os.path.join(image_dir, dir_name))
                os.rename(os.path.join(directory, img),
                          os.path.join(image_dir, dir_name, img))
                files[data].append(img)
            assert len(os.listdir(os.path.join(root, data))) == 0
            os.rmdir(os.path.join(root, data))

        # get junk and good indices
        if root.split('/')[-1].split('-')[0] == "Market":
            indices = defaultdict(dict)
            for gt in os.listdir(os.path.join(root, 'gt_query')):
                x = loadmat(
                    os.path.join(root, 'gt_query', gt))
                img = ('_').join(gt.split('_')[:-1]) + '.jpg'
                type = gt.split('_')[-1].split('.')[0]
                indices[img][type] = x[type + '_index']

        files['indices'] = indices

        # store paths to files in json file
        with open(os.path.join(root, 'info.json'), 'w') as file:
            json.dump(files, file)


    with open(os.path.join(root, 'info.json'), 'r') as file:
        data = json.load(file)

    return data


if __name__ == '__main__':
    load_data(root='../../../datasets/Market-1501-v15.09.15')
