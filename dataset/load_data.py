import os
import json
from .extract_market import marketlike
from .extract_cuhk import cuhk03
from .extract_duke import  duke
import random

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


def load_data(root: str = None, mode: str='single', val=0, seed=0):
    image_dir = os.path.join(root, 'images')

    # check if json file already exists --> if not: generate image folders
    if not os.path.isfile(os.path.join(root, 'info.json')) or \
            not os.path.isfile(os.path.join(root, 'labels.json')) or \
            not os.path.isdir(os.path.join(root, 'images')):

        # names of zip files
        if os.path.basename(os.path.dirname(root)) == 'cuhk03':
            check_zip = os.path.join(os.path.dirname(os.path.dirname(root)),
                                     'cuhk03_release.zip')
        elif os.path.basename(os.path.dirname(root)) == 'cuhk03-np':
            check_zip = os.path.join(os.path.dirname(os.path.dirname(root)),
                                     'cuhk03-np.zip')
        elif os.path.basename(root) == 'Market-1501-v15.09.15':
            check_zip = os.path.join(os.path.dirname(root),
                                     'Market-1501-v15.09.15.zip')
        elif os.path.basename(root) == 'dukemtmc':
            check_zip = os.path.join(os.path.dirname(root),
                                     'dukemtmc.zip')

        # check if zip file or extracted directory exists
        if not os.path.isfile(check_zip) and not os.path.isdir(root):
            if os.path.basename(root) == 'cuhk03-np':
                path = 'https://drive.google.com/file/d/1pBCIAGSZ81pgvqjC-lUHtl0OYV1icgkz/view'
            elif os.path.basename(os.path.dirname(root)) == 'cuhk03':
                path = 'https://drive.google.com/uc?id=0BxJeH3p7Ln48djNVVVJtUXh6bXc&export=download'
            elif os.path.basename(root) == 'Market-1501-v15.09.15':
                path = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
            elif os.path.basename(root) == 'dukemtmc':
                path = 'https://drive.google.com/uc?id=1qJ-7-o7OhDYj1T071z9a8uwFMiPhXEQc&export=download'
            print('Please download dataset from ' + path)
            quit()

        # generate image folders for dataset and splits
        if os.path.basename(os.path.dirname(root)) == 'cuhk03' or \
                os.path.basename(root) == 'cuhk03':
            cuhk03(root=root, check_zip=check_zip)
        elif os.path.basename(root) == 'dukemtmc':
            duke(root=root, image_dir=image_dir, check_zip=check_zip)
        else:
            marketlike(root=root, image_dir=image_dir, check_zip=check_zip)

    # if both take detected and labeled for training
    if mode == 'both':
        root_lab = os.path.join(os.path.dirname(root), 'labeled')
        root_det = os.path.join(os.path.dirname(root), 'detected')
        # load image paths and labels for splits
        with open(os.path.join(root_lab, 'info.json'), 'r') as file:
            data_lab = json.load(file)

        with open(os.path.join(root_lab, 'labels.json'), 'r') as file:
            labels_lab = json.load(file)

        with open(os.path.join(root_det, 'info.json'), 'r') as file:
            data_det = json.load(file)

        with open(os.path.join(root_det, 'labels.json'), 'r') as file:
            labels_det = json.load(file)

        # make list if not, for cuhk03 classic split is list
        if type(data_lab) != list:
            data_lab, labels_lab, data_det, labels_det = [data_lab], [
                labels_lab], [data_det], [labels_det]

        data, labels = list(), list()
        for dl, ll, dd, ld in zip(data_lab, labels_lab, data_det, labels_det):
            assert len(set(ll['bounding_box_train']).difference(
                set(ld['bounding_box_train']))) == 0
            assert len(set(ll['query']).difference(
                set(ld['query']))) == 0
            d, l = dict(), dict()
            for t in ['bounding_box_train', 'bounding_box_test', 'query']:
                d[t] = {'labeled': dl[t], 'detected': dd[t]}
                l[t] = {'labeled': ll[t], 'detected': ld[t]}
            data.append(d)
            labels.append(l)
    elif mode == 'all':
        root_cuhk03 = os.path.join(os.path.dirname(root), 'cuhk03', 'detected')
        root_market = os.path.join(os.path.dirname(root), 'Market-1501-v15.09.15')
        # load image paths and labels for splits
        with open(os.path.join(root_cuhk03, 'info.json'), 'r') as file:
            data_cuhk03 = json.load(file)

        with open(os.path.join(root_cuhk03, 'labels.json'), 'r') as file:
            labels_cuhk03 = json.load(file)

        with open(os.path.join(root_market, 'info.json'), 'r') as file:
            data_market = json.load(file)

        with open(os.path.join(root_market, 'labels.json'), 'r') as file:
            labels_market = json.load(file)

        # make list if not, for cuhk03 classic split is list
        if type(data_market) == list:
            data_market, labels_market = data_market[0], labels_market[0]
        if type(data_cuhk03) == list:
            data_cuhk03, labels_cuhk03 = data_cuhk03[0], labels_cuhk03[0]

        labels, data = list(), list()
        l, d = dict(), dict()
        for t in ['query', 'bounding_box_test', 'bounding_box_train']:
            l[t] = {'market': labels_market[t], 'cuhk03': labels_cuhk03[t]}
            d[t] = {'market': data_market[t], 'cuhk03': data_cuhk03[t]}

        labels.append(l)
        data.append(t)

    else:
        # load image paths and labels for splits
        with open(os.path.join(root, 'info.json'), 'r') as file:
            data = json.load(file)

        with open(os.path.join(root, 'labels.json'), 'r') as file:
            labels = json.load(file)

        if val and os.path.basename(os.path.dirname(root)) == 'cuhk03':
            print(seed)
            for i in range(len(data)):
                random.seed(seed+i)
                ind = random.sample(
                    set(labels[i]['bounding_box_train']), 100)

                data[i]['bounding_box_train'] = [
                    data[i]['bounding_box_train'][j] for j in
                    range(len(data[i]['bounding_box_train'])) if
                    labels[i]['bounding_box_train'][j] not in ind]

                labels[i]['bounding_box_train'] = [
                    labels[i]['bounding_box_train'][j] for j in
                    range(len(labels[i]['bounding_box_train'])) if
                    labels[i]['bounding_box_train'][j] not in ind]

        elif os.path.basename(root) == 'Market-1501-v15.09.15':
            for key, value in labels.items():
                junk = [i for i, v in enumerate(value) if v == -1]
                labels[key] = [v for i, v in enumerate(value) if i not in junk]
                data[key] = [v for i, v in enumerate(data[key]) if i not in junk]

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
    '''
    lab, data = load_data(root='../../../datasets/cuhk03-np/detected', both=0)
    print(len(lab))
    lab, data = load_data(root='../../../datasets/cuhk03/labeled', both=1)
    print(len(lab))

    lab, data = load_data(root='../../../datasets/cuhk03-np/detected')
    print(len(lab))
    lab, data = load_data(root='../../../datasets/cuhk03-np/labeled')
    print(len(lab))
    lab, data = load_data(root='../../../datasets/Market-1501-v15.09.15')
    print(len(lab))'''

    lab, data = load_data(root='../../../datasets/cuhk03/detected', val=1)
    print(len(set(lab[0]['bounding_box_train'])))
    quit()
    print(len(lab))
    lab, data = load_data(root='../../../datasets/cuhk03/labeled')
    print(len(lab))
