import os.path as osp
import os
from zipfile import ZipFile

from collections import defaultdict
from scipy.io import loadmat
import json


def marketlike(root: str = None, image_dir: str = None, check_zip: str = None):

    if not osp.isdir(root):
        print("Extracting zip file...")
        with ZipFile(check_zip) as z:
            z.extractall(os.path.dirname(check_zip))

    print("Start processing data...")
    files = defaultdict(list)
    for data in ['bounding_box_train', 'bounding_box_test', 'query']:
        directory = os.path.join(root, data)
        for img in os.listdir(directory):
            try:
                pid = int(img.split('_')[0])
                cam = int(img.split('_')[1][1:2])
                number = img.split('_')[2:]
            except:
                if img.split('.')[-1] == 'db':
                    os.remove(os.path.join(directory, img))
                    continue
                else:
                    print('Can not process file {}'.format(img))
                    quit()

            # make identity directory in image directory
            dir_name = '{:05d}'.format(pid)
            os.makedirs(os.path.join(image_dir, dir_name), exist_ok=True)

            # move image to image directory
            img_new = '_'.join([dir_name, str(cam)] + number)
            os.rename(os.path.join(directory, img),
                      os.path.join(image_dir, dir_name, img_new))

            # add new image name to train images
            files[data].append(img_new)
        assert len(os.listdir(os.path.join(root, data))) == 0
        os.rmdir(os.path.join(root, data))

    assert len(files['query']) <= len(files['bounding_box_test'])
    assert set(files['bounding_box_train']).isdisjoint(set(files['query']))
    assert set(files['bounding_box_train']).isdisjoint(set(files['bounding_box_test']))

    # store paths to files in json file
    with open(os.path.join(root, 'info.json'), 'w') as file:
        json.dump(files, file)

    labels = defaultdict(list)
    for type in ['bounding_box_train', 'bounding_box_test', 'query']:
        for img in files[type]:
            labels[type].append(int(img.split('_')[0]))
        assert len(files[type]) == len(labels[type])

    # store paths to files in json file
    with open(os.path.join(root, 'labels.json'), 'w') as file:
        json.dump(labels, file)
