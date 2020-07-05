import os.path as osp
import os
from zipfile import ZipFile

from collections import defaultdict
from scipy.io import loadmat
import json

url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
md5 = '65005ab7d12ec1c44de4eeafe813e68a'


def duke(root: str = None, image_dir: str = None, check_zip: str = None):

    if not osp.isdir(root):
        print("Extracting zip file")
        with ZipFile(check_zip) as z:
            z.extractall(os.path.dirname(check_zip))

    with open(os.path.join(root, 'splits.json'), 'r') as f:
        data = json.load(f)[0]

    files_dict = defaultdict(list)
    for img in os.listdir(image_dir):
        pid = int(img.split('_')[0])
        if not os.path.isdir(os.path.join(image_dir, '{:05d}'.format(pid))):
            os.makedirs(os.path.join(image_dir, '{:05d}'.format(pid)))
        os.rename(os.path.join(image_dir, img),
                  os.path.join(image_dir, '{:05d}'.format(pid), img))
        files_dict[pid].append(img)
    files = defaultdict(list)
    labels = defaultdict(list)
    for k, m in zip(['query', 'trainval', 'gallery'], ['query', 'bounding_box_train', 'bounding_box_test']):
        for pid in data[k]:
            files[m].append(files_dict[pid])
            labels[m].append([pid]*len(files_dict[pid]))
        files[m] = [f for p in files[m] for f in p]
        labels[m] = [l for p in labels[m] for l in p]
        assert len(files[m]) == len(labels[m])
        print(len(data[k]))
    assert len(files['query']) <= len(files['bounding_box_test'])
    assert set(files['bounding_box_train']).isdisjoint(set(files['query']))
    assert set(files['bounding_box_train']).isdisjoint(set(files['bounding_box_test']))

    # store paths to files in json file
    with open(os.path.join(root, 'info.json'), 'w') as file:
        json.dump(files, file)

    # store paths to files in json file
    with open(os.path.join(root, 'labels.json'), 'w') as file:
        json.dump(labels, file)
