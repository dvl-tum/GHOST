import os.path as osp
import os
from zipfile import ZipFile

from collections import defaultdict
from scipy.io import loadmat
import json

url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
md5 = '65005ab7d12ec1c44de4eeafe813e68a'


def marketlike(root: str = None, image_dir: str = None, check_zip: str = None):

    if not osp.isdir(root):
        print("Extracting zip file")
        with ZipFile(check_zip) as z:
            z.extractall(os.path.dirname(check_zip))

    files = defaultdict(list)
    for data in ['bounding_box_train', 'bounding_box_test', 'query']:
        directory = os.path.join(root, data)
        for img in os.listdir(directory):
            try:
                pid = int(img.split('_')[0])
            except:
                if img.split('.')[-1] == 'db':
                    os.remove(os.path.join(directory, img))
                    continue
                else:
                    print('Can not process file {}'.format(img))
                    quit()
            dir_name = '{:05d}'.format(pid)
            if not os.path.isdir(os.path.join(image_dir, dir_name)):
                os.makedirs(os.path.join(image_dir, dir_name))
            img_list = img.split('_')
            cam = img_list[1][1:2]
            img_new = '_'.join([img_list[0], cam] + img_list[2:])
            os.rename(os.path.join(directory, img),
                      os.path.join(image_dir, dir_name, img_new))
            files[data].append(img_new)

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
            indices[img][type] = x[type + '_index'].tolist()

        files['indices'] = indices

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
