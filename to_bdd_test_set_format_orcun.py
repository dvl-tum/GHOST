import os
import numpy as np
import pandas as pd
import json

col_names=[
        'frame',
        'id',
        'bb_left',
        'bb_top',
        'bb_width',
        'bb_height',
        'conf',
        '?',
        'label',
        'vis']

p = 'out/dets_bdd_byte_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.9lenthresh:0newtrackthresh:0.45unconfirmed:0.0lastnframes:10MM:1sum_0.40.30.30.3InactPat:50ConfThresh:0.35_orig'
p3 = '/storage/slurm/seidensc/datasets/BDD100/bdd100k/images/track/val'

count = 0
for seq in os.listdir(p):
    if 'json' in seq:
        continue
    if seq in os.listdir(p3):
        count += 1
        print(count)
        df = pd.read_csv(os.path.join(p, seq), names=col_names, index_col=False)

        df['label'] = df['label'].values + np.ones(df.shape[0])

        final_out = df.sort_values(by=['frame', 'id'])
        sequence_name = seq
        output_file_path = 'bdd_final_val/' + seq + '.json'
        BDD_NAME_MAPPING = {
            1: "pedestrian",
            2: "rider",
            3: "car",
            4: "truck", 
            5: "bus",
            6: "train",
            7: "motorcycle",
            8: "bicycle"
        }

        det_list = list()
        # Find the max frame
        df = df.reset_index()
        max_frame = int(sorted(os.listdir(p3 + '/' + seq))[-1][:-4][-4:])
        for frame in range(1, max_frame+1):
            frame_dict = dict()
            frame_df = final_out[final_out['frame'] == frame]
            frame_dict['name'] = sequence_name + '/' + sequence_name + "-" + f"{frame:07d}.jpg"
            frame_dict['index'] = int(frame - 1)
            labels_list = list()
            for idx, row in frame_df.iterrows():
                labels_dict = dict()
                labels_dict['id'] = row['id']
                labels_dict['score'] = row['conf']
                labels_dict['category'] = BDD_NAME_MAPPING[int(row['label'])]
                labels_dict['box2d'] = {
                    'x1': row['bb_left'],
                    'x2': row['bb_left'] + row['bb_width'],
                    'y1': row['bb_top'],
                    'y2': row['bb_top'] + row['bb_height']
                }
                labels_list.append(labels_dict)

                # frame_path = '/storage/slurm/cetintas/BDD/val/b1c9c847-3bda4659' + f'img1/{int(frame):07}.jpg'
                # img_path = '/usr/wiss/cetintas/img/'
                # img = mpimg.imread(frame_path)
                # img = img[int(labels_dict['box2d']['y1']):int(labels_dict['box2d']['y2']), int(labels_dict['box2d']['x1']):int(labels_dict['box2d']['x2'])]
                # imgplot = plt.imshow(img)
                # identity = row['ped_id']
                # plt.savefig(img_path + f'frame{frame}'+f'id{identity}.jpg')


            frame_dict['labels'] = labels_list
            det_list.append(frame_dict)

        with open(output_file_path, 'w') as f:
            json.dump(det_list, f)