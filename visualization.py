import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
import matplotlib.colors as mcolors
import random
import os.path as osp

# MOT17
# p = '../out'
# p = '../OtherTrackers/bounding_boxes'
# data_path = '/storage/slurm/seidensc/datasets/MOT/MOT17/train'

# MOT20
p = '/storage/slurm/seidensc'
data_path = '/storage/slurm/seidensc/datasets/MOT/MOT20/test'

colors = mcolors.CSS4_COLORS
color_list = list(colors.keys())
color_list = color_list + color_list
random.shuffle(color_list)

# MOT17
# trackers = ['CenterTrackPub']
# dets = ['FRCNN']

# MOT20
trackers = ['detections_mot20_byte_for_orcun']
dets = ['']

for det in dets:
    for tracker in trackers:
        id_to_col = dict()
        col = 0
        for det_file in os.listdir(os.path.join(p, tracker)):
            if "01" in det_file or "02" in det_file or "03" in det_file or "05" in det_file:
                continue
            if "08" in det_file or "06" in det_file or "07" in det_file or "check.py" in det_file:
                continue
            print(det_file)
            #quit()
            
            # MOT17
            # seq = det_file[:-4]

            # MOT20
            seq = det_file

            df = pd.read_csv(osp.join(p, tracker, det_file), names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis', '?'])
            df['bb_left'] -= 1 # Coordinates are 1 based
            df['bb_top'] -= 1
            df['bb_right'] = (df['bb_left'] + df['bb_width']).values #- 1
            df['bb_bot'] = (df['bb_top'] + df['bb_height']).values #- 1

            os.makedirs(os.path.join('visualizations', tracker, seq), exist_ok=True)

            # Video writer
            mat = os.listdir(os.path.join(data_path, seq, 'img1'))[0]
            mat = matplotlib.image.imread(os.path.join(data_path, seq, 'img1', mat))
            print(seq, mat.shape)

            # video = cv2.VideoWriter(seq + '.avi', cv2.VideoWriter_fourcc(*'avc1'), 1, (mat.shape[0],mat.shape[1]))
                        
            '''for my_id in dets['my_id'].unique():
                print("new id {}".format(my_id))
                dets_id = dets[dets['my_id']==my_id]'''
            print(df['frame'].unique())
            for frame in df['frame'].unique():
                if frame % 20 == 0:
                    print(frame)
                # error
                frame_df = df[df['frame'] == frame]
                frame = int(frame)
                frame_path = os.path.join(data_path, seq, 'img1', f"{frame:06d}.jpg")
                img = matplotlib.image.imread(frame_path)
                figure, ax = plt.subplots(1)
                figure.set_size_inches(mat.shape[1]/100, mat.shape[0]/100)
                for i, row in frame_df.iterrows():
                    if row['id'] not in id_to_col.keys():
                        id_to_col[row['id']] = color_list[col]
                        col += 1
                    rect = matplotlib.patches.Rectangle((row['bb_left'],row['bb_top']),row['bb_width'],row['bb_height'], edgecolor=id_to_col[row['id']], facecolor="none", linewidth=2)
                    ax.add_patch(rect)
                ax.imshow(img)
                plt.axis('off')
                plt.savefig(os.path.join('visualizations', tracker, seq, f"{frame:06d}.jpg"), bbox_inches='tight', pad_inches = 0, dpi=100)
                #canvas = FigureCanvas(figure)
                #canvas.draw()
                #mat = np.array(canvas.renderer._renderer)
                #mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                #cv2.imwrite(os.path.join(seq, f"{frame:06d}.jpg"), mat)
                # write frame to video
                #video.write(mat)
                
                plt.close()


        # close video writer
        #video.release()
        #cv2.destroyAllWindows()
        print(id_to_col[row])
