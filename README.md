# GHOST Tracker
This is the official repository of [Simple Cues Lead to a Strong Multi-Object Tracker](https://arxiv.org/abs/2206.04656).

## Git Repo
To set up this repository follow the following steps
```
git clone https://github.com/dvl-tum/GHOST.git
cd GHOST
git clone https://github.com/dvl-tum/TrackEvalForGHOST.git
```

## Environment
Download anaconcda and create conda evironment using env_from_history.yml file by:
```
conda env create -f env_from_history.yml
```
Then activate the environment using:
```
conda activate GHOST
```

##  Dataset Setup 
Download [MOT17](https://motchallenge.net/data/MOT17/), [MOT20](https://motchallenge.net/data/MOT20/), and [DanceTrack](https://drive.google.com/drive/folders/1-uxcNTi7dhuDNGC5MmzXyllLzmVbzXay) tracking datasets. For [BDD100k](https://bdd-data.berkeley.edu/) download MOT 2020 Labels and MOT 2020 images. Unzip all of them to ```datasets```. 

Finally also download our [detections](https://vision.in.tum.de/webshare/u/seidensc/GHOST/detections_GHOST.zip) used and also extract into ```dataset```. For MOT17 we also provide the bounding boxes from various trackers on the validation set, i.e., first half of all training sequences.


The final data structure should look like the following:
```
datasets/
    - bdd100k
        - images
            - track
                - train
                - val
                - test
        - labels
            - box_track_20
                - train
                - val
    - DanceTrack
        - train
        - val
        - test
    - MOT17
        - train
        - test
    - MOT20
        - train
        - test
    - detections_GHOST
        - bdd100k
            - train
            - val
            - test
        - DanceTrack
            - val
            - test
        - MOT17
            - train
            - test
        - MOT20
            - train
            - test
```

##  ReID Setup 
Download our pretrained ReID [weights](https://vision.in.tum.de/webshare/u/seidensc/GHOST/trained_reid_models.zip) and extract them into ```ReID/trained_models``` so that in the end the data structure looks like the the following:
```
ReID/
    * trained_models
        * market_models
            * resnet50_Market.pth
        * MOT_models
            * split1_resnet50_MOT17.pth
            * split2_resnet50_MOT17.pth
            * split3_resnet50_MOT17.pth
```

## Tracking
To run our tracker run on MOT17 private detections run:
```
bash scripts/main_17.sh
```
and to run in with public center track preprocessed detections run:
```
bash scripts/main_17_pub.sh
```
Similarly, you can find scripts for ```main_20.sh```, ```main_20_pub.sh``` using tracktor preprocessed detections, ```main_dance.sh```, and ```main_bdd.sh``` in the scripts directory.

You can define the following parameters directly in the bash file:

| Parameter         | description           |
|-----------        |------------           |
| --config_path     | Path to config file   |
| --det_conf        | Minimum detection confidence   |
| --act     | Matching threshold for active tracks   |
| --inact     | Matching threshold for inactive tracks   |
| --det_file     | Detections to be used (see dataset/detections_GHOST for names)   |
| --only_pedestrian     | If only pedestrian class should be used for evaluation   |
| --inact_patience     | Patience for inactive tracks to be used during tracking   |
| --combi    | How to combine motion and appearance distance (sum_0.3 means weighted sum with motion weight 0.3)  |
| --store_feats     | Store features for analysis  |
| --on_the_fly         | If using on the fly domain adaptation   |
| --do_inact         | If using proxy distance / proxy feature compuatation for inactive tracks |
| --splits         | Which split to use (see data/splits.py for different splits)|
| --len_thresh         | Minimum length of tracks (default set to 0) |
| --new_track_conf         | Confidence threshold for detection to start new track |
| --remove_unconfirmed         | If removing unconfirmed tracks (tracks that are only initialized and then no detection added in next frame, default is 0) |
| --last_n_frames         | Number of last frames used to compute velocity for motion model |


 For others, like data paths, please refer directly to the config files in ```config/```.


 ### Test submission
 For the test submissions please adapt the split in the configuration parameters to the corresponding splits (```data/splits.py```). 
 #### MOT17, MOT20, DanceTrack
 For submission please zip the files in the corresponding output directories and submit to the test servers of [MOT17, MOT20](https://motchallenge.net/instructions/), [DanceTrack](https://codalab.lisn.upsaclay.fr/competitions/5830).
 #### BDD
 If you want to submit to [BDD server](https://eval.ai/web/challenges/challenge-page/1836/overview), please utilize the corresponding experiment directory in the ```bdd_for_submission``` directory that is directly generated, zip the files directly (not the directory), and upload under Submit to the server.

 ### Using distance computations
 If you want to use different distance computations than the current proxy distance computation, you have to change the ```avg_act``` and ```avg_inact``` sections in the config files the following for other proxy distances:

| Do | num | proxy  |Description |
|----|----|-----|-------|
|1 | 1| 'each_sample' | Min of distances between features of new detection and features of all prior detections in track |
|1 | 2| 'each_sample' | Mean of distances between features of new detection and features of all prior detections in track |
|1 | 3| 'each_sample' | Max of distances between features of new detection and features of all prior detections in track |
|1 | 4| 'each_sample' | (Min + Max)/2 of distances between features of new detection and features of all prior detections in track |
|1 | 5| 'each_sample' | Median of distances between features of new detection and features of all prior detections in track |
|1 | x | 'first' | Uses the features of the first detection in the track for distance computation with the features of the new detection, x does not matter |
|1 | x | 'last' | Uses the features of the last detection in the track for distance computation with the features of the new detection, x does not matter |
|1 | x | 'mv_avg' | Uses the moving average of features in the track for distance computation with the features of the new detection, x is the update weight |
|1 | x | 'mean' | Uses the mean of features in the track for distance computation with the features of the new detection, x number of last detections to be used  |
|1 | x | 'median' | Uses the median of features in the track for distance computation with the features of the new detection, x number of last detections to be used  |
|1 | x | 'mode' | Uses the mode of features in the track for distance computation with the features of the new detection, x number of last detections to be used  |

If you set do to 0 it falls back to using the features of the last detection in a track.


## Histogram and Quantile analysis
If you want to run the histogram or quantile analysis as in the paper please first run an experiment on the detection set you want to use. The features will be stored to ```features```. Then run the 
```
python tools/investigate_features.py
```
the corresponding figures will be stored to ```histograms_features```.
