
# ReID netowrk

This part of the repository contains code to train a ReID network on Market-1501 dataset as well as on MOT17 files.

## Get datasets

Market dataset:
Download Market-1501 dataset from [here](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) and put zip file into datasets.

MOT17 ReID dataset:
Download dataset from [here](https://vision.in.tum.de/webshare/u/seidensc/MOT17_ReID.zip) and put zip file into datasets.

## Train
To train on Market-1501 or MOT17 ReID dataset run:
```
tools/train.py
```

or 
```
tools/train_MOT.py
```

The latter will run on three different splits of MOT17 Sequences on default:

* split_1
    * train: (2, 5, 9, 10, 13)
    * test: (4, 11)
* split_2
    * train: (2, 4, 11, 10, 13)
    * test: (5, 9)
* split_3
    * train: (4, 5, 9, 11)
    * test': (2, 10, 13)

If you want to run a 50-50 split, i.e., splitting each of the sequences along the time dimension change config.dataset.split to 50-50. Then it train on the following splits:

* 50-50-1
    * train: 50-50-1
    * test: 50-50-2
* 50-50-2
    * train: 50-50-2
    * test: 50-50-1

The trained networks will be stored to ```checkpoints/```.

## Trained netoworks
The trained networks on Market dataset can be downloaded from [here](https://vision.in.tum.de/webshare/u/seidensc/ReIDModels.zip). Please extract them into ```trained_models/```.