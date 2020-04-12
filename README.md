
# About

This repository contains a PyTorch implementation of [`The Group Loss for Deep Metric Learning`] https://arxiv.org/abs/1912.00385.

The same parameters were used as described in the paper. 

# PyTorch version

We use torch 1.1 and torchvision 0.2. While the training and inference should be able to be done correctly with the newer versions of the libraries, be aware that at times the network trained with torch > 1.2 might diverge or reach lower results.

We also support half-precision training via Nvidia Apex. 

# Reproducing Results

As in the paper we support training in 3 datasets: CUB-200-2011, CARS 196 and Stanford Online Products. Simply provide the path to the dataset in train.py and declare what dataset you want to use for the training. Training on some other dataset should be straightforwars as long as you structure the dataset in the same way as those three datasets.

The majority of experiments are done in inception with batch normalization. We provide support for the entire family of resnet and densenets. Simply define the type of the network you want to use in train.py
