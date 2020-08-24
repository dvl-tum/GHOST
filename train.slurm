#!/bin/bash
#SBATCH --job-name=GL
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=3
#SBATCH --exclude=node11,node12,node9,node7,node1
#SBATCH --mem=32G
#SBATCH --time=200:00:00

source activate gnn_reid
python train.py
