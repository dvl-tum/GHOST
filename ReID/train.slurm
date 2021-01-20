#!/bin/bash
#SBATCH --job-name=GL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=128G
#SBATCH --time=120:00:00

source activate gnn_reid
python train.py
