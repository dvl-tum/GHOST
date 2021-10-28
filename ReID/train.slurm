#!/bin/bash
#SBATCH --job-name=ResNetBackbonesAllReID
#SBATCH --nodes=1
#SBATCH --partition=DEADLINE
#SBATCH --comment='For CVPR submission.'
#SBATCH --gpus=rtx_8000:1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=128G
#SBATCH --partition=DEADLINE
#SBATCH --comment='For CVPR submission.'
#SBATCH --time=40:00:00

source activate allreid_tv
python train.py
