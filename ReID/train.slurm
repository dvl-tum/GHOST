#!/bin/bash
#SBATCH --job-name=ResNetBackbonesAllReID
#SBATCH --nodes=1
#SBATCH --partition=DEADLINE
#SBATCH --comment='For CVPR submission.'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=128G
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --partition=DEADLINE
#SBATCH --comment='For CVPR submission.'
#SBATCH --time=40:00:00

source activate allreid_tv
python tools/train.py
