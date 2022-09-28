#!/bin/bash
#SBATCH --job-name=AllReIDTracker
#SBATCH --nodes=1
#SBATCH --output=/storage/slurm/seidensc/AllReID/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/seidensc/AllReID/logs/slurm-%j.out
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --nodelist=node[8]
#SBATCH --gres=gpu:1
#SBATCH --partition=DEADLINE
#SBATCH --comment='checking sth for orcun'
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate allreid_tv
python main_track.py --thresh -0.6 --act 0.85 --inact 0.77 --det_file byte_dets.txt --config_path config/config_tracker_dance.yaml
