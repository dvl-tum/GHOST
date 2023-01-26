#!/bin/bash
#SBATCH --job-name=AllReIDTracker
#SBATCH --nodes=1
#SBATCH --output=/storage/slurm/seidensc/AllReID/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/seidensc/AllReID/logs/slurm-%j.out
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1
#SBATCH --partition=DEADLINE
#SBATCH --comment='CVPR Deadline'
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate allreid_tv
python main_track.py --thresh -0.6 --act 0.9 --inact 0.8 --do_inact 1 --eval_bb 0 --inact_thresh 50 --store_feats 0 --store_dist 0 --ioa_threshold sum_0.5 --det_file byte_dets.txt --config_path config/config_tracker_dance.yaml

