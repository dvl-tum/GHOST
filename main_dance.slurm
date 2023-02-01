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
python main_track.py --thresh 0.6 --act 0.8 --inact 0.9 --inact_thresh 50 --ioa_threshold sum_0.4 --do_inact 0 --eval_bb 0 --store_feats 0 --store_dist 0 --det_file byte_dets.txt --config_path config/config_tracker_danceV.yaml --new_track_conf 0.6 --len_thresh 0 --unconfirmed 0 --last_n_frames 5




