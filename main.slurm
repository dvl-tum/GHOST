#!/bin/bash
#SBATCH --job-name=AllReIDTracker
#SBATCH --output=/storage/slurm/seidensc/AllReID/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/seidensc/AllReID/logs/slurm-%j.out
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1
#SBATCH --partition=DEADLINE 
#SBATCH --comment="CVPR DEADLINE"
#SBATCH --cpus-per-gpu=1
#SBATCH --nodelist=node[1,7]
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate allreid_tv


python tools/main_track.py \
    --config_path config/config_tracker.yaml \
    --splits mot17_train \
    --act 0.7 \
    --inact 0.7 \
    --do_inact 1 \
    --inact_patience 50 \
    --combi sum_0.6 \
    --eval_bb 0 \
    --store_feats 0 \
    --det_file byte_val.txt \
    --det_conf 0.5 \
    --new_track_conf 0.55 \
    --last_n_frames 90





