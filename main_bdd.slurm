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
python main_track.py --thresh 0.5 --act 0.8 --inact 0.7 --det_file qdtrack_dets.txt --config_path config/config_tracker_bdd.yaml
