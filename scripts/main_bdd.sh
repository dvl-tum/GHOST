#!/bin/sh
python tools/main_track.py \
    --det_conf 0.35 \
    --act 0.9 \
    --inact 0.8 \
    --inact_patience 50 \
    --combi sum_0.4 \
    --on_the_fly 1 \
    --store_feats 0 \
    --det_file byte_dets.txt \
    --config_path config/config_tracker_bdd.yaml\
     --split bdd100k_val \
     --only_pedestrian 0 \
     --new_track_conf 0.45 \
     --len_thresh 0 \
     --remove_unconfirmed 0 \
     --last_n_frames 10
