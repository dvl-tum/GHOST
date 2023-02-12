#!/bin/sh
python tools/main_track.py \
        --det_file tracktor_prepr_det.txt \
        --splits mot20_train \
        --config_path config/config_tracker_20.yaml \
        --det_conf -10 \
        --act 0.7 \
        --inact 0.8 \
        --inact_patience 50 \
        --combi sum_0.5 \
        --store_feats 0 \
        --new_track_conf -10 \
        --len_thresh 0 \
        --remove_unconfirmed 0 \
        --last_n_frames 60
