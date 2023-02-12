#!/bin/sh
python tools/main_track.py \
    --config_path config/config_tracker.yaml \
    --splits mot17_train \
    --act 0.7 \
    --inact 0.8 \
    --do_inact 1 \
    --inact_patience 50 \
    --combi sum_0.4 \
    --on_the_fly 1 \
    --store_feats 0 \
    --det_file center_track.txt \
    --det_conf -10 \
    --new_track_conf -10 \
    --last_n_frames 10
