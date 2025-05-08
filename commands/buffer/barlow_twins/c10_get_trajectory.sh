#!/bin/bash

num_runs=5
current_datetime=$(date +"%Y%m%d_%H%M%S")

script_params="\
    --dataset=CIFAR10 \
    --num_experts=5 \
    --buffer_path="./buffers_barlow_twins" \
    --train_labels_path="/home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt""

for device in {0..3}; do
    for ((i=0; i<num_runs; i++)); do
        session_name="${current_datetime}_c10_new_barlow_twins_collect_trajectory_${device}_$i"
        tmux new-session -d -s "$session_name"
        tmux send-keys "CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
    done
done
