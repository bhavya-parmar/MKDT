#!/bin/bash

num_runs=5

script_params="\
    --dataset=CIFAR100 \
    --num_experts=5 \
    --buffer_path="./buffers_barlow_twins" \
    --train_labels_path="/home/jennyni/MKDT/target_rep_krrst_original_test/CIFAR100_resnet18_target_rep_train.pt""

for device in {0..3}; do
    for ((i=0; i<num_runs; i++)); do
        tmux new-session -d -s "c100_barlow_twins_collect_trajectory_${device}_$i"
        tmux send-keys "CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
    done
done
