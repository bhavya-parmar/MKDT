#!/bin/bash

num_runs=5

script_params="\
    --dataset=Tiny \
    --model=ConvNetD4 \
    --num_experts=5 \
    --buffer_path="./buffers_barlow_twins" \
    --train_labels_path="/home/jennyni/MKDT/target_rep_krrst_original_test/Tiny_resnet18_target_rep_train.pt""

for device in {4..7}; do
    for ((i=0; i<num_runs; i++)); do
        tmux new-session -d -s "tiny_barlow_twins_collect_trajectory_${device}_$i"
        tmux send-keys "CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
    done
done
