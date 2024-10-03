#!/bin/bash

cleanup() {
    echo "Terminating all running processes..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT

RESULT_DIR1="/home/sjoshi/mtt-distillation/logged_files/CIFAR100/2024-01-10_13:30:36/"
RESULT_DIR2="/home/sjoshi/krrst_orig/results/2024-01-0814:47:42.550038_CIFAR100_high_loss_5000_new_arch/"
RESULT_DIR3="/home/sjoshi/krrst_orig/results/2024-01-0814:46:15.950178_CIFAR100_random_5000_new_arch/"


# best us
CUDA_VISIBLE_DEVICES=0 python eval.py --result_dir "${RESULT_DIR1}" --pre_lr 0.174 > ft_output_1_full.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python eval.py --result_dir "${RESULT_DIR1}" --pre_lr 0.174 --test_num_subset 100 > ft_output_1_100.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python eval.py --result_dir "${RESULT_DIR1}" --pre_lr 0.174 --test_num_subset 1000 > ft_output_1_1000.txt 2>&1 &


# high loss krrst
CUDA_VISIBLE_DEVICES=1 python eval.py --result_dir "${RESULT_DIR2}" > ft_output_2_full.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval.py --result_dir "${RESULT_DIR2}" --test_num_subset 100 > ft_output_2_100.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval.py --result_dir "${RESULT_DIR2}" --test_num_subset 1000 > ft_output_2_1000.txt 2>&1 &


# regular krrst
CUDA_VISIBLE_DEVICES=2 python eval.py --result_dir "${RESULT_DIR3}" > ft_output_3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval.py --result_dir "${RESULT_DIR3}" --test_num_subset 100 > ft_output_3_100.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval.py --result_dir "${RESULT_DIR3}" --test_num_subset 1000 > ft_output_3_1000.txt 2>&1 &


wait
echo "Completed"