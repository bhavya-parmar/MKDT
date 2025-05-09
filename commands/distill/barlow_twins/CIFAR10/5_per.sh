#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python distill.py \
  --dataset CIFAR10 \
  --model ConvNet \
  --iters 5000 \
  --train_labels_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt \
  --expert_epochs 2 \
  --lr_img 10000 \
  --syn_steps 40 \
  --image_init_idx_path /home/jennyni/MKDT/init/cifar10/CIFAR10_barlow_twins_5_high_loss_indices.pkl \
  --max_start_epoch 5 \
  --expert_dir /home/jennyni/MKDT/buffers_barlow_twins/CIFAR10/ConvNet