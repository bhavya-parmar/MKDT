#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python distill.py \
  --dataset CIFAR100 \
  --model ConvNet \
  --iters 5000 \
  --train_labels_path /home/jennyni/MKDT/target_rep_krrst_original_test/CIFAR100_resnet18_target_rep_train.pt \
  --expert_epochs 2 \
  --lr_img 10000 \
  --syn_steps 40 \
  --image_init_idx_path /home/jennyni/MKDT/init/cifar100/CIFAR100_barlow_twins_5_high_loss_indices.pkl \
  --max_start_epoch 5 \
  --expert_dir /home/jennyni/MKDT/buffers_barlow_twins/CIFAR100/ConvNet