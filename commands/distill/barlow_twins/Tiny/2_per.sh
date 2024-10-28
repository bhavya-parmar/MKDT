#!/bin/bash

CUDA_VISIBLE_DEVICES=5,6,7 python distill.py \
  --dataset Tiny \
  --model ConvNetD4 \
  --iters 10000 \
  --train_labels_path /home/jennyni/MKDT/target_rep_krrst_original_test/Tiny_resnet18_target_rep_train.pt \
  --expert_epochs 2 \
  --lr_img 100000 \
  --syn_steps 10 \
  --image_init_idx_path /home/jennyni/MKDT/init/tiny_imagenet/Tiny_barlow_twins_2_high_loss_indices.pkl \
  --max_start_epoch 2 \
  --expert_dir /home/jennyni/MKDT/buffers_barlow_twins/Tiny/ConvNetD4