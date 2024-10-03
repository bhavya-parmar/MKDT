
#!/bin/bash

cleanup() {
    echo "Terminating all running processes..."
    pkill -P $$
    exit
}

trap cleanup SIGINT

EXTRA_ARGS="$@"

DATE_TIME=$(date "+%Y-%m-%d_%H-%M-%S")


RESULT_DIR="experiment_results/$DATE_TIME/limited_data"


# Create the results directory if it doesn't exist
mkdir -p "$RESULT_DIR"

GPUS=(1 2 3 4 5 6 7)


### CIFAR 100 2%

(
# No Pre-Training 
python eval.py --pre_data_name cifar100 --distill_method no_pre  --pre_lr 0.1 --use_random --gpu_id ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_nopretrain_2per.txt"

# Random
python eval.py --pre_data_name cifar100 --distill_method random --subset_path /home/jennyni/ssl-mtt/init/cifar100/cifar100_1000random.pkl --pre_lr 0.1 --gpu_id ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_random_2per.txt" 

# High Loss Subset 
# python eval.py --pre_data_name cifar100 --distill_method high_loss  --subset_path /home/jennyni/ssl-mtt/init/cifar100/CIFAR100_2_high_loss_indices.pkl --pre_lr 0.1 --gpu_id ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_hl_2per.txt" 

# SAS Subset
# python eval.py --pre_data_name cifar100 --distill_method sas --subset_path /home/jennyni/ssl-mtt/sas_subset/cifar100-cl-core-idx_1000.pkl --pre_lr 0.1 --gpu_id ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_sas_2per.txt" 


) & 

(
# MKDT Random
# python eval.py --pre_data_name cifar100 --distill_method mkdt_random  --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-13_20:44:35None --distilled_steps 5000 --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_random_2per.txt" 

# MKDT High Loss
python eval.py --pre_data_name cifar100 --distill_method mkdt  --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-29_22:12:45None --distilled_steps 5000 --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_high_2per.txt" 

# KRRST
# python eval.py --pre_data_name cifar100 --distill_method krrst  --result_dir /home/jennyni/ssl-mtt/synthetic_data/CIFAR100/krr_st --pre_epoch 20 --train_dataset CIFAR100 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR100_resnet18_target_rep_train.pt --use_krrst --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_krrst_2per.txt" 

) & 

# (
# Full Data 
# python eval.py --pre_data_name cifar100 --distill_method full_data  --pre_lr 0.1 --gpu_id ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_full.txt" 

# No Pre-training Finetuning
# python eval.py --pre_data_name cifar100 --distill_method no_pre_ft  --pre_lr 0.1 --use_random --test_algorithm full_finetune --gpu_id ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_nopretrain_finetune_2per.txt" 


# MKDT High Loss Finetuning 
# python eval.py --pre_data_name cifar100 --distill_method mkdt_ft  --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-04_22:46:47None --distilled_steps 5000 --test_algorithm full_finetune --gpu_id ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_high_2per_finetune.txt" 

# ) &

# (
# ### CIFAR 100 5%
# # Random
# python eval.py --subset_path /home/jennyni/ssl-mtt/init/cifar100/cifar100_2500random.pkl --pre_lr 0.1 --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_random_5per.txt" 

# # MKDT Random
# python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-13_20:44:35None --distilled_steps 5000 --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_random_5per.txt" 

# # MKDT High Loss
# python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-04_22:47:08None --distilled_steps 5000 --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_high_5per.txt" 
# ) &

### CIFAR 10 2%

(
# Random
# python eval.py --pre_data_name cifar10 --distill_method random  --subset_path /home/jennyni/ssl-mtt/init/cifar10/cifar10_random100ipc.pkl --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_random_2per.txt" 

# MKDT High Loss
python eval.py --pre_data_name cifar10 --distill_method mkdt  --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR10/2024-05-29_22:13:04None --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --distilled_steps 5000 --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_mkdt_high_2per.txt" 

# KRR-ST (PENDING?)
# python eval.py --pre_data_name cifar10 --train_dataset cifar10 --distill_method krrst  --result_dir /home/jennyni/ssl-mtt/synthetic_data/CIFAR10/krr_st --pre_epoch 20 --train_dataset CIFAR100 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --use_krrst --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_krrst_2per.txt" 


# No Pre-Training 
# python eval.py --pre_data_name cifar10 --distill_method no_pre --pre_epoch 20 --use_random --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_nopretrain.txt" 

# SAS subset
# python eval.py --pre_data_name cifar10 --distill_method sas  --subset_path /home/jennyni/ssl-mtt/sas_subset/cifar10-cl-core-idx_1000.pkl --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_random_2per.txt" 


) 

# (
# Full Data 
# python eval.py --pre_data_name cifar10 --distill_method full_data --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --gpu_id ${GPUS[4]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_full.txt" 


# No Pre-training Finetuning
# python eval.py --pre_data_name cifar10 --distill_method no_pre_ft --pre_epoch 20 --use_random --test_algorithm full_finetune --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --gpu_id ${GPUS[4]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_nopretrain_finetune.txt" 

# MKDT High Loss Finetuning 
# python eval.py --pre_data_name cifar10 --distill_method mkdt_ft --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR10/2024-05-05_11:49:29None --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --distilled_steps 5000 --test_algorithm full_finetune --gpu_id ${GPUS[4]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_mkdt_high_2per_finetune.txt" 

# ### CIFAR 10 5%

# # Random
# python eval.py --subset_path /home/jennyni/ssl-mtt/init/cifar10/cifar10_total2500random.pkl --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --gpu_id ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_random_5per.txt" 

# # MKDT High Loss
# python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR10/2024-05-05_11:51:31None  --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR10_resnet18_target_rep_train.pt --distilled_steps 5000 --gpu_id ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_mkdt_high_5per.txt" 

# ) &


### Tiny 2%

(
# Random
# python eval.py --pre_data_name tiny --distill_method random --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --subset_path /home/jennyni/ssl-mtt/init/tiny_imagenet/tiny_random_2perc.pkl --gpu_id ${GPUS[5]} $EXTRA_ARGS > "$RESULT_DIR/tiny_random_2per.txt" 

# MKDT High Loss
python eval.py --pre_data_name tiny --distill_method mkdt --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-05-29_22:13:51None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --distilled_steps 5000 --gpu_id ${GPUS[5]} $EXTRA_ARGS > "$RESULT_DIR/tiny_mkdt_high_2per.txt" 

# KRRST
# python eval.py --pre_data_name tiny --distill_method krrst --result_dir /home/jennyni/ssl-mtt/synthetic_data/Tiny/krr_st --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --use_krrst --gpu_id ${GPUS[5]} $EXTRA_ARGS > "$RESULT_DIR/tiny_krrst_2per.txt" 

) &

# (
# No Pre-Training
# python eval.py --pre_data_name tiny --distill_method no_pre --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --use_random --gpu_id ${GPUS[6]} $EXTRA_ARGS > "$RESULT_DIR/tiny_nopretrain.txt" 

# Full Data 
# python eval.py --pre_data_name tiny --distill_method full_data --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --gpu_id ${GPUS[6]} $EXTRA_ARGS > "$RESULT_DIR/tiny_full.txt" 

# No Pre-training Finetuning
# python eval.py --pre_data_name tiny --distill_method no_pre_ft --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --use_random --test_algorithm full_finetune --gpu_id ${GPUS[6]} $EXTRA_ARGS > "$RESULT_DIR/tiny_nopretrain_finetune.txt" 

# MKDT High Loss Finetuning 
# python eval.py --pre_data_name tiny --distill_method mkdt_ft --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-05-04_22:48:05None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --distilled_steps 5000 --test_algorithm full_finetune --gpu_id ${GPUS[6]} $EXTRA_ARGS > "$RESULT_DIR/tiny_mkdt_high_2per_finetune.txt" 

# ### Tiny 5%

# # Random
# python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --subset_path /home/jennyni/ssl-mtt/init/tiny_imagenet/tiny_5000random.pkl --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/tiny_random_5per.txt" 

# # MKDT High Loss
# python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-05-04_22:55:27None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt --distilled_steps 5000 --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/tiny_mkdt_high_5per.txt" 

# ) 

### ImageNet

# Random

# MKDT High Loss

# KRRST


# (
# ### Supervised
# python eval.py --supervised_pretrain -s random_sup --gpu_id ${GPUS[4]} > "$RESULT_DIR/random_sup.txt" 

# python eval.py --supervised_pretrain -s kmeans --gpu_id ${GPUS[4]} > "$RESULT_DIR/kmeans.txt" 

# python eval.py --supervised_pretrain -s dsa --gpu_id ${GPUS[4]}  > "$RESULT_DIR/dsa.txt" 

# python eval.py --supervised_pretrain -s dm --gpu_id ${GPUS[4]}  > "$RESULT_DIR/dm.txt" 

# python eval.py --supervised_pretrain -s mtt --gpu_id ${GPUS[4]} > "$RESULT_DIR/mtt.txt" 

# python eval.py --supervised_pretrain -s kip --gpu_id ${GPUS[4]} > "$RESULT_DIR/kip.txt" 

# python eval.py --supervised_pretrain -s frepo --gpu_id ${GPUS[4]} > "$RESULT_DIR/frepo.txt" 

# )


wait
echo "Completed"


