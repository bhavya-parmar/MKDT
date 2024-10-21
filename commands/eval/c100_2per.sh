#!/bin/bash

cleanup() {
    echo "Terminating all running processes..."
    pkill -P $$
    exit
}

trap cleanup SIGINT

SUBSET_FRAC="default"

# Parse command-line arguments to extract --subset_frac value
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset_frac)
      SUBSET_FRAC="$2"
      EXTRA_ARGS+="--subset_frac $2 "
      shift 2
      ;;
    *)
      EXTRA_ARGS+="$1 "
      shift
      ;;
  esac
done

DATE_TIME=$(date "+%Y-%m-%d_%H-%M-%S")

RESULT_DIR="experiment_results/cifar100_2per_${SUBSET_FRAC}_labeled_data/$DATE_TIME/"

mkdir -p "$RESULT_DIR"

GPUS=(1 2 3 4 5 6 7)

### CIFAR 100 2%

# No Pre-Training 
python eval.py --pre_data_name cifar100 --distill_method no_pre  --pre_lr 0.1 --use_random --gpu_id ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_nopretrain_2per.txt" &

# Random
python eval.py --pre_data_name cifar100 --distill_method random --subset_path /home/jennyni/ssl-mtt/init/cifar100/cifar100_1000random.pkl --pre_lr 0.1 --gpu_id ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_random_2per.txt" &

# High Loss Subset 
python eval.py --pre_data_name cifar100 --distill_method high_loss  --subset_path /home/jennyni/ssl-mtt/init/cifar100/CIFAR100_2_high_loss_indices_first.pkl --pre_lr 0.1 --gpu_id ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_hl_2per.txt" &

# SAS Subset
python eval.py --pre_data_name cifar100 --distill_method sas --subset_path /home/jennyni/ssl-mtt/sas_subset/cifar100-cl-core-idx_1000.pkl --pre_lr 0.1 --gpu_id ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_sas_2per.txt" &

# MKDT Random
python eval.py --pre_data_name cifar100 --distill_method mkdt_random  --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-13_20:44:35None --distilled_steps 5000 --gpu_id ${GPUS[4]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_random_2per.txt" &

# MKDT High Loss
python eval.py --pre_data_name cifar100 --distill_method mkdt  --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-05-04_22:46:47None --distilled_steps 5000 --gpu_id ${GPUS[5]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_mkdt_high_2per.txt" &

# KRRST
python eval.py --pre_data_name cifar100 --distill_method krrst  --result_dir /home/jennyni/ssl-mtt/synthetic_data/CIFAR100/krr_st --pre_epoch 20 --train_dataset CIFAR100 --label_path /home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR100_resnet18_target_rep_train.pt --use_krrst --gpu_id ${GPUS[6]} $EXTRA_ARGS > "$RESULT_DIR/cifar100_krrst_2per.txt" &

wait
