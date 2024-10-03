#!/bin/bash

cleanup() {
    echo "Terminating all running processes..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT


# Command 1
python eval.py --subset_path /home/jennyni/ssl-mtt/init/cifar100/cifar100_500random.pkl --pre_lr 0.1 --gpu_id 4 --use_random > c100_output_cifar100_500random.txt & 

# Command 2
python eval.py --subset_path /home/sjoshi/mtt-distillation/init/cifar100/high_loss_1pct.pkl --pre_lr 0.1 --gpu_id 5 > c100_output_high_loss_1pct.txt & 

# Command 3
python eval.py --subset_path /home/jennyni/ssl-mtt/sas_subset/cifar100-cl-core-idx_500.pkl --pre_lr 0.1 --gpu_id 6 > c100_output_cl-core-idx_500.txt &

# Command 4
python eval.py --subset_path /home/jennyni/ssl-mtt/init/cifar100/cifar100_2500random.pkl --pre_lr 0.1 --gpu_id 7 > c100_output_cifar100_2500random.txt &

# Command 5
python eval.py --subset_path /home/sjoshi/mtt-distillation/init/cifar100/high_loss_5pct.pkl --pre_lr 0.1 --gpu_id 4 > c100_output_high_loss_5pct.txt &

# Command 6
python eval.py --subset_path /home/jennyni/ssl-mtt/sas_subset/cifar100-cl-core-idx_2500.pkl --pre_lr 0.1 --gpu_id 5 > c100_output_cl-core-idx_2500.txt &

# Command 7
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-01-27_04:44:33None --distilled_steps 5000 --gpu_id 6 > c100_output_1per_2024-01-27_04:44:33.txt &

# Command 8
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-01-20_19:15:29None --distilled_steps 5000 --gpu_id 7 > c100_output_1per_2024-01-20_19:15:29.txt &

# Command 9
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-01-20_19:44:20None --distilled_steps 5000 --gpu_id 4 > c100_output_5per_2024-01-20_19:44:20.txt &

# Command 10
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR100/2024-01-27_04:46:32None --distilled_steps 5000 --gpu_id 5 > c100_output_5per_2024-01-27_04:46:32.txt &

# Command 11
python eval.py --result_dir /home/jennyni/krrst_orig/results/2024-01-2023:31:42.055353_cifar100_None_0 --pre_lr 0.1 --use_krrst --gpu_id 6 > c100_output_krrst_1per.txt &

# Command 12
python eval.py --result_dir /home/jennyni/krrst_orig/results/2024-01-2023:32:07.975905_cifar100_None_0 --pre_lr 0.1 --use_krrst --gpu_id 7 > c100_output_krrst_5per.txt

wait
echo "Completed"
