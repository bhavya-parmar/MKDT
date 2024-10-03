# #!/bin/bash

# num_runs=5

# script_params="--dataset=CIFAR10 --num_experts=5 --train_labels_path /data/simclr_teacher_rep/cifar10/cifar10_512dim_r18_teacher_representations.pt"

# for device in {4..7}; do
#     for ((i=0; i<num_runs; i++)); do
#         tmux new-session -d -s "c10_simclr_collect_buffer_new_${device}_$i"
#         tmux send-keys "conda activate ddcl && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
#     done
# done


#!/bin/bash

# Number of runs
num_runs=5

# Parameters for python script
script_params="--dataset=CIFAR100 --num_experts=5 --train_labels_path /data/simclr_teacher_rep/cifar100/cifar100_512dim_r18_teacher_representations.pt"

# Loop through devices (0 to 3) and run in tmux sessions
for device in {4..7}; do
    for ((i=0; i<num_runs; i++)); do
        tmux new-session -d -s "simclr_100_new_target_${device}_$i"
        tmux send-keys "conda activate ddcl && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
    done
done


# # !/bin/bash

# # Number of runs
# num_runs=5

# # Parameters for python script
# script_params="--dataset=Tiny --num_experts=5 --model ConvNetD4 --train_labels_path /home/jennyni/ssl-mtt/target_rep_krrst_original/Tiny_resnet18_target_rep_train.pt"

# # Loop through devices (0 to 3) and run in tmux sessions
# for device in {4..7}; do
#     for ((i=0; i<num_runs; i++)); do
#         tmux new-session -d -s "timgnetnew_${device}_$i"
#         tmux send-keys "conda activate ddcl && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
#     done
# done



# # !/bin/bash

# # Number of runs
# num_runs=1

# # Parameters for python script
# script_params="--dataset=ImageNet --num_experts=25 --model ConvNetD4 --train_labels_path /home/jennyni/ssl-mtt/target_rep_krrst_original/ImageNet_resnet18_target_rep_train.pt"

# # Loop through devices (0 to 3) and run in tmux sessions
# for device in {4..7}; do
#     for ((i=0; i<num_runs; i++)); do
#         tmux new-session -d -s "new_imagenet_buffer_${device}_$i"
#         tmux send-keys "conda activate ddcl && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
#     done
# done

# ################


# #!/bin/bash

# # Number of runs
# num_runs=5

# # Parameters for python script
# script_params="--dataset=CIFAR100 --num_experts=5 --train_labels_path /home/jennyni/ssl-mtt/target_rep_bt/CIFAR100_target_rep_train.pt"

# # Loop through devices (0 to 3) and run in tmux sessions
# for device in {0..3}; do
#     for ((i=0; i<num_runs; i++)); do
#         tmux new-session -d -s "c100ftd_${device}_$i"
#         tmux send-keys "conda activate ddcl && CUDA_VISIBLE_DEVICES=$device python buffer_FTD.py $script_params" C-m
#     done
# done



# #!/bin/bash

# # Number of runs
# num_runs=5

# # Parameters for python script
# script_params="--dataset=CIFAR10 --num_experts=5 --train_labels_path /home/jennyni/sas-data-efficient-contrastive-learning/target_rep_simclr_18/cifar10_target_rep_simclr18_train.pt"

# # Loop through devices (0 to 3) and run in tmux sessions
# for device in {0..3}; do
#     for ((i=0; i<num_runs; i++)); do
#         tmux new-session -d -s "c10simclr18_${device}_$i"
#         tmux send-keys "conda activate ddcl && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
#     done
# done