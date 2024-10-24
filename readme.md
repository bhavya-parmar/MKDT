# MKDT
Code for Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-training of Deep Networks

## Overview

## Installation

```
git clone git@github.com:jiayini1119/MKDT.git
pip install -r requirements.txt
```


## Commands to Run the Experiments

### 1. Train the Teacher Model Using SSL and Getting Target Representation.
We obtained the teacher model trained with [Barlow Twins](https://arxiv.org/abs/2103.03230) using the checkpoint provided in the [KRRST](https://github.com/db-Lee/selfsup_dd). To get the target representation:

```
python get_target_rep.py --dataset {CIFAR10/CIFAR100/Tiny} --data_path {dataset path} --result_dir {directory to store the target representations} --device {device}
```


### 2. Get Expert Trajectories Using Knowledge Distillation.
Run the following sripts to get expert trajectories: 

CIFAR 10: `commands/buffer/barlow_twins/c10_get_trajectory.sh`

CIFAR 100: `commands/buffer/barlow_twins/c10_get_trajectory.sh`

Tiny ImageNet: `commands/buffer/barlow_twins/tiny_get_trajectory.sh`

The buffers will be saved in the directory `buffer/{ssl_algorithm}/{dataset}/{model}`.

### 3. Get the High Loss Subset.
To obtain the high loss subset for distilled dataset initialization: 
```
python get_target_rep.py --dataset {CIFAR10/CIFAR100/Tiny} --data_path {dataset path} --model {model} --num_buffers {number of buffers} --ssl_algo {Algorithm to train the ssl} --train_labels_path {path to the target representation of the dataset} --batch_train {batch size of the train dataset} --device {device}
```

For example, 

```
python get_target_rep.py --dataset CIFAR100 --data_path /home/data --model ConvNet --num_buffers 100 --ssl_algo barlow_twins --train_labels_path /home/jennyni/MKDT/target_rep_krrst_original_test/CIFAR100_resnet18_target_rep_train.pt
```

### 4. Run Distillation.



### 5. Evaluation.




## Acknowledgement
The code is based on the following repositories. 

https://github.com/GeorgeCazenavette/mtt-distillation

https://github.com/db-Lee/selfsup_dd
