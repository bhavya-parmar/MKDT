# MKDT (ICLR 2025)
Code for [Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-training of Deep Networks](https://arxiv.org/abs/2410.02116)

## Overview
Dataset distillation (DD) generates small synthetic datasets that can efficiently train
deep networks with a limited amount of memory and compute. Despite the success
of DD methods for supervised learning, DD for self-supervised pre-training of deep
models has remained unaddressed. Pre-training on unlabeled data is crucial for
efficiently generalizing to downstream tasks with limited labeled data. In this work,
we propose the first effective DD method for SSL pre-training. First, we show,
theoretically and empirically, that naïve application of supervised DD methods to
SSL fails, due to the high variance of the SSL gradient. Then, we address this issue
by relying on insights from knowledge distillation (KD) literature. Specifically, we
train a small student model to match the representations of a larger teacher model
trained with SSL. Then, we generate a small synthetic dataset by matching the
training trajectories of the student models. As the KD objective has considerably
lower variance than SSL, our approach can generate synthetic datasets that can
successfully pre-train high-quality encoders. Through extensive experiments, we
show that our distilled sets lead to up to 13% higher accuracy than prior work,
on a variety of downstream tasks, in the presence of limited labeled data.

## Installation

```
git clone git@github.com:jiayini1119/MKDT.git
pip install -r requirements.txt
```


## Commands to Run the Experiments

### 1. Train the Teacher Model Using SSL and Getting Target Representation.
#### For Barlow Twins:
We obtained the teacher model trained with [Barlow Twins](https://arxiv.org/abs/2103.03230) using the checkpoint provided in the [KRRST](https://github.com/db-Lee/selfsup_dd). Download and save the checkpoints under the repository `/krrst_teacher_ckpt`.

#### For SimCLR:
We obtained the teacher model trained with [SimCLR](https://arxiv.org/abs/2002.05709) using the checkpoint provided in the [SAS](https://github.com/BigML-CS-UCLA/sas-data-efficient-contrastive-learning).

To get the target representation:

```
python get_target_rep.py --dataset {CIFAR10/CIFAR100/Tiny} --model {model: ConvNetD4 for TinyImageNet and ConvNet for other datasets} --ssl_algorithm {barlow_twins/simclr} --data_path {dataset path} --result_dir {directory to store the target representations} --device {device}
```

By default, the target representations will be saved in `/{result_dir}_{ssl_algorithm}/{dataset}_target_rep_train.pt`.


### 2. Get Expert Trajectories Using Knowledge Distillation.
Run the following sripts to get expert trajectories: 

CIFAR 10: `commands/buffer/{ssl_algorithm}/c10_get_trajectory.sh`

CIFAR 100: `commands/buffer/{ssl_algorithm}/c10_get_trajectory.sh`

Tiny ImageNet: `commands/buffer/{ssl_algorithm}/tiny_get_trajectory.sh`

The buffers will be saved in the directory `buffer/{ssl_algorithm}/{dataset}/{model}`.

### 3. Get the High Loss Subset.
To obtain the high loss subset for distilled dataset initialization: 
```
python get_target_rep.py --dataset {CIFAR10/CIFAR100/Tiny} --data_path {dataset path} --model {model} --num_buffers {number of buffers} --ssl_algo {Algorithm to train the ssl} --train_labels_path {path to the target representation of the dataset} --batch_train {batch size of the train dataset} --device {device}
```

For example, 

```
python get_target_rep.py --dataset CIFAR100 --data_path /home/data --model ConvNet --num_buffers 100 --ssl_algo barlow_twins --train_labels_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR100_target_rep_train.pt
```

### 4. Run Distillation.

Run the following sripts to distill the dataset (SSL algorithm using barlow twins):

**CIFAR 10** 

2 percent: `commands/distill/barlow_twins/CIFAR10/2_per.sh`

5 percent: `commands/distill/barlow_twins/CIFAR10/5_per.sh`

**CIFAR 100**

2 percent: `commands/distill/barlow_twins/CIFAR100/2_per.sh`

5 percent: `commands/distill/barlow_twins/CIFAR100/5_per.sh`

**Tiny ImageNet**

2 percent: `commands/distill/barlow_twins/Tiny/2_per.sh`

5 percent: `commands/distill/barlow_twins/Tiny/5_per.sh`


### 5. Evaluation.

The following scripts contains the commands to run the evaluation for different subsets (e.g., MKDT, random, KRRST).

Append `--subset_frac {0.01/0.05/0.1/0.5}` to the command to evaluate the datasets using different evaluation subset fractions

**CIFAR 10** 

2 percent: `commands/eval/c10_2per.sh --subset_frac {0.01/0.05/0.1/0.5}`

5 percent: `commands/eval/c10_5per.sh --subset_frac {0.01/0.05/0.1/0.5}`

**CIFAR 100**

2 percent: `commands/eval/c100_2per.sh --subset_frac {0.01/0.05/0.1/0.5}`

5 percent: `commands/eval/c100_5per.sh --subset_frac {0.01/0.05/0.1/0.5}`

**Tiny ImageNet**

2 percent: `commands/eval/tiny_2per.sh --subset_frac {0.01/0.05/0.1/0.5}`

5 percent: `commands/eval/tiny_5per.sh --subset_frac {0.01/0.05/0.1/0.5}`

You can visualize the tables comparing different subset results for a dataset using `commands/exp_plotting.ipynb`.


## Acknowledgement
The code is based on the following repositories. 

https://github.com/GeorgeCazenavette/mtt-distillation

https://github.com/db-Lee/selfsup_dd

## BibTeX
```
@inproceedings{joshi2025kd,
  title={Dataset Distillation via Knowledge Distillation: Towards Efficient Self-Supervised Pre-Training of Deep Networks},
  author={Joshi, Siddharth and Ni, Jiayi and Mirzasoleiman, Baharan},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
