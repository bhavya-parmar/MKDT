# MKDT

## 1. Train the Teacher Model Using SSL and Getting Target Representation.
We obtained the teacher model trained with [Barlow Twins](https://arxiv.org/abs/2103.03230) using the checkpoint provided in the [KRRST](https://github.com/db-Lee/selfsup_dd). To get the target representation:

```
python get_target_rep.py --dataset {CIFAR10/CIFAR100/Tiny} --data_path {dataset path} --result_dir {directory to store the target representations} --device {device}
```


## 2. Get Expert Trajectories Using Knowledge Distillation.
Run the following sripts to get expert trajectories: 

CIFAR 10: `commands/buffer/barlow_twins/c10_get_trajectory.sh`

CIFAR 100: `commands/buffer/barlow_twins/c10_get_trajectory.sh`

Tiny ImageNet: `commands/buffer/barlow_twins/tiny_get_trajectory.sh`


## 3. Get the High Loss Subset.
To obtain the high loss subset for distilled dataset initialization: 




Acknowledgement
https://github.com/GeorgeCazenavette/mtt-distillation

https://github.com/db-Lee/selfsup_dd
