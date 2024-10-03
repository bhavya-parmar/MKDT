#!/bin/bash

cleanup() {
    echo "Terminating all running processes..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT

# Command 1
python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --subset_path /home/jennyni/ssl-mtt/init/tiny_imagenet/tiny_1000random.pkl --gpu_id 4 > tiny_1000random.txt & 

# Command 2
python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --subset_path /home/sjoshi/mtt-distillation/init/tiny_imagenet/high_loss_1pct.pkl --gpu_id 4 > tiny_high_loss_1_pc.txt & 

# Command 4
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-01-19_20:22:02None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --distilled_steps 5000 --gpu_id 5 > tiny_1000_2024-01-19_20:22.txt &

# Command 5
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-01-27_14:42:28None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --distilled_steps 5000 --gpu_id 5 > tiny_1000_2024-01-27_14:42:28.txt &

# Command 7
python eval.py --result_dir /home/jennyni/krrst_orig/results/2024-01-1823:23:36.967381_aircraft_None_0 --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --use_krrst --gpu_id 6 > tiny_krrst_1_per.txt &

# Command 8
python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --subset_path /home/jennyni/ssl-mtt/init/tiny_imagenet/tiny_5000random.pkl --gpu_id 6 > tiny_random_2500.txt &

# Command 9
python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --subset_path /home/sjoshi/mtt-distillation/init/tiny_imagenet/high_loss_5pct.pkl --gpu_id 7 > tiny_high_loss_5pct.txt &

# Command 10
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-01-27_10:05:50None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --distilled_steps 5000 --gpu_id 7 > tiny_2500_2024-01-27_10:05:50.txt &

# Command 11
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-01-21_10:17:54None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --distilled_steps 5000 --gpu_id 6 > tiny_2500_2024-01-21_10:17:54.txt &

# Command 12
# python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/ssl-mtt/target_rep_bt/Tiny_target_rep_train.pt --gpu_id 2 > tiny_full.txt

wait
echo "Completed"
