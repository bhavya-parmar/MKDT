import os
import random
import argparse
import pickle
import wandb
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
from utils import get_dataset, get_network, SoftLabelDataset, ParamDiffAug
from pretrain_methods import pretrain_dc, pretrain_frepo, pretrain_mse
from linear_evaluation import le_run
from linear_eval_sgd import linear_sgd_run
from full_finetune import ft_run

def aggregate_results(results):
    aggregated_results = {}
    for dataset in results[0].keys():
        dataset_results = np.array([result[dataset] for result in results])
        mean = np.mean(dataset_results, axis=0)
        std = np.std(dataset_results, axis=0)
        aggregated_results[dataset] = {'mean': mean.tolist(), 'std': std.tolist()}
            
    return aggregated_results

def main(args):
    print(args)
    # target_datasets = ["Tiny", "CIFAR100", "CIFAR10", "aircraft", "cub2011", "dogs", "flowers"]
    # target_datasets = ["urbancars", "waterbirds", "bffhq", "celeba", "camelyon", "terra_incognita", "office", "VLCS", "PACS"]

    if args.supervised_pretrain_method is not None and not args.supervised_pretrain:
        raise ValueError        

    # if not args.distribution_shift:
    target_datasets = ["CIFAR100", "CIFAR10", "aircraft", "cub2011", "dogs", "flowers", "Tiny"]
    # else:
    #     target_datasets = ["urbancars", "waterbirds", "bffhq", "celeba", "camelyon", "terra_incognita", "office", "VLCS", "PACS"]


    # Wandb init
    wandb.init(
        project="data_distillation_evaluation",
        config=args
    )

    if args.result_dir is not None:
        path = os.path.basename(args.result_dir.replace('/', '_')) 
    else:
        path = f"train_{args.train_dataset}_{os.path.basename(args.subset_path) if args.subset_path is not None else None}_{args.num_subset_within}"

    print(f"Evaluate for {path}")

    final_res = {}

    for td in target_datasets:
        try:
            args.test_dataset = td
            res_list = []
            device = torch.device(f"cuda:{args.gpu_id}")
            torch.cuda.set_device(device)

            # default augment
            args.dsa_param = ParamDiffAug()
            args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate' 

            # seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            # data
            _, train_img_size, num_pretrain_classes, dst_train_original, _, args.zca_trans = get_dataset(args.train_dataset, args.data_path, args.batch_size, args.subset, args=args)     
            args.channel, im_size, num_classes, dst_train, dst_test, _ = get_dataset(args.test_dataset, args.data_path, args.batch_size, args.subset, subset_size=args.subset_frac, epc=args.epc, args=args)
            # Used for test (le and ft)
            dl_tr = torch.utils.data.DataLoader(dst_train, batch_size=args.test_batch_size, shuffle=False if args.test_algorithm == "linear_evaluation" else True, num_workers=4, pin_memory=True)
            dl_te = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

            print("Test training data: ", len(dst_train))

            labels_all = torch.load(args.label_path, map_location="cpu")
            
            args.train_img_shape = train_img_size
            args.test_img_shape = im_size
            args.num_classes = num_classes
            args.num_pretrain_classes = num_pretrain_classes

            if not args.supervised_pretrain and args.result_dir is None:
                if args.subset_path is not None:
                    if args.num_subset_within is not None:
                        with open(args.subset_path, "rb") as f:
                            subset_idx = pickle.load(f)[0:args.num_subset_within]
                    else:
                        with open(args.subset_path, "rb") as f:
                            subset_idx = pickle.load(f)
                
                else:
                    subset_idx = None
                    
                dst_train_initial = SoftLabelDataset(dst_train_original, labels_all, subset_idx)
                print(f"num_images: {len(dst_train_initial)}")
                dl_syn = torch.utils.data.DataLoader(dst_train_initial, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
                # visualize
                all_images = torch.stack([img for img, _ in dst_train_initial], dim=0)

                # torch.save(all_images.cpu(), "newtinyrandom5per.pt")
            
                grid = make_grid(all_images.clone(), nrow=100)
                wandb.log({"initial images": wandb.Image(grid.detach().cpu())})
                del all_images
            

            else:
                if args.supervised_pretrain:
                    x_syn = torch.load(f"/home/jennyni/ssl-mtt/synthetic_data/{args.train_dataset}/{args.supervised_pretrain_method}/x_syn.pt", map_location="cpu").detach()
                    y_syn = torch.load(f"/home/jennyni/ssl-mtt/synthetic_data/{args.train_dataset}/{args.supervised_pretrain_method}/y_syn.pt", map_location="cpu").detach()
                    if args.supervised_pretrain_method != "kip" and args.supervised_pretrain_method != "frepo":
                        y_syn = y_syn.long()

                else:
                    if args.use_krrst:
                        x_syn = torch.load(f"{args.result_dir}/x_syn.pt") 
                        x_syn = x_syn.detach().cpu()
                    else:
                        x_syn = torch.load(f"{args.result_dir}/images_{args.distilled_steps}.pt")
                        x_syn = x_syn.detach().cpu()
                    try:
                        syn_lr = torch.load(f"{args.result_dir}/synlr_{args.distilled_steps}.pt") 
                        args.pre_lr = syn_lr.detach().cpu()
                    except:
                        print("No syn lr found, fall back to original pre_lr")
                    try:
                        if args.use_krrst:
                            y_syn = torch.load(f"{args.result_dir}/y_syn.pt")
                            y_syn = y_syn.detach().cpu()
                        else:
                            y_syn = torch.load(f"{args.result_dir}/labels_{args.distilled_steps}.pt")
                    except:
                        ### not useful
                        # target model (TODO: change to a resnet trained with SimCLR)
                        from bt_model import BTModel
                        target_model = BTModel(feature_dim=1024, dataset=args.train_dataset)
                        target_model = target_model.to(device)        
                        target_model.load_state_dict(torch.load(f"/home/sjoshi/krrst_orig/saved_bt_models/barlow_twins_resnet18_{args.train_dataset}.pth", map_location="cpu"), strict=False)
            

                        target_model.eval()
                        with torch.no_grad():
                            x_syn = x_syn.to(device)
                            y_syn = target_model(x_syn)
                            y_syn = y_syn.detach().clone()
                
                assert x_syn.requires_grad == False and y_syn.requires_grad == False
                assert x_syn.grad is None and y_syn.grad is None


                if not args.supervised_pretrain:
                    assert y_syn.shape[-1] == labels_all.shape[1]

                x_syn, y_syn = x_syn.detach(), y_syn.detach()
                x_syn, y_syn = x_syn.to(device), y_syn.to(device)

                dst_train_final = TensorDataset(x_syn, y_syn)
                print(f"num_images: {len(dst_train_final)}")
                dl_syn = DataLoader(dst_train_final, batch_size=args.batch_size, shuffle=True, num_workers=0)

                # visualize
                grid = make_grid(x_syn.clone(), nrow=100)
                wandb.log({"final images": wandb.Image(grid.detach().cpu())})


            args.num_target_features = labels_all.shape[1]
            print(args.pre_lr)

            ckpt_dir = "ckpt_dir"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            # check random encoder perf
            if args.use_random:
                random_model =get_network(args.train_model, args.channel, args.num_target_features, args.train_img_shape, fix_net=True).to(device)
                if args.test_algorithm == "linear_evaluation":
                    rd_loss, rd_acc = le_run(args, device, random_model, dl_tr, dl_te)
                else:
                    rd_loss, rd_acc = ft_run(args, device, random_model, dl_tr, dl_te)
                print("rd_loss", rd_loss)
                print("rd_acc", rd_acc)
                wandb.log({f"{td}_random_loss": rd_loss, f"{td}_random_accuracy": rd_acc})
                del random_model
                res_list.append(rd_acc)

            # final performance
            # ckpt_file_final = f"{ckpt_dir}/{path}_new2_{args.test_algorithm}_pre_epoch_{args.pre_epoch}_{args.test_epoch}_{args.seed}_{args.train_dataset}_{args.train_model}_{args.distilled_steps}_{args.pre_lr}_final_full.pt"
            # if os.path.exists(ckpt_file_final):
            #     print(f"find {ckpt_file_final}")
            #     final_model = get_network(args.train_model, args.channel, args.num_target_features, args.train_img_shape, fix_net=True).to(device)
            #     final_model.load_state_dict(torch.load(ckpt_file_final, map_location=device))
            # else:
            #     final_model = pretrain_run(args, device, dl_syn)
            #     torch.save(final_model.state_dict(), ckpt_file_final)

            else:
                if args.supervised_pretrain_method == "random_sup" or args.supervised_pretrain_method == "kmeans" or args.supervised_pretrain_method == "dsa" or args.supervised_pretrain_method == "dm" or args.supervised_pretrain_method == "mtt":
                    final_model = pretrain_dc(args, device, x_syn, y_syn)
                elif args.supervised_pretrain_method == "kip" or args.supervised_pretrain_method == "frepo":
                    final_model = pretrain_frepo(args, device, x_syn, y_syn)
                else:
                    final_model = pretrain_mse(args, device, dl_syn)

                # torch.save(final_model.state_dict(), ckpt_file_final)
                if args.test_algorithm == "linear_evaluation":
                    if td == "celeba" or td == "camelyon":
                        final_loss, final_acc = linear_sgd_run(args, device, final_model, dl_tr, dl_te)
                    else:
                        final_loss, final_acc = le_run(args, device, final_model, dl_tr, dl_te)
                else:
                    final_loss, final_acc = ft_run(args, device, final_model, dl_tr, dl_te)
                del final_model
                print("final_loss", final_loss)
                print("final_acc", final_acc)
                wandb.log({f"{td}_final_loss": final_loss, f"{td}_subset_final_accuracy": final_acc})
                res_list.append(final_acc)
            final_res[td] = res_list
        except Exception as e:
            print(f"Error evaluating on {td}")
            print(e)

    print(final_res)

    wandb.log(final_res)

    wandb.finish(quiet=True)

    return final_res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=0)

    # data
    parser.add_argument('--data_path', type=str, default="/data")
    parser.add_argument('--train_dataset', type=str, default="CIFAR100")
    parser.add_argument('--test_dataset', type=str, default="CIFAR100")
    parser.add_argument('--num_workers', type=int, default=0)

    # label
    parser.add_argument('--label_path', type=str, default="/home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR100_resnet18_target_rep_train.pt")

    # subset
    # parser.add_argument('--subset_path', type=str, default="/home/jennyni/mtt-distillation/sorted_idx/sorted_indices_CIFAR100.pkl")
    parser.add_argument('--subset_path', type=str, default=None)
    parser.add_argument('--num_subset_within', type=int, default=None)

    # original
    parser.add_argument('--use_flip', action="store_true")
    parser.add_argument('--use_diff_aug', action="store_true")
    parser.add_argument('--zca', action="store_true")
    parser.add_argument('--subset', type=str, default=None, help='subset')

    
    # hparams for model
    parser.add_argument('--train_model', type=str, default="ConvNet")


    # args for logging 
    parser.add_argument("--pre_data_name", type=str)
    parser.add_argument("--distill_method", type=str)
    parser.add_argument("--results_csv", default="mkdt_results.csv")
    
    # hparms for pretrain
    parser.add_argument('--pre_opt', type=str, default="sgd") 
    parser.add_argument('--pre_epoch', type=int, default=20)  # 20 for linear probe
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.1) # 0.174
    parser.add_argument('--pre_wd', type=float, default=1e-4)
    parser.add_argument('--pre_sch', action="store_true")

    # hparms for test
    # parser.add_argument('--test_algorithm', type=str, choices=['linear_evaluation', 'full_finetune'], default="linear_evaluation") # Changed
    parser.add_argument('--test_algorithm', type=str, choices=['linear_evaluation', 'full_finetune'], default="linear_evaluation")
    parser.add_argument('--test_opt', type=str, default="sgd")
    parser.add_argument('--test_epoch', type=int, default=50) 
    parser.add_argument('--test_lr', type=float, default=0.05)
    parser.add_argument('--test_wd', type=float, default=0.0)
    parser.add_argument('--test_batch_size', type=int, default=256) # 256 for linear evaluation, 512 for ft
    # parser.add_argument('--test_batch_size', type=int, default=512) # 256 for linear evaluation, 512 for ft

    parser.add_argument('--subset_frac', type=float, default=None)
    parser.add_argument('--epc', type=int, default=None) # examples per class

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)

    # for distilled set directory
    parser.add_argument('--result_dir', type=str, default=None)

    parser.add_argument('--distilled_steps', type=int, default=1000) 
    parser.add_argument('--use_krrst', action="store_true")
    parser.add_argument('--use_random', action="store_true")
    parser.add_argument('--distribution_shift', action="store_true")
    parser.add_argument('--supervised_pretrain', action="store_true")
    parser.add_argument('--supervised_pretrain_method', '-s', type=str, default=None, choices=["random_sup", "kmeans", "dsa", "dm", "mtt", "kip", "frepo"])



    # args = parser.parse_args()

    seeds = [0, 1, 2]
    results = []

    for seed in seeds:
        args = parser.parse_args()
        args.seed = seed
        results.append(main(args))

    final_aggregated_results = aggregate_results(results)
    print(final_aggregated_results)

    curr_results_df = pd.DataFrame(index=[0])
    curr_results_df["timestamp"] = pd.Timestamp.now()
    args_dict = vars(args)
    for key in args_dict:
        try:
            curr_results_df[key] = args_dict[key]
        except:
            print(f"Couldn't save {key}")
        
    for dataset, stats in final_aggregated_results.items():
        mean = stats['mean'][0]
        std = stats['std'][0]
        print(f"{dataset}: {mean:.2f} ± {std:.2f}")
        curr_results_df[dataset] = "$" + f"{mean:.2f}" + "\scriptstyle{ \pm " + f"{std:.2f}" + "}" + "$"
    
    print(curr_results_df)  
    if os.path.exists(args.results_csv):
        results_df = pd.read_csv(args.results_csv)
    else:
        results_df = pd.DataFrame()

    results_df = pd.concat([results_df, curr_results_df], ignore_index=True)
    results_df.to_csv(args.results_csv, index=False)

    # wandb.log(final_aggregated_results)
    wandb.finish(quiet=True)