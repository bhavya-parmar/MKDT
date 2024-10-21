import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam, TensorDataset, epoch, ParamDiffAug
import copy
from bt_model import BTModel
from tqdm import tqdm
import pickle
from reparam_module import ReparamModule
import numpy as np



def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, _, dst_train, dst_test, args.zca_trans = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # trainloader
    images_all = []
    
    print("BUILDING TRAINSET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))

    args.model = "ConvNet" if (args.dataset != "Tiny" and args.dataset != "ImageNet") else "ConvNetD4"
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.load(f"/data/simclr_teacher_rep/{args.dataset.lower()}/{args.dataset.lower()}_512dim_r18_teacher_representations.pt", map_location="cpu")
    
    print("train label shape", labels_all.shape)
    
    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=False, num_workers=4, pin_memory=True)

    epoch_losses = {}

    num_examples = len(trainloader.dataset)
    print(num_examples)
    epoch_losses = np.zeros(num_examples) 
    num_buffers = 100

    for num_buffer in range(1, num_buffers + 1):
        checkpoint_path = f'/home/jennyni/ssl-mtt/buffers_simclr_rep/{args.dataset}_NO_ZCA_None/{args.model}/replay_buffer_{num_buffer-1}.pt'
        param_all = torch.load(checkpoint_path)
        last_checkpoint = param_all[0][-1] 
        # # Change to use the first epoch
        # last_checkpoint = param_all[0][1]
        last_checkpoint = torch.cat([p.data.to(args.device).reshape(-1) for p in last_checkpoint], 0)

        teacher_net = get_network(args.model, channel, labels_all.shape[1], im_size).to(args.device)
        teacher_net = ReparamModule(teacher_net)

        for i, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            with torch.no_grad():
                outputs = teacher_net(inputs, flat_param=last_checkpoint)
                mse_losses = torch.mean((outputs - targets) ** 2, dim=1)
                individual_losses = mse_losses.detach().cpu().numpy()
                epoch_losses[i * args.batch_train:(i + 1) * args.batch_train] += individual_losses

    epoch_losses /= num_buffers

    sorted_indices = np.argsort(epoch_losses)[::-1]

    print("Sorted indices of high loss subset:", sorted_indices)

    dset_name = (args.dataset).lower() if args.dataset != "Tiny" else "tiny_imagenet"

    if args.dataset != "ImageNet":
        top_2_percent_idx = sorted_indices[:int(0.02 * num_examples)]
        top_5_percent_idx = sorted_indices[:int(0.05 * num_examples)]

        with open(f'/home/jennyni/ssl-mtt/init/{dset_name}/{args.dataset}_simclr_2_high_loss_indices.pkl', 'wb') as f:
            pickle.dump(top_2_percent_idx, f)
        # with open(f'/home/jennyni/ssl-mtt/init/{dset_name}/{args.dataset}_5_high_loss_indices.pkl', 'wb') as f:
        #     pickle.dump(top_5_percent_idx, f)
        
    else:
        top_1000 = sorted_indices[:1000]
        with open(f'/home/jennyni/ssl-mtt/init/{dset_name}/{args.dataset}_simclr_1000_high_loss_indices.pkl', 'wb') as f:
            pickle.dump(top_1000, f)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--subset', type=str, default=None, help='subset')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='False', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)


