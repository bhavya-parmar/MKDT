import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm
from utils import get_dataset, get_network, TensorDataset, epoch
import copy
import pickle

def main(args):

    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu' 
    print('Hyper-parameters: \n', args.__dict__)

    # Get the dataset   
    channel, im_size, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)

    # Location to save the buffers
    save_dir = os.path.join(args.buffer_path, args.dataset)
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Organize real datasets (images and the target representation)
    # TODO: fast loading of imagenet
    images_all = []
    
    print("BUILDING TRAINSET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))

    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.load(args.train_labels_path, map_location="cpu")
    print("train label shape", labels_all.shape)
    
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    
    # Subset using distill idx if specified 
    if args.distill_idx is not None:
        with open(args.distill_idx, "rb") as f:
            distill_idx = pickle.load(f)
            dst_train = Subset(dst_train, indices=distill_idx)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4, pin_memory=True)
    print("Dataset creation complete")
    
    # Loss function
    criterion = None
    if args.criterion == "mse":
        criterion = nn.MSELoss().to(args.device)
    elif args.criterion == "ce":
        criterion = nn.CrossEntropyLoss().to(args.device)
    
    # Get the expert trajectories
    trajectories = []
    for it in range(0, args.num_experts):

        # Sample model 
        teacher_net = get_network(args.model, channel, labels_all.shape[1], im_size).to(args.device) # get a random model

        # random_net_path = f"ssd_{args.dataset}_net.pt"
        # if os.path.exists(random_net_path):
        #     print(f"find {random_net_path}")
        #     teacher_net.load_state_dict(torch.load(random_net_path, map_location=args.device))
        # else:
        #     torch.save(teacher_net.state_dict(), random_net_path)

        if it == 0:
            print(teacher_net)
        
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)
        teacher_optim.zero_grad()

        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()]) # Random network

        # Train on real data to get trajectory
        for e in range(args.train_epochs):

            train_loss = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, device=args.device)

            print(f"Itr: {it} \tEpoch: {e} \tTrain Loss: {train_loss}")

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        trajectories.append(timestamps)
        
        # Save trajectory checkpoint
        print("Saving trajectory checkpoint")
        n = 0
        while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
            n += 1
        print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
        torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
        trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Trajectories Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers_barlow_twins', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
    parser.add_argument('--train_labels_path', type=str, default="/home/jennyni/ssl-mtt/target_rep_krrst_original/CIFAR100_resnet18_target_rep_train.pt")
    parser.add_argument('--criterion', type=str, default="mse")
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    parser.add_argument('--distill_idx', type=str, default=None, help='path to indices to distill')
    
    args = parser.parse_args()
    main(args)