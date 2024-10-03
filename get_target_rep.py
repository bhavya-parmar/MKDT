import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam, TensorDataset, epoch, ParamDiffAug
from torchvision.models import resnet18


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, dst_train, dst_test, args.zca_trans = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    ''' organize the real dataset '''
    images_all = []
    
    print("BUILDING TRAINSET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))

    images_all = torch.cat(images_all, dim=0).to("cpu")

    result_dir = "target_rep_krrst_original"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    labels_all = []
    print(images_all.shape)
    target_model = resnet18()
    target_model.fc = nn.Identity()
    target_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    target_model.maxpool = nn.Identity()
    target_model = target_model.to(args.device)    
    checkpoint = torch.load(f"/home/jennyni/teacher_ckpt/barlow_twins_resnet18_{args.dataset.lower()}.pt", map_location="cpu")

    keys_to_remove = ["fc.weight", "fc.bias"]
    for key in keys_to_remove:
        checkpoint.pop(key, None) 
    target_model.load_state_dict(checkpoint)
    target_model.eval()

    with torch.no_grad():
        for image in images_all:
            image = image.to(args.device).unsqueeze(0)
            image_repr = target_model(image)
            labels_all.append(image_repr)
    
    labels_all = torch.cat(labels_all, dim=0)

    torch.save(labels_all.detach().cpu(), f"{result_dir}/{args.dataset}_{args.model}_target_rep_train.pt")

    print("train label shape", labels_all.shape)

    # print("BUILDING TESTSET")
    # images_test = []
    # for i in tqdm(range(len(dst_test))):
    #     sample = dst_test[i]
    #     images_test.append(torch.unsqueeze(sample[0], dim=0))

    # images_test = torch.cat(images_test, dim=0).to("cpu")

    # print("Dataset creation complete")
    
    # labels_test = []
    # print(images_test.shape)
    # target_model = resnet18()
    # target_model.fc = nn.Identity()
    # target_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    # target_model.maxpool = nn.Identity()
    # target_model = target_model.to(args.device)      
    # target_model.load_state_dict(torch.load(f"/home/jennyni/teacher_ckpt/barlow_twins_resnet18_{(args.dataset).lower()}.pt", map_location="cpu")) 

    # target_model.eval()
    # with torch.no_grad():
    #     for image in images_test:
    #         image = image.to(args.device).unsqueeze(0)
    #         image_repr = target_model(image)
    #         labels_test.append(image_repr)
    
    # labels_test = torch.cat(labels_test, dim=0)

    # torch.save(labels_test.detach().cpu(), f"{result_dir}/{args.dataset}_{args.model}_target_rep_test.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--subset', type=str, default=None, help='subset')
    parser.add_argument('--model', type=str, default='resnet18', help='model')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='False', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    args = parser.parse_args()
    main(args)



