import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset
from torchvision.models import resnet18

def main(args):
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    _, _, _, dst_train, _ = get_dataset(dataset=args.dataset, data_path=args.data_path)

    ''' Organize the real dataset '''
    images_all = []
    
    print("BUILDING TRAINSET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))

    images_all = torch.cat(images_all, dim=0).to("cpu")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

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

    torch.save(labels_all.detach().cpu(), f"{args.result_dir}/{args.dataset}_resnet18_target_rep_train.pt")

    print("train label shape", labels_all.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--data_path', type=str, default='/home/data', help='dataset path')
    parser.add_argument('--result_dir', type=str, default='target_rep_krrst_original_test', help='dataset path')
    parser.add_argument('--device', type=int, default=0, help='gpu number')

    args = parser.parse_args()
    main(args)



