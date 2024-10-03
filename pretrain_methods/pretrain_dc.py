import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from utils import DiffAugment, get_network


def pretrain_dc(
    args, device, 
    x_syn, y_syn
):  
    x_syn, y_syn = x_syn.detach(), y_syn.detach()
    x_syn, y_syn = x_syn.to(device), y_syn.to(device)
    dl_syn = DataLoader(TensorDataset(x_syn, y_syn), batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # model and opt
    model = get_network(args.train_model, args.channel, args.num_pretrain_classes, args.train_img_shape).to(device)
    if args.pre_opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.pre_lr, momentum=0.9, weight_decay=args.pre_wd)
    elif args.pre_opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
    else:
        raise NotImplementedError
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[args.pre_epoch // 2], gamma=0.1)    
    
    # pretrain        
    model.train()
    for _ in trange(1, args.pre_epoch+1):
        for x_syn, y_syn in dl_syn:
            # loss
            with torch.no_grad():
                x_syn = DiffAugment(x_syn, args.dsa_strategy, param=args.dsa_param)
            loss = F.cross_entropy(model(x_syn), y_syn)
            # update
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        sch.step()

    return model
            


            