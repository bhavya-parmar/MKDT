import torch
import torch.nn.functional as F
from tqdm import trange
from utils import get_network



def pretrain_mse(
    args, device, 
    dl_syn
):  

    # model and opt
    model = get_network(args.train_model, args.channel, args.num_target_features, args.train_img_shape).to(device)
    if args.pre_opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.pre_lr, momentum=0.9, weight_decay=args.pre_wd)
    elif args.pre_opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
    else:
        raise NotImplementedError
    
    # pretrain        
    model.train()
    for _ in trange(1, args.pre_epoch+1):
        for x_syn, y_syn in dl_syn:
            # loss
            x_syn = x_syn.to(device)
            y_syn = y_syn.to(device)
            loss = F.mse_loss(model(x_syn), y_syn)
            # update
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model