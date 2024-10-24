from tqdm import trange
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils import get_network
from utils import InfIterator

def linear_sgd_run(
    args, device, 
    init_model, 
    dl_tr, dl_te,
):  

    # model
    model = get_network(args.train_model, args.channel, args.num_classes, args.test_img_shape, fix_net=True).to(device)
    if hasattr(init_model, "classifier"):
        del init_model.classifier
    model.load_state_dict(init_model.state_dict(), strict=False)
    model.fc = nn.Identity()

    model.eval()
    with torch.no_grad():
        # tr feature
        F_tr, Y_tr = [], []
        for x_tr, y_tr in tqdm(dl_tr, desc="encoding train"):
            x_tr = x_tr.to(device)
            f_tr = model(x_tr).cpu()
            F_tr.append(f_tr); Y_tr.append(y_tr)
        F_tr, Y_tr = torch.cat(F_tr, dim=0), torch.cat(Y_tr, dim=0)
        num_features = F_tr.shape[-1]
        dl_feature_tr = DataLoader(TensorDataset(F_tr, Y_tr), batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        iter_feature_tr = InfIterator(dl_feature_tr)

        # te feature
        F_te, Y_te = [], []
        for x_te, y_te in tqdm(dl_te, desc="encoding test"):
            x_te = x_te.to(device)
            f_te = model(x_te).cpu()
            F_te.append(f_te); Y_te.append(y_te)
        F_te, Y_te = torch.cat(F_te, dim=0), torch.cat(Y_te, dim=0)
        dl_feature_te = DataLoader(TensorDataset(F_te, Y_te), batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)    

    # opt
    linear = nn.Linear(num_features, args.num_classes).to(device)
    opt = torch.optim.SGD(linear.parameters(), lr=args.test_lr, momentum=0.9, weight_decay=args.test_wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.test_epoch)
    
    linear.train()
    for _ in trange(args.test_epoch):
        x_tr, y_tr = next(iter_feature_tr)
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        loss = F.cross_entropy(linear(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    
    linear.eval()
    with torch.no_grad():        
        loss, acc, denominator = 0., 0., 0.
        for x_te, y_te in dl_feature_te:
            x_te, y_te = x_te.to(device), y_te.to(device)
            l_te = linear(x_te)
            loss += F.cross_entropy(l_te, y_te, reduction="sum")
            acc += torch.eq(l_te.argmax(dim=-1), y_te).float().sum()
            denominator += x_te.shape[0]
        loss /= denominator; acc /= (denominator/100.)

    del model, linear

    return loss.item(), acc.item()