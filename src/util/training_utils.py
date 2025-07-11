import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple
import os
import torch.nn as nn
from util.utils import *
import torch.nn.functional as F


def load_model(model_class, model_params:Dict, args):
    """
    model parameter를 불러오는 함수
    
    Args:
        model_name (str): model weights가 저장된 directory명
        model_class (class): 사용하는 모델이 정의된 class
        model_params (Dict): model hyper-parameter
        args (fuction): model_name, model weights가 저장된 directory path, learning rate, device parsed
    
    Returns:
        model, optimizer, epoch, trainloss, validloss
    """
    model_name = args.model_name
    model_path = args.model_path
    lr = args.lr 
    device = args.device
    model = model_class( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    # 초기값 
    train_loss = {'total':[],'loss1':[],'loss2':[],'loss3':[], 'loss4':[]}
    valid_loss = {'total':[],'loss1':[],'loss2':[],'loss3':[], 'loss4':[]}
    epoch = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    if os.path.exists(model_path+model_name+'/model.pkl'):
        checkpoint = torch.load(model_path+model_name+'/model.pkl', map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        print("Restarting at epoch", epoch)
        
    else: 
        if not os.path.exists(model_path+model_name):
            print("Creating a new dir at models/"+model_name)
            os.mkdir(model_path+model_name)
    
    return model, optimizer, epoch, train_loss, valid_loss


def run_an_epoch_x0(model,optimizer,loader,noiser,train:bool,args) -> List:
    """
    training, validation 1 epoch 진행
    for DDPM
    
    Args:
        model : 사용하는 모델
        optimizer : optimizer
        loader : dataloader
        noiser (class): Diffusion
        train (bool): True of False
        args (function): t_dim(time embedding dimension) parsed
    
    Returns:
        List : temp_loss 
    """
    temp_loss = {'total':[], 'loss1':[]}
    device = args.device
    t_dim = args.t_dim
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(device) 
        pepidx = G.pepidx.clone()   
        
        ddpm = noiser(device, args.timestep, G)
        base, used_alpha_bars_xyz, idxs = ddpm.time_embedding(t_dim, None)

        xyz = G.node_xyz.clone()
        xyz_label = xyz[G.pepidx].clone()
        
        G.node_attr = torch.cat([G.node_attr,base],dim=1)

        x_tilde, alpha_bars_xyz, _ = ddpm.forward_process(used_alpha_bars_xyz, pepidx)
        
        Loss_weight_xyz = torch.clamp(alpha_bars_xyz / (1 - alpha_bars_xyz + 1e-7), max=5.0) 

        G.node_xyz[pepidx] = x_tilde
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)  

            loss = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])

            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)

                loss = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])

        temp_loss["total"].append(loss.cpu().detach().numpy())

    return temp_loss

def run_an_epoch_x0pdl(model,optimizer,loader,noiser,train:bool,args) -> List:
    """
    training, validation 1 epoch 진행
    for DDPM
    
    Args:
        model : 사용하는 모델
        optimizer : optimizer
        loader : dataloader
        noiser (class): Diffusion
        train (bool): True of False
        args (function): t_dim(time embedding dimension) parsed
    
    Returns:
        List : temp_loss 
    """
    temp_loss = {'total':[], 'loss1':[], 'loss2':[]}
    device = args.device
    t_dim = args.t_dim
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(device) 
        pepidx = G.pepidx.clone()   
        
        ddpm = noiser(device, args.timestep, G)
        base, used_alpha_bars_xyz, idxs = ddpm.time_embedding(t_dim, None)

        xyz = G.node_xyz.clone()
        xyz_label = xyz[G.pepidx].clone()
        
        G.node_attr = torch.cat([G.node_attr,base],dim=1)

        x_tilde, alpha_bars_xyz, _ = ddpm.forward_process(used_alpha_bars_xyz, pepidx)
        
        Loss_weight_xyz = torch.clamp(alpha_bars_xyz / (1 - alpha_bars_xyz + 1e-7), max=5.0) 

        original_distmap,_,_,_,_ = self_distmap(xyz, G.batch, G.pepidx2, idxs, args.timestep*0.25)

        G.node_xyz[pepidx] = x_tilde
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)  
            output_distmap,_,_,_,_ = self_distmap(output, G.batch, G.pepidx2, idxs, args.timestep*0.25)

            loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])
            loss2 = F.mse_loss(original_distmap, output_distmap)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)
                output_distmap,_,_,_,_ = self_distmap(output, G.batch, G.pepidx2, idxs, args.timestep*0.25)

                loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])
                loss2 = F.mse_loss(original_distmap, output_distmap)

                loss = loss1 + loss2

        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["loss1"].append(loss1.cpu().detach().numpy())
        temp_loss["loss2"].append(loss2.cpu().detach().numpy())

    return temp_loss


def run_an_epoch_x0pdlvar(model,optimizer,loader,noiser,train:bool,args) -> List:
    """
    training, validation 1 epoch 진행
    for DDPM
    
    Args:
        model : 사용하는 모델
        optimizer : optimizer
        loader : dataloader
        noiser (class): Diffusion
        train (bool): True of False
        args (function): t_dim(time embedding dimension) parsed
    
    Returns:
        List : temp_loss 
    """
    temp_loss = {'total':[], 'loss1':[], 'loss2':[], 'loss3':[], 'loss4':[]}
    device = args.device
    t_dim = args.t_dim
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(device) 
        pepidx = G.pepidx.clone()   
        
        ddpm = noiser(device, args.timestep, G)
        base, used_alpha_bars_xyz, idxs = ddpm.time_embedding(t_dim, None)

        xyz = G.node_xyz.clone()
        xyz_label = xyz[G.pepidx].clone()
        
        G.node_attr = torch.cat([G.node_attr,base],dim=1)

        x_tilde, alpha_bars_xyz, _ = ddpm.forward_process(used_alpha_bars_xyz, pepidx)
        
        Loss_weight_xyz = torch.clamp(alpha_bars_xyz / (1 - alpha_bars_xyz + 1e-7), max=5.0) 

        original_distmap, oCACAdist, oCACBdist, oCACAvar, oCACBvar = self_distmap(xyz, G.batch, G.pepidx2, idxs, args.timestep*0.25)

        G.node_xyz[pepidx] = x_tilde
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)  
            output_distmap, pCACAdist, pCACBdist, pCACAvar, pCACBvar = self_distmap(output, G.batch, G.pepidx2, idxs, args.timestep*0.25)

            loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])
            loss2 = F.mse_loss(original_distmap, output_distmap)
            loss3 = F.mse_loss(oCACAdist, pCACAdist) + F.mse_loss(oCACBdist, pCACBdist) 
            loss4 = F.mse_loss(oCACAvar, pCACAvar) + F.mse_loss(oCACBvar, pCACBvar)*10

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)
                output_distmap, pCACAdist, pCACBdist, pCACAvar, pCACBvar = self_distmap(output, G.batch, G.pepidx2, idxs, args.timestep*0.25)

                loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])
                loss2 = F.mse_loss(original_distmap, output_distmap)
                loss3 = F.mse_loss(oCACAdist, pCACAdist) + F.mse_loss(oCACBdist, pCACBdist) 
                loss4 = F.mse_loss(oCACAvar, pCACAvar) + F.mse_loss(oCACBvar, pCACBvar)*10

                loss = loss1 + loss2 + loss3 + loss4

        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["loss1"].append(loss1.cpu().detach().numpy())
        temp_loss["loss2"].append(loss2.cpu().detach().numpy())
        temp_loss["loss3"].append(loss3.cpu().detach().numpy())
        temp_loss["loss4"].append(loss4.cpu().detach().numpy())

    return temp_loss

def run_an_epoch_x0pdlvar2(model,optimizer,loader,noiser,train:bool,args) -> List:
    """
    training, validation 1 epoch 진행
    for DDPM
    
    Args:
        model : 사용하는 모델
        optimizer : optimizer
        loader : dataloader
        noiser (class): Diffusion
        train (bool): True of False
        args (function): t_dim(time embedding dimension) parsed
    
    Returns:
        List : temp_loss 
    """
    temp_loss = {'total':[], 'loss1':[], 'loss2':[], 'loss3':[]}
    device = args.device
    t_dim = args.t_dim
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(device) 
        pepidx = G.pepidx.clone()   
        
        ddpm = noiser(device, args.timestep, G)
        base, used_alpha_bars_xyz, idxs = ddpm.time_embedding(t_dim, None)

        xyz = G.node_xyz.clone()
        xyz_label = xyz[G.pepidx].clone()
        
        G.node_attr = torch.cat([G.node_attr,base],dim=1)

        x_tilde, alpha_bars_xyz, _ = ddpm.forward_process(used_alpha_bars_xyz, pepidx)
        
        Loss_weight_xyz = torch.clamp(alpha_bars_xyz / (1 - alpha_bars_xyz + 1e-7), max=5.0) 

        original_distmap, oCACAdist, oCACBdist, oCACAvar, oCACBvar = self_distmap(xyz, G.batch, G.pepidx2, idxs, args.timestep*0.25)

        G.node_xyz[pepidx] = x_tilde
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)  
            output_distmap, pCACAdist, pCACBdist, pCACAvar, pCACBvar = self_distmap(output, G.batch, G.pepidx2, idxs, args.timestep*0.25)

            loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])
            loss2 = F.mse_loss(original_distmap, output_distmap) 
            loss3 = F.mse_loss(oCACAvar, pCACAvar) + F.mse_loss(oCACBvar, pCACBvar)*10

            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)
                output_distmap, pCACAdist, pCACBdist, pCACAvar, pCACBvar = self_distmap(output, G.batch, G.pepidx2, idxs, args.timestep*0.25)

                loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * Loss_weight_xyz[pepidx])
                loss2 = F.mse_loss(original_distmap, output_distmap)
                loss3 = F.mse_loss(oCACAvar, pCACAvar) + F.mse_loss(oCACBvar, pCACBvar)*10

                loss = loss1 + loss2 + loss3 

        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["loss1"].append(loss1.cpu().detach().numpy())
        temp_loss["loss2"].append(loss2.cpu().detach().numpy())
        temp_loss["loss3"].append(loss3.cpu().detach().numpy())

    return temp_loss

def run_an_epoch_ep(model,optimizer,loader,noiser,train:bool,args) -> List:
    """
    training, validation 1 epoch 진행
    for DDPM
    
    Args:
        model : 사용하는 모델
        optimizer : optimizer
        loader : dataloader
        noiser (class): Diffusion
        train (bool): True of False
        args (function): t_dim(time embedding dimension) parsed
    
    Returns:
        List : temp_loss 
    """
    temp_loss = {'total':[]}
    device = args.device
    t_dim = args.t_dim
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(device) 
        pepidx = G.pepidx.clone() 
        # G.node_xyz = G.node_xyz * 10  
        
        ddpm = noiser(device, args.timestep, G)
        base, used_alpha_bars_xyz, idxs = ddpm.time_embedding(t_dim, None)
        
        G.node_attr = torch.cat([G.node_attr,base],dim=1)

        x_tilde, _, epsilon = ddpm.forward_process(used_alpha_bars_xyz, pepidx)
        x_t = x_tilde.clone()
        G.node_xyz[pepidx] = x_tilde
        
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)  

            loss = F.mse_loss(output[pepidx], epsilon)

            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)

                loss = F.mse_loss(output[pepidx], epsilon)

        temp_loss["total"].append(loss.cpu().detach().numpy())

    return temp_loss


def run_an_epoch_eppdl(model,optimizer,loader,noiser,train:bool,args) -> List:
    """
    training, validation 1 epoch 진행
    for DDPM
    
    Args:
        model : 사용하는 모델
        optimizer : optimizer
        loader : dataloader
        noiser (class): Diffusion
        train (bool): True of False
        args (function): t_dim(time embedding dimension) parsed
    
    Returns:
        List : temp_loss 
    """
    temp_loss = {'total':[], 'loss1':[], 'loss2': []}
    device = args.device
    t_dim = args.t_dim
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(device) 
        pepidx = G.pepidx.clone()   

        original_xyz = G.node_xyz.clone() 
        
        ddpm = noiser(device, args.timestep, G)
        base, used_alpha_bars_xyz, idxs = ddpm.time_embedding(t_dim, None)

        odistmap = get_distmap(original_xyz, G.batch, G.pepidx2, idxs, args.timestep*0.25)
        
        G.node_attr = torch.cat([G.node_attr,base],dim=1)

        x_tilde, _, epsilon = ddpm.forward_process(used_alpha_bars_xyz, pepidx)
        G.node_xyz[pepidx] = x_tilde
        x_t = G.node_xyz.clone()
        
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr) 
            x0_pred = get_x0(used_alpha_bars_xyz, output, x_t)
            x_t[pepidx] = x0_pred[pepidx]
            pdistmap = get_distmap(x_t, G.batch, G.pepidx2, idxs, args.timestep*0.25)

            loss1 = F.mse_loss(output[pepidx], epsilon)
            loss2 = F.mse_loss(pdistmap, odistmap)
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)
                x0_pred = get_x0(used_alpha_bars_xyz, output, x_t)
                x_t[pepidx] = x0_pred[pepidx]
                pdistmap = get_distmap(x_t, G.batch, G.pepidx2, idxs, args.timestep*0.25)

                loss1 = F.mse_loss(output[pepidx], epsilon)
                loss2 = F.mse_loss(pdistmap, odistmap)
                loss = loss1 + loss2

        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["loss1"].append(loss1.cpu().detach().numpy())
        temp_loss["loss2"].append(loss2.cpu().detach().numpy())

    return temp_loss


def save_model(epoch:int, model, optimizer, train_loss, valid_loss, args) -> None:
    """
    model을 현재 epoch에 대해 model.pkl에 save, best model이라면 best.pkl에 save
    
    Args:
        epoch (int): 현재 epoch
        model : 사용한 model
        optimizer : optimizer
        train_loss : train loss
        valid_loss : validation loss
        args : model_path, model_name parsed
    
    Returns:
        
    """
    # Update the best model if necessary:
    if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
            }, args.model_path+'/'+args.model_name+"/best.pkl")
            
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss
        }, args.model_path+'/'+args.model_name+"/model.pkl")


def self_distmap(xyz, G_batch, pepidx2, idxs, timestep_crit):
    idxs = torch.tensor(idxs).to(device=xyz.device)
    distmap_list = [] 
    for batch_num in G_batch.unique().tolist(): 
        node_mask = (G_batch == batch_num)
        
        selected = idxs[node_mask]
        assert torch.all(selected == selected[0]), "Batch idxs Timestep Error!"
        timestep = idxs[node_mask][0]
        if timestep > timestep_crit: continue 

        pepidx = pepidx2[batch_num]
        batch_xyz = xyz[node_mask]
        pepxyz = batch_xyz[pepidx]

        distmap = torch.cdist(pepxyz, pepxyz)
        distmap_list.append(distmap.flatten())

        CAxyz = pepxyz[::2]
        CBxyz = pepxyz[1::2]
        CACAdist = torch.norm(CAxyz[1:] - CAxyz[:-1], dim=1)
        CACBdist = torch.norm(CAxyz - CBxyz, dim=1)
        CACAvar = torch.var(CACAdist)
        CACBvar = torch.var(CACBdist)

    if len(distmap_list) == 0:
        no_value = torch.tensor([0.0])
        distmap_list.append(no_value.clone())
        CACAdist = no_value.clone() 
        CACBdist = no_value.clone()
        CACAvar = no_value.clone()
        CACBvar = no_value.clone()

    return torch.cat(distmap_list), CACAdist, CACBdist, CACAvar, CACBvar


def get_distmap(xyz, G_batch, pepidx2, idxs, timestep_crit):
    idxs = torch.tensor(idxs).to(device=xyz.device)
    distmap_list = [] 
    for batch_num in G_batch.unique().tolist(): 
        node_mask = (G_batch == batch_num)
        
        selected = idxs[node_mask]
        assert torch.all(selected == selected[0]), "Batch idxs Timestep Error!"
        timestep = idxs[node_mask][0]
        if timestep > timestep_crit: continue 

        pepidx = pepidx2[batch_num]
        batch_xyz = xyz[node_mask]
        pepxyz = batch_xyz[pepidx]

        distmap = torch.cdist(pepxyz, pepxyz)
        distmap_list.append(distmap.flatten())

    if len(distmap_list) == 0:
        no_value = torch.tensor([0.0])
        distmap_list.append(no_value.clone())

    return torch.cat(distmap_list)


def dist_edges(G):
    edge_index = G.edge_index 
    xyz = G.node_xyz

    u,v = edge_index 

    dist_map = torch.sqrt(torch.sum((xyz[None,:,:] - xyz[:,None,:])**2, dim=2))

    diff_xyz = dist_map[u,v].unsqueeze(1)

    edge_attr = torch.cat([diff_xyz,G.edge_attr], dim=1)
    G.edge_attr = edge_attr

    return G


def get_x0(alpha_bar_t, eps_pred, x_t): 

    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar 

    return x0_pred
