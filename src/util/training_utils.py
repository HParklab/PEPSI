import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple
import os
import torch.nn as nn
from util.utils import *
import torch.nn.functional as F


def load_model(model_class, model_params:Dict, args):
   
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


def run_an_epoch(model,optimizer,loader,noiser,train:bool,args) -> List:

    temp_loss = {'total':[], 'loss1':[], 'loss2':[]}
    model = model.to(args.device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue
        G = G.to(args.device) 
        pepidx = G.pepidx.clone()   
        
        ddpm = noiser(args.device, args.timestep, G)
        t_embed, used_alpha_bars = ddpm.time_embedding(args.t_dim, None)

        xyz = G.node_xyz.clone()
        xyz_label = xyz[G.pepidx].clone()
        
        G.node_attr = torch.cat([G.node_attr,t_embed],dim=1)

        x_tilde = ddpm.forward_process(used_alpha_bars)
        
        snr_weight = torch.clamp(used_alpha_bars / (1 - used_alpha_bars + 1e-7), max=5.0) 

        original_dist_value = pad_xyz_by_idx_condition(xyz, G.batch, G.target_idx2, torch.sqrt(snr_weight))

        G.node_xyz[pepidx] = x_tilde
        G = dist_edges(G)

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)

            output_xyz = xyz.clone() 
            output_xyz[G.pepidx] = output[G.pepidx]
            output_dist_value = pad_xyz_by_idx_condition(output_xyz, G.batch, G.target_idx2, torch.sqrt(snr_weight))

            loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * snr_weight[pepidx])
            loss2 = F.mse_loss(original_dist_value, output_dist_value)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)

                output_xyz = xyz.clone() 
                output_xyz[G.pepidx] = output[G.pepidx]
                output_dist_value = pad_xyz_by_idx_condition(output_xyz, G.batch, G.target_idx2, torch.sqrt(snr_weight))

                loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * snr_weight[pepidx])
                loss2 = F.mse_loss(original_dist_value, output_dist_value)

                loss = loss1 + loss2

        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["loss1"].append(loss1.cpu().detach().numpy())
        temp_loss["loss2"].append(loss2.cpu().detach().numpy())

    return temp_loss


def save_model(epoch:int, model, optimizer, train_loss, valid_loss, args) -> None:
    "
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


def pad_xyz_by_idx_condition(xyz, G_batch, target_idx2, weights):

    weighted_xyz = xyz*weights

    graph_xyzs = []
    max_nodes = 0
    target_masks = []
    total_masks = []

    for g in G_batch.unique().tolist():
        node_mask = (G_batch == g)
        
        this_xyz = weighted_xyz[node_mask]
        graph_xyzs.append(this_xyz)
        max_nodes = max(max_nodes, this_xyz.shape[0])

        target_mask = torch.zeros(this_xyz.shape[0], dtype=torch.bool)
        target_mask[target_idx2[g]] = 1
        target_masks.append(target_mask)

        total_mask = torch.ones(this_xyz.shape[0], dtype=torch.bool)
        total_masks.append(total_mask)

    # padding
    padded = [
        F.pad(x, (0, 0, 0, max_nodes - x.shape[0]), value=0.0)
        for x in graph_xyzs
    ]
    padded_target_mask = [
        F.pad(x, (0, max_nodes - x.shape[0]), value=0)
        for x in target_masks
    ]
    padded_total_mask = [
        F.pad(x, (0, max_nodes - x.shape[0]), value=0)
        for x in total_masks
    ]

    if len(padded) == 0: value = torch.tensor(0.0)
    else:
        padded_xyz = torch.stack(padded)
        padded_target_mask = torch.stack(padded_target_mask)
        padded_total_mask = torch.stack(padded_total_mask)

        m1 = padded_target_mask.unsqueeze(2) | padded_target_mask.unsqueeze(1)
        m2 = padded_total_mask.unsqueeze(2) & padded_total_mask.unsqueeze(1)
        mask = m1 & m2

        distance_map = torch.cdist(padded_xyz, padded_xyz)
        value = distance_map[mask]

    return value

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
