import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple, Type
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch_geometric.data import Data


def load_model(
        model_class:Type[nn.Module], 
        model_params:Dict, 
        model_name:str, 
        model_path:str, 
        device:str, 
        lr:float           
    ) -> Tuple[nn.Module, torch.optim.Optimizer, int, Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Initialize or load a model and optimizer from a saved checkpoint if available.

    Args:
        model_class (Type[nn.Module]): Class of the model to be instantiated.
        model_params (Dict): Dictionary of arguments to initialize the model.
        model_name (str): Subdirectory name for the model checkpoint.
        model_path (str): Base directory path where the model is saved.
        device (str): Device for loading the model (e.g., 'cuda', 'cpu').
        lr (float): Learning rate for optimizer.

    Returns:
        Tuple[
            nn.Module: The model instance (loaded or freshly initialized),
            torch.optim.Optimizer: The AdamW optimizer,
            int: The starting epoch,
            dict: Training loss history,
            dict: Validation loss history
        ]
    """
    model = model_class( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    # 초기값 
    train_loss = {'total':[],'loss1':[],'loss2':[],'loss3':[], 'loss4':[]}
    valid_loss = {'total':[],'loss1':[],'loss2':[],'loss3':[], 'loss4':[]}
    epoch = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    if os.path.exists(model_path+model_name+'/model.pkl'): # if checkpoint exists == if model exists
        checkpoint = torch.load(model_path+model_name+'/model.pkl', map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        print("Restarting at epoch", epoch)
        
    else: # if checkpoint doesn't exist -> create new model
        if not os.path.exists(model_path+model_name):
            print("Creating a new dir at models/"+model_name)
            os.mkdir(model_path+model_name)
    
    return model, optimizer, epoch, train_loss, valid_loss


def run_an_epoch(
        model:nn.Module, 
        optimizer:torch.optim.Optimizer, 
        loader:data.DataLoader, 
        noiser:Type, 
        device:str, 
        timestep:int, 
        t_dim:int, 
        train:bool
    ) -> Dict[str, List[float]]:
    """
    Run one training or evaluation epoch over the given data loader.

    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer (used only if train=True).
        loader (DataLoader): Data loader providing batches of graphs.
        noiser (Type): The noise/diffusion scheduler class.
        device (str): The device to run computation on ('cuda' or 'cpu').
        timestep (int): Number of diffusion steps (T).
        t_dim (int): Time embedding dimension.
        train (bool): Whether to perform training (backprop) or evaluation.

    Returns:
        Dict[str, List[float]]: Dictionary of loss values accumulated over the epoch.
    """
    temp_loss = {'total':[], 'loss1':[], 'loss2':[]}
    model = model.to(device)
    
    for i,G in enumerate(loader):
        if train: optimizer.zero_grad()
        if not G or G == None: continue # Skip empty graphs

        G = G.to(device) 
        pepidx = G.pepidx.clone()   
        
        ddpm = noiser(device, timestep, G)
        t_embed, used_alpha_bars = ddpm.time_embedding(t_dim, None)
        G.node_attr = torch.cat([G.node_attr,t_embed],dim=1)

        xyz = G.node_xyz.clone()
        xyz_label = xyz[G.pepidx].clone() # Ground truth coordinates of peptide

        x_tilde = ddpm.forward_process(used_alpha_bars) # Diffuse peptide coordinates
        
        snr_weight = torch.clamp(used_alpha_bars / (1 - used_alpha_bars + 1e-7), max=5.0) 

        original_dist_value = pad_xyz_by_idx_condition(xyz, G.batch, G.pepidx2, torch.sqrt(snr_weight)) # Get reference distance map

        G.node_xyz[pepidx] = x_tilde # Update coords
        G = dist_edges(G)            # Update distance edge feature

        if train:
            _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr) # predicted x0 of peptide nodes

            output_xyz = xyz.clone() 
            output_xyz[G.pepidx] = output[G.pepidx]
            output_dist_value = pad_xyz_by_idx_condition(output_xyz, G.batch, G.pepidx2, torch.sqrt(snr_weight)) # Get predicted x0 distance map

            loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * snr_weight[pepidx]) # MSE Loss of position
            loss2 = F.mse_loss(original_dist_value, output_dist_value)*0.2 # MSE Loss of pair distance

            loss = loss1 + loss2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        else:
            with torch.no_grad():
                _,output = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)

                output_xyz = xyz.clone() 
                output_xyz[G.pepidx] = output[G.pepidx]
                output_dist_value = pad_xyz_by_idx_condition(output_xyz, G.batch, G.pepidx2, torch.sqrt(snr_weight))

                loss1 = torch.mean(F.mse_loss(output[pepidx], xyz_label, reduction='none') * snr_weight[pepidx])
                loss2 = F.mse_loss(original_dist_value, output_dist_value)*0.2

                loss = loss1 + loss2

        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["loss1"].append(loss1.cpu().detach().numpy())
        temp_loss["loss2"].append(loss2.cpu().detach().numpy())

    return temp_loss


def save_model(
        epoch:int, 
        model:nn.Module, 
        optimizer:torch.optim.Optimizer, 
        train_loss:Dict[str, List[float]], 
        valid_loss:Dict[str, List[float]], 
        model_path:str, 
        model_name:str
    ) -> None:
    """
    Save model and optimizer states, along with loss history.
    Also saves a separate 'best.pkl' if current validation loss is the lowest so far.
    """
    # Update the best model if necessary:
    if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
            }, model_path+'/'+model_name+"/best.pkl")
            
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss
        }, model_path+'/'+model_name+"/model.pkl")


def pad_xyz_by_idx_condition(
        xyz:Tensor, 
        G_batch:Tensor, 
        target_idx2:Tensor, 
        weights:Tensor
    ) -> Tensor:
    """
    Compute pairwise distances within each graph after padding, only between target-other nodes.

    Args:
        xyz (Tensor): Node coordinates of shape (N, 3).
        G_batch (Tensor): Graph assignment for each node (length N).
        target_idx2 (Tensor): List-like structure mapping graph index to peptide atom indices.
        weights (Tensor): Per-node scalar weight (length N), e.g. SNR-based.

    Returns:
        Tensor: Flattened 1D tensor of distances between (target,target) node pairs across all graphs.
    """
    weighted_xyz = xyz * weights # Apply per-node weights to coordinates

    graph_xyzs = []
    max_nodes = 0
    target_masks = [] # Mask of target atoms
    total_masks = []  # Mask of batch atoms

    for g in G_batch.unique().tolist(): # Loop over each graph in batch
        node_mask = (G_batch == g) # Get mask of nodes belonging to graph g
        
        this_xyz = weighted_xyz[node_mask] # Get coords of nodes in this graph
        graph_xyzs.append(this_xyz)
        max_nodes = max(max_nodes, this_xyz.shape[0])

        target_mask = torch.zeros(this_xyz.shape[0], dtype=torch.bool)
        target_mask[target_idx2[g]] = 1 # Mark peptide nodes for graph g
        target_masks.append(target_mask)

        total_mask = torch.ones(this_xyz.shape[0], dtype=torch.bool) # Mark this graph nodes
        total_masks.append(total_mask)

    # pad coords and masks to max_nodes
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
        padded_xyz = torch.stack(padded)                     # Shape: (B, max_nodes, 3)
        padded_target_mask = torch.stack(padded_target_mask) # Shape: (B, max_nodes)
        padded_total_mask = torch.stack(padded_total_mask)   # Shape: (B, max_nodes)

        m1 = padded_target_mask.unsqueeze(2) | padded_target_mask.unsqueeze(1) # if at least one of the two nodes belongs to target_idx
        m2 = padded_total_mask.unsqueeze(2) & padded_total_mask.unsqueeze(1)   # both must be belongs to same graph
        mask = m1 & m2

        distance_map = torch.cdist(padded_xyz, padded_xyz)
        value = distance_map[mask]

    return value

def dist_edges(G:Data) -> Data:
    """
    Append Euclidean distance between connected node pairs as an edge feature.

    Args:
        G (Data): PyTorch Geometric Data object with:

    Returns:
        Data: Updated graph with distance added as the first edge feature.
    """
    xyz = G.node_xyz
    edge_index = G.edge_index 
    u,v = edge_index 

    dist_map = torch.sqrt(torch.sum((xyz[None,:,:] - xyz[:,None,:])**2, dim=2))
    diff_xyz = dist_map[u,v].unsqueeze(1)

    edge_attr = torch.cat([diff_xyz,G.edge_attr], dim=1) # (E, F) -> (E, F+1)
    G.edge_attr = edge_attr

    return G
