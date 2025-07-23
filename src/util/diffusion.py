import torch
from torch import Tensor
from typing import Tuple, List
from torch_geometric.data import Data
import numpy as np
import math

class Diffusion:

    def __init__(self, device:str, T:int, G:Data) -> None:
        
        self.device = device
        self.T = T
        self.G = G.to(device)

        betas = self.cos_scheduling(T)
        self.betas = betas
        self.alphas = 1-betas
        self.alpha_bars = torch.cumprod(1-betas, dim=0).to(device)

    def time_embedding(self, N:int, idx:int) -> Tuple[Tensor, Tensor] | Tensor:
        """
        Appending the timestep to the node features so that the model can be aware of the diffusion step.

        Args:
            N (int): timestep embedding dimension
            idx (None or int): None for training, int for sampling => time step of sample
        """
        pe = self.positional_encoding(self.T,N)

        if idx == None:
            B = self.G.ptr.shape[0] - 1
            
            batch_resnum = [(self.G.ptr[i] - self.G.ptr[i - 1]) for i in range(1, B + 1)]
            idx = torch.randint(0, len(self.alpha_bars), (B,), device=self.device)
            idxs = torch.cat([i.repeat(n) for i, n in zip(idx, batch_resnum)]).tolist()

            used_alpha_bars = self.alpha_bars[idxs].unsqueeze(1)
            t_embed = pe[idxs].to(self.device)
            
            return t_embed, used_alpha_bars
        
        else:
            t_embed = pe[idx].to(self.device)
            t_embed = torch.tile(t_embed, (self.G.node_xyz.shape[0],1))
            
            return t_embed
        
        
    def forward_process(self, used_alpha_bars:Tensor) -> Tensor:

        x = self.G.node_xyz[self.G.pepidx]

        batch = self.G.batch[self.G.pepidx] 

        epsilon = torch.randn_like(x)
        com = torch.zeros(batch.max()+1, 3, device=epsilon.device)
        com = com.index_add(0, batch, epsilon) / torch.bincount(batch).unsqueeze(1)
        epsilon = epsilon - com[batch]

        x_tilde = torch.sqrt(used_alpha_bars[self.G.pepidx])*x + torch.sqrt(1-used_alpha_bars[self.G.pepidx])*epsilon
        
        return x_tilde
    
    def reverse_process(self, x:Tensor, x0:Tensor, t:int) -> Tensor:

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_tm1 = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)

        coef_x0 = (torch.sqrt(alpha_bar_tm1) * beta_t) / (1 - alpha_bar_t)
        coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_tm1)) / (1 - alpha_bar_t)

        mean = coef_x0 * x0 + coef_xt * x

        if t == 0:
            return mean  # deterministic
        else:
            var = ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * beta_t
            noise = torch.randn_like(x)
            noise = noise - torch.mean(noise, dim=0)
            return mean + torch.sqrt(var) * noise

    def positional_encoding(self, seq_len:int, d_model:int) -> Tensor:

        position = np.arange(seq_len)[:, np.newaxis]  
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)  
        pe[:, 1::2] = np.cos(position * div_term)  
        pe = torch.tensor(pe, dtype=torch.float)
        
        return pe
    
    
    def cos_scheduling(self, timesteps:int, s:float=0.008) -> Tensor:

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        return torch.clip(betas, 0.0001, 0.9999)
    