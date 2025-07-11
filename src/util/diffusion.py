import torch
from util.utils import *
from torch import Tensor
from typing import Tuple, List
import torch.nn.functional as F
import tqdm
import numpy as np

class Diffusion:

    def __init__(self, device:str, T:int, G) -> None:
        """
        initializing Diffusion

        Args:
            device (str): device
            T (int): max timestep
            G (graph): graph
        """
        self.device = device
        self.T = T
        self.graph = G.to(device)
        self.pepidx = G.pepidx
        # self.target_idx = G.target_idx

        self.betas = cos_scheduling(T)
        self.alphas = 1 - self.betas
        self.alpha_bars_xyz = torch.cumprod(1-self.betas, dim=0).to(device)

    def time_embedding(self, N:int, idx) -> Tuple[Tensor, Tensor]:
        """
        node_attr에 추가할 time embedding을 원하는 차원 크기(N)에 맞게 만드는 함수

        Args:
            N (int): time embedding dimension
            idx (None or int): None for training, int for sampling
        """
        pe = positional_encoding(self.T,N)

        if idx == None:
            B = self.graph.ptr.shape[0] - 1
            
            batch_resnum = [(self.graph.ptr[i] - self.graph.ptr[i - 1]) for i in range(1, B + 1)]

            # t = 1.0 - torch.sigmoid(torch.randn(B, device=self.device) * 1.2 - 1.2)
            # idx = (t * self.T).long().clamp(max=len(self.alpha_bars_xyz) - 1)
            idx = torch.randint(0, len(self.alpha_bars_xyz), (B,), device=self.device)

            # residue 수에 맞춰 index 확장
            idxs = torch.cat([i.repeat(n) for i, n in zip(idx, batch_resnum)]).tolist()

            used_alpha_bars_xyz = self.alpha_bars_xyz[idxs].unsqueeze(1)
            base = pe[idxs].to(self.device)
            
            return base, used_alpha_bars_xyz, idxs
        
        else:
            base = pe[idx].to(self.device)
            base = torch.tile(base, (self.graph.node_xyz.shape[0],1))
            
            return base

        
    def forward_process(self, used_alpha_bars_xyz:Tensor, idx:List) -> Tuple[Tensor, Tensor]:
        """
        noise를 scheduling에 맞게 더하는 함수 (batch 마다 다르게)

        Args:
            x (Tensor): noise가 추가되기 전 원본
            used_alpha_bars (Tensor): idx에 맞는 alpha bar
        """

        G = self.graph 
        x = G.node_xyz[idx]

        batch = G.batch[idx] 

        # noise 부분
        epsilon1 = torch.randn_like(x)
        com = torch.zeros(batch.max()+1, 3, device=epsilon1.device)
        com = com.index_add(0, batch, epsilon1) / torch.bincount(batch).unsqueeze(1)
        epsilon1 = epsilon1 - com[batch]

        # used_alpha_bars_xyz = used_alpha_bars_xyz[G.target_idx]

        # noise낀 x
        x_tilde = torch.sqrt(used_alpha_bars_xyz[G.pepidx])*x + torch.sqrt(1-used_alpha_bars_xyz[G.pepidx])*epsilon1
        
        return x_tilde, used_alpha_bars_xyz, epsilon1
    
    def forward_process_ep(self, used_alpha_bars_xyz, idx): 

        G = self.graph 
        x = G.node_xyz

        epsilon = torch.randn_like(x)
        epsilon -= torch.mean(epsilon[idx], dim=0, keepdim=True)

        x_tilde = torch.sqrt(used_alpha_bars_xyz)*x + torch.sqrt(1-used_alpha_bars_xyz)*epsilon
        x_tilde -= torch.mean(x_tilde[idx], dim=0, keepdim=True) 

        return x_tilde[idx], epsilon[idx]

    
    def reverse_process_on_x(self, x:Tensor, x0:Tensor, t:int) -> Tensor:
        """
        noise가 낀 x에서 noise를 한 step 벗겨내는 함수

        Args:
            x (Tensor): 노이즈가 낀 xyz coordinate
            pred_ep (Tensor): model(EGNN)으로 예측한 x에서의 noise
        """

        noise1 = torch.randn_like(x)
        noise1 = noise1 - torch.mean(noise1, dim=0, keepdim=True)

        if t != 0: 
            x_t1 = x0*torch.sqrt(self.alpha_bars_xyz[t-1]) + noise1*torch.sqrt(1-self.alpha_bars_xyz[t-1])

        else: 
            x_t1 = x0*torch.sqrt(self.alpha_bars_xyz[0]) + noise1*torch.sqrt(1-self.alpha_bars_xyz[0])

        return x_t1

        # beta_t = self.betas[t]
        # alpha_t = self.alphas[t]
        # alpha_bar_t = self.alpha_bars_xyz[t]
        # alpha_bar_tm1 = self.alpha_bars_xyz[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)

        # coef_x0 = (torch.sqrt(alpha_bar_tm1) * beta_t) / (1 - alpha_bar_t)
        # coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_tm1)) / (1 - alpha_bar_t)

        # mean = coef_x0 * x0 + coef_xt * x

        # if t == 0:
        #     return mean  # deterministic
        # else:
        #     var = ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * beta_t
        #     noise = torch.randn_like(x)
        #     noise = noise - torch.mean(noise, dim=0)
        #     return mean + torch.sqrt(var) * noise
    
    def reverse_process_ep(self, x, pred_ep, t): 
        # pred_ep = pred_ep - torch.mean(pred_ep[self.pepidx], dim=0)

        noise = torch.randn_like(x)
        noise -= torch.mean(noise, dim=0, keepdim=True)

        mean = (x - (1-self.alphas[t])*(pred_ep[self.pepidx])/torch.sqrt(1-self.alpha_bars_xyz[t]))/torch.sqrt(self.alphas[t])
        if t == 0:
            var = torch.zeros_like(mean)
        else:
            var = (1-self.alpha_bars_xyz[t-1])*self.betas[t]/(1-self.alpha_bars_xyz[t])
        
        x_t1 = mean + torch.sqrt(var)*noise
        x_t1 -= torch.mean(x_t1,dim=0,keepdim=True)

        return x_t1

        # alpha_t = self.alphas[t]
        # beta_t = self.betas[t]
        # alpha_bar_t = self.alpha_bars_xyz[t]
        # alpha_bar_tm1 = self.alpha_bars_xyz[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)

        # # 1. x0 reconstruction
        # x0_hat = (x - torch.sqrt(1 - alpha_bar_t) * pred_ep[self.pepidx]) / torch.sqrt(alpha_bar_t)

        # # 2. posterior mean
        # coef1 = (torch.sqrt(alpha_bar_tm1) * beta_t) / (1 - alpha_bar_t)
        # coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_tm1)) / (1 - alpha_bar_t)
        # mean = coef1 * x0_hat + coef2 * x

        # # 3. variance
        # if t == 0:
        #     return mean  # deterministic at final step
        # else:
        #     var = ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * beta_t
        #     noise = torch.randn_like(x)
        #     noise = noise - torch.mean(noise, dim=0)
        #     return mean + torch.sqrt(var) * noise


        
    