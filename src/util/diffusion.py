import torch
from torch import Tensor
from typing import Tuple
from torch_geometric.data import Data
import numpy as np
import math

class Diffusion:

    def __init__(self, device:str, T:int, G:Data) -> None:
        """
        Initialize the diffusion process and its scheduling parameters.

        Args:
            device (str): Device identifier string (e.g., 'cuda' or 'cpu') to move data and tensors to.
            T (int): Total number of diffusion time steps.
            G (Data): A PyTorch Geometric Data object representing the initial graph.

        Attributes:
            betas (Tensor): Noise schedule (βₜ) of shape (T,), computed from a cosine schedule.
            alphas (Tensor): Values of αₜ = 1 - βₜ at each time step.
            alpha_bars (Tensor): Cumulative product ∏ₜ αₜ used for closed-form posterior and sampling equations.
            G (Data): The input graph moved to the specified device.
        """
        self.device = device
        self.T = T
        self.G = G.to(device)

        def cos_scheduling(timesteps:int, s:float=0.008) -> Tensor:
            """
            Compute the beta schedule using a cosine-based cumulative ᾱₜ schedule.

            Args:
                timesteps (int): Number of total diffusion steps.
                s (float, optional): Small offset to prevent singularity at t=0. Default is 0.008.

            Returns:
                Tensor: A tensor of shape (T,) containing βₜ values for each time step.
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            
            return torch.clip(betas, 0.0001, 0.9999)

        betas = cos_scheduling(T)
        self.betas = betas
        self.alpha_bars = torch.cumprod(1-betas, dim=0).to(device)

    def time_embedding(self, N:int, idx:int) -> Tuple[Tensor, Tensor] | Tensor:
        """
        Generate sinusoidal time-step embeddings to inject diffusion timestep awareness into node features.

        Args:
            N (int): The dimension of the time embedding.
            idx (int | None): 
                - If None: Used during training; randomly sample a time-step per graph in batch.
                - If int: Used during inference/sampling; use the specific time-step index.

        Returns:
            Union[Tuple[Tensor, Tensor], Tensor]:
                - If training (idx is None):
                    - t_embed (Tensor): Time embeddings for all nodes. Shape: (num_nodes, N)
                    - used_alpha_bars (Tensor): ᾱₜ for each node. Shape: (num_nodes, 1)
                - If sampling (idx is int):
                    - t_embed (Tensor): Tiled time embedding for all nodes. Shape: (num_nodes, N)
        """ 
        def positional_encoding(seq_len:int, d_model:int) -> Tensor:
            """
            Generate standard sinusoidal positional encoding (as in Vaswani et al., 2017).

            Args:
                seq_len (int): Total time steps (T).
                d_model (int): Embedding dimension (N).
            
            Returns:
                Tensor: A tensor of shape (seq_len, d_model) containing sinusoidal encodings.
            """
            position = np.arange(seq_len)[:, np.newaxis] # (T, 1)
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  
            
            pe = np.zeros((seq_len, d_model))
            pe[:, 0::2] = np.sin(position * div_term)  
            pe[:, 1::2] = np.cos(position * div_term)  
            pe = torch.tensor(pe, dtype=torch.float)
    
            return pe
        
        pe = positional_encoding(self.T,N) # (T, N)

        if idx == None: # Training Mode
            B = self.G.ptr.shape[0] - 1 # Number of graphs in batch
            batch_resnum = [(self.G.ptr[i] - self.G.ptr[i - 1]) for i in range(1, B + 1)] # nodes per graph

            idx = torch.randint(0, len(self.alpha_bars), (B,), device=self.device)  # Random timestep per graph
            idxs = torch.cat([i.repeat(n) for i, n in zip(idx, batch_resnum)]).tolist() # Broadcast to nodes

            used_alpha_bars = self.alpha_bars[idxs].unsqueeze(1) # Shape: (num_nodes, 1)
            t_embed = pe[idxs].to(self.device)                   # Shape: (num_nodes, N)
            
            return t_embed, used_alpha_bars
        
        else: # Sampling Mode
            t_embed = pe[idx].to(self.device)
            t_embed = torch.tile(t_embed, (self.G.node_xyz.shape[0],1)) # Shape: (num_nodes, N)
            
            return t_embed
        
        
    def forward_process(self, used_alpha_bars:Tensor) -> Tensor:
        """
        Forward (noising) step of the diffusion process.

        Adds noise to the peptide coordinates according to the provided ᾱₜ (used_alpha_bars).
        The noise is zero-centered **per sample** to ensure translation invariance.

        Args:
            used_alpha_bars (Tensor): A tensor of shape (num_nodes, 1), representing ᾱₜ for each node 
                                    in the full graph (used to control the noise strength).

        Returns:
            Tensor: Noised coordinates x̃ₜ of shape (num_pep_nodes, 3).
        """
        # Extract coordinates and batch information for peptide atoms only
        x = self.G.node_xyz[self.G.pepidx]              # Shape: (num_pep_nodes, 3)
        batch = self.G.batch[self.G.pepidx]             # Shape: (num_pep_nodes,)


        # Generate isotropic Gaussian noise
        epsilon = torch.randn_like(x)                   # ε ~ N(0, I)

        # Center the noise per graph to ensure translation invariance
        com = torch.zeros(batch.max() + 1, 3, device=epsilon.device)  # (num_graphs, 3)
        com = com.index_add(0, batch, epsilon) / torch.bincount(batch).unsqueeze(1)
        epsilon = epsilon - com[batch]  # Subtract center-of-mass per graph

        # Forward diffusion step (q(x_t | x_0))
        x_tilde = torch.sqrt(used_alpha_bars[self.G.pepidx])*x + torch.sqrt(1-used_alpha_bars[self.G.pepidx])*epsilon
        
        return x_tilde # Shape: (num_pep_nodes, 3)
    
    def reverse_process(self, x:Tensor, x0:Tensor, t:int) -> Tensor:
        """
        Perform one step of the reverse diffusion process.

        This function approximates the distribution q(x_{t-1} | x_t, x_0) using a closed-form
        posterior, and samples from it if t > 0. At t = 0, the process becomes deterministic.

        Args:
            x (Tensor): Current noised coordinates x_t at timestep t.
            x0 (Tensor): Predicted original coordinates x_0.
            t (int): The current timestep (0 ≤ t < T).

        Returns:
            Tensor: The denoised coordinates x_{t-1} (or the mean, if t == 0).
        """
        # Get diffusion parameters at timestep t and t-1
        beta_t = self.betas[t]                        # βₜ
        alpha_t = 1 - beta_t                          # αₜ = 1 - βₜ
        alpha_bar_t = self.alpha_bars[t]              # ᾱₜ = ∏_{s=1}^t αₛ
        alpha_bar_tm1 = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)

        # Coefficients from the posterior q(x_{t-1} | x_t, x_0)
        coef_x0 = (torch.sqrt(alpha_bar_tm1) * beta_t) / (1 - alpha_bar_t)
        coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_tm1)) / (1 - alpha_bar_t)

        # Compute the posterior mean
        mean = coef_x0 * x0 + coef_xt * x

        if t == 0: # At timestep 0, no sampling is needed — just return the mean
            return mean 
        else:
            # Sample from the posterior with computed variance
            var = ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * beta_t
            noise = torch.randn_like(x)
            noise = noise - torch.mean(noise, dim=0)
            return mean + torch.sqrt(var) * noise
    
    
    