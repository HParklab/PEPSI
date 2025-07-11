from abc import ABC

import numpy as np
import torch
from torch import nn, Tensor


def stack_batch_dim(data: Tensor) -> Tensor:
    """data: [B, N, C] -> [B * N, C]"""
    return data.reshape(-1, data.size()[-1])


def scatter_batch_dim(data: Tensor, b_dim: int) -> Tensor:
    """data: [B * N, C] -> [B, N, C]"""
    return data.reshape(b_dim, -1, data.size()[-1])


def embed_rbf(
    dist: Tensor, dist_min: float = 0.0, dist_max: float = 15.0, dist_count: int = 16
) -> Tensor:
    """Computes a Radial Basis Function (RBF) embedding of a distance tensor.
    (From https://github.com/jingraham/neurips19-graph-protein-design)

    Parameters
    ----------
    dist : torch.Tensor
        Input tensor containing distance values.
    dist_min : float, optional
        Minimum value for the RBF centers.
    dist_max : float, optional
        Maximum value for the RBF centers.
    dist_count : int, optional
        Number of RBF basis functions.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(..., dist_count)`, where each original distance
        is transformed into an RBF embedding along the last dimension.

    """
    device = dist.device
    D_mu = torch.linspace(dist_min, dist_max, dist_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (dist_max - dist_min) / dist_count

    RBF = torch.exp(-(((dist - D_mu) / D_sigma) ** 2))
    return RBF


class AbstractFourierEmbedding(ABC, nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        self.n_freq = num_frequencies
        self.dim = num_frequencies * 2


class BasicFourierEmbedding(AbstractFourierEmbedding):
    def __init__(self, num_frequencies: int = 12, div_factor: int = 8):
        super().__init__(num_frequencies)
        self.div_factor = div_factor

    def forward(self, x: Tensor):
        device = x.device
        scales = (
            1.5 ** torch.arange(self.n_freq, device=device, dtype=x.dtype)
            / self.div_factor
        )
        emb = 2 * np.pi * x / scales
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GaussianFourierEmbedding(AbstractFourierEmbedding):
    def __init__(self, num_frequencies: int = 64, gauss_scale: float = 1.0):
        super().__init__(num_frequencies)
        b_gauss = torch.randn(1, num_frequencies, dtype=torch.float32) * gauss_scale
        self.register_buffer("b_gauss", b_gauss)

    def forward(self, x: Tensor):
        emb = (2 * np.pi * x).matmul(self.b_gauss)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
