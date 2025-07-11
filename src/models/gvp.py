"""
Some of the code is imported from DiffHopp.
https://github.com/jostorge/diffusion-hopping

MIT License

Copyright (c) 2022 Jos Torge, Charles Harris, Simon Mathis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC
import math
from functools import partial
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .egnn import EquivariantUpdate, coord2diff, coord2cross
from .common import (
    AbstractFourierEmbedding,
    BasicFourierEmbedding,
    GaussianFourierEmbedding,
    embed_rbf,
)

from typing import Optional


def coord2feat(
    coords: Tensor, edge_index: Tensor, eps: float = 1e-4
) -> tuple[Tensor, Tensor]:
    V = coords[edge_index[0]] - coords[edge_index[1]]  # [n_edges, 3]
    s = torch.linalg.norm(V, dim=-1, keepdim=True)  # [n_edges, 1]
    V = (V / torch.clip(s, min=eps))[..., None, :]  # [n_edges, 1, 3]
    return s, V


def tuple_sum(*args) -> tuple[int | float, int | float]:
    """Compute the element-wise sum of multiple tuples.

    Parameters
    ----------
    *args : tuple of numeric values
        Any number of tuples of the form `(s, V)`, where each tuple consists of
        numerical values.

    Returns
    -------
    tuple
        A tuple containing the element-wise sum of the input tuples.

    Examples
    --------
    >>> tuple_sum((1, 2), (3, 4), (5, 6))
    (9, 12)
    """
    return tuple(map(sum, zip(*args)))


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def cycle_counts(adj):
    assert (adj.diag() == 0).all()
    assert (adj == adj.T).all()

    A = adj.float()
    d = A.sum(dim=-1)

    # Compute powers
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    A5 = A4 @ A

    x3 = A3.diag() / 2
    x4 = (A4.diag() - d * (d - 1) - A @ d) / 2

    # Triangle count matrix (indicates for each node i how many triangles it shares with node j)
    T = adj * A2
    x5 = (A5.diag() - 2 * T @ d - 4 * d * x3 - 2 * A @ x3 + 10 * x3) / 2

    return torch.stack([x3, x4, x5], dim=-1)


def normalize_vector(tensor: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Normalizes a tensor along the specified dimension while avoiding NaNs.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be normalized.
    dim : int, optional
        The dimension along which to normalize. Default is `-1` (last dimension).
    eps : float, optional
        A small value added to the denominator to prevent division by zero.
        Default is `1e-8`.

    Returns
    -------
    torch.Tensor
        The normalized tensor with the same shape as the input.

    Examples
    --------
    >>> x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> _normalize(x, dim=0)
    tensor([[0.2425, 0.3714, 0.4472],
            [0.9701, 0.9285, 0.8944]])
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True) + eps)
    )


class GVPDropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.s_dropout = nn.Dropout(drop_rate)
        self.v_dropout = nn.Dropout1d(drop_rate)

    def forward(
        self, x: tuple[Tensor, Tensor] | Tensor
    ) -> tuple[Tensor, Tensor] | Tensor:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if isinstance(x, Tensor):
            return self.s_dropout(x)

        s, v = x
        return self.s_dropout(s), self.v_dropout(v)


class VectorLayerNorm(nn.Module):
    """
    Equivariant normalization of vector-valued features inspired by:
    Liao, Yi-Lun, and Tess Smidt.
    "Equiformer: Equivariant graph attention transformer for 3d atomistic graphs."
    arXiv preprint arXiv:2206.11990 (2022).
    Section 4.1, "Layer Normalization"
    """

    def __init__(
        self, n_channels: int, learnable_weight: bool = True, eps: float = 1e-5
    ):
        super().__init__()
        self.gamma = (
            nn.Parameter(torch.ones(1, n_channels, 1)) if learnable_weight else None
        )  # (1, c, 1)
        self.eps = math.sqrt(eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes LN(x) = ( x / RMS( L2-norm(x) ) ) * gamma
        :param x: input tensor (n, c, 3)
        :return: layer normalized vector feature
        """
        norm2 = _norm_no_nan(x, axis=-1, keepdims=True, sqrt=False)  # (n, c, 1)
        rms = torch.sqrt(torch.mean(norm2, dim=-2, keepdim=True))  # (n, 1, 1)
        x = torch.clip(x / rms, self.eps)  # (n, c, 3)
        if self.gamma is not None:
            x = x * self.gamma
        return x


class GVPLayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(
        self,
        dims: tuple[int, int],
        learnable_vector_weight: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.s, self.v = dims
        self.eps = math.sqrt(eps)
        self.scalar_norm = nn.LayerNorm(self.s, eps=eps)
        self.vector_norm = (
            VectorLayerNorm(self.v, learnable_vector_weight, eps)
            if self.v > 0
            else None
        )

    def forward(
        self, x: tuple[Tensor, Tensor] | Tensor
    ) -> tuple[Tensor, Tensor] | Tensor:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        return self.scalar_norm(s), self.vector_norm(v)


class GVP(nn.Module):
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        in_scalar, in_vector = in_dims
        out_scalar, out_vector = out_dims
        self.sigma, self.sigma_plus = activations

        if self.sigma is None:
            self.sigma = nn.Identity()
        if self.sigma_plus is None:
            self.sigma_plus = nn.Identity()

        self.h = max(in_vector, out_vector)
        self.W_h = nn.Parameter(torch.empty((self.h, in_vector)))
        if out_vector > 0:
            self.W_mu = nn.Parameter(torch.empty((out_vector, self.h)))

        self.W_m = nn.Linear(self.h + in_scalar, out_scalar)
        self.v = in_vector
        self.mu = out_vector
        self.n = in_scalar
        self.m = out_scalar
        self.vector_gate = vector_gate

        if vector_gate:
            self.sigma_g = nn.Sigmoid()
            self.W_g = nn.Linear(out_scalar, out_vector)

        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_h, a=math.sqrt(5))
        if self.mu > 0:
            torch.nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        self.W_m.reset_parameters()
        if self.vector_gate:
            self.W_g.reset_parameters()

    def forward(
        self, x: torch.Tensor | tuple[Tensor, Tensor]
    ) -> torch.Tensor | tuple[Tensor, Tensor]:
        """Geometric vector perceptron"""
        s, V = (
            x if self.v > 0 else (x, torch.empty((x.shape[0], 0, 3), device=x.device))
        )

        assert s.shape[-1] == self.n, (
            f"{s.shape[-1]} != {self.n} Scalar dimension mismatch"
        )
        assert V.shape[-2] == self.v, (
            f" {V.shape[-2]} != {self.v} Vector dimension mismatch"
        )
        assert V.shape[0] == s.shape[0], "Batch size mismatch"

        V_h = self.W_h @ V
        s_h = torch.clip(torch.norm(V_h, dim=-1), min=self.eps)
        s_hn = torch.cat([s, s_h], dim=-1)
        s_m = self.W_m(s_hn)
        s_dash = self.sigma(s_m)
        if self.mu > 0:
            V_mu = self.W_mu @ V_h
            if self.vector_gate:
                V_dash = self.sigma_g(self.W_g(self.sigma_plus(s_m)))[..., None] * V_mu
            else:
                v_mu = torch.clip(torch.norm(V_mu, dim=-1, keepdim=True), min=self.eps)
                V_dash = self.sigma_plus(v_mu) * V_mu
            retval = s_dash, V_dash
        else:
            retval = s_dash
        return retval


class GVPMessagePassing(MessagePassing, ABC):
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        hidden_dims: Optional[tuple[int, int]] = None,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
    ):
        super().__init__(aggr)
        if hidden_dims is None:
            hidden_dims = out_dims

        in_scalar, in_vector = in_dims
        hidden_scalar, hidden_vector = hidden_dims

        edge_scalar, edge_vector = edge_dims

        self.out_scalar, self.out_vector = out_dims
        self.in_vector = in_vector
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.normalization_factor = normalization_factor

        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.edge_gvps = nn.Sequential(
            GVP_(
                (2 * in_scalar + edge_scalar, 2 * in_vector + edge_vector),
                hidden_dims,
            ),
            GVP_(hidden_dims, hidden_dims),
            GVP_(hidden_dims, out_dims, activations=(None, None)),
        )

        self.attention = attention
        if attention:
            self.attention_gvp = GVP_(
                out_dims, (1, 0), activations=(torch.sigmoid, None), vector_gate=False
            )

    def forward(
        self,
        x: tuple[Tensor, Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        s, V = x
        v_dim = V.shape[-1]
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return self.propagate(edge_index, s=s, V=V, edge_attr=edge_attr, v_dim=v_dim)

    def message(self, s_i, s_j, V_i, V_j, edge_attr, v_dim):
        V_i = V_i.view(*V_i.shape[:-1], self.in_vector, v_dim)
        V_j = V_j.view(*V_j.shape[:-1], self.in_vector, v_dim)
        edge_scalar, edge_vector = edge_attr

        s = torch.cat([s_i, s_j, edge_scalar], dim=-1)
        V = torch.cat([V_i, V_j, edge_vector], dim=-2)
        s, V = self.edge_gvps((s, V))

        if self.attention:
            att = self.attention_gvp((s, V))
            s, V = att * s, att[..., None] * V
        return self._combine(s, V)

    def update(self, aggr_out: Tensor) -> tuple[Tensor, Tensor]:
        s_aggr, V_aggr = self._split(aggr_out, self.out_scalar, self.out_vector)
        if self.aggr == "add" or self.aggr == "sum":
            s_aggr = s_aggr / self.normalization_factor
            V_aggr = V_aggr / self.normalization_factor
        return s_aggr, V_aggr

    @staticmethod
    def _combine(s, V) -> Tensor:
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return torch.cat([s, V], dim=-1)

    @staticmethod
    def _split(s_V: Tensor, scalar: int, vector: int) -> tuple[Tensor, Tensor]:
        s = s_V[..., :scalar]
        V = s_V[..., scalar:]
        V = V.view(*V.shape[:-1], vector, -1)
        return s, V

    def reset_parameters(self):
        for gvp in self.edge_gvps:
            gvp.reset_parameters()
        if self.attention:
            self.attention_gvp.reset_parameters()


class GVPConvLayer(GVPMessagePassing, ABC):
    def __init__(
        self,
        node_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        drop_rate: float = 0.0,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        residual: bool = True,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
    ):
        super().__init__(
            node_dims,
            node_dims,
            edge_dims,
            hidden_dims=node_dims,
            activations=activations,
            vector_gate=vector_gate,
            attention=attention,
            aggr=aggr,
            normalization_factor=normalization_factor,
        )
        self.residual = residual
        self.drop_rate = drop_rate
        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([GVPLayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([GVPDropout(drop_rate) for _ in range(2)])

        self.ff_func = nn.Sequential(
            GVP_(node_dims, node_dims),
            GVP_(node_dims, node_dims, activations=(None, None)),
        )
        self.residual = residual

    def forward(
        self,
        x: tuple[Tensor, Tensor] | torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        s, V = super().forward(x, edge_index, edge_attr)
        if self.residual:
            s, V = self.dropout[0]((s, V))
            s, V = x[0] + s, x[1] + V
            s, V = self.norm[0]((s, V))

        x = (s, V)
        s, V = self.ff_func(x)

        if self.residual:
            s, V = self.dropout[1]((s, V))
            s, V = s + x[0], V + x[1]
            s, V = self.norm[1]((s, V))

        return s, V


class GVPNetwork(nn.Module):
    def __init__(
        self,
        in_node_dims: tuple[int, int],
        in_edge_dims: tuple[int, int],
        hidden_dims: tuple[int, int],
        n_layers: int,
        out_node_dims: tuple[int, int] | None = None,
        dist_rbf_dim: int | None = None,
        attention: bool = False,
        normalization_factor: float = 100.0,
        aggr: str = "add",
        activations=(F.silu, None),
        vector_gate: bool = True,
        fourier_embedding: AbstractFourierEmbedding | None = None,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dist_rbf_dim = dist_rbf_dim
        self.fourier_embedding = fourier_embedding
        self.eps = eps

        if out_node_dims is None:
            out_node_dims = in_node_dims

        self.embedding_in = nn.Sequential(
            GVPLayerNorm(in_node_dims),
            GVP(
                in_node_dims,
                hidden_dims,
                activations=(None, None),
                vector_gate=vector_gate,
            ),
        )
        self.embedding_out = nn.Sequential(
            GVPLayerNorm(hidden_dims),
            GVP(
                hidden_dims,
                out_node_dims,
                activations=activations,
                vector_gate=vector_gate,
            ),
        )

        # Determine edge input dimensions
        in_edge_dims = (
            tuple_sum(in_edge_dims, (dist_rbf_dim, 1))
            if dist_rbf_dim
            else tuple_sum(in_edge_dims, (1, 1))
        )
        if fourier_embedding is not None:
            in_edge_dims = tuple_sum(in_edge_dims, (fourier_embedding.dim, 0))

        self.edge_embedding = nn.Sequential(
            # Plus 1: distance, direction vector
            GVPLayerNorm(in_edge_dims),
            GVP(
                in_edge_dims,
                (hidden_dims[0], 1),
                activations=(None, None),
                vector_gate=vector_gate,
            ),
        )

        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    hidden_dims,
                    (hidden_dims[0], 1),
                    activations=activations,
                    vector_gate=vector_gate,
                    residual=True,
                    attention=attention,
                    aggr=aggr,
                    normalization_factor=normalization_factor,
                )
                for _ in range(n_layers)
            ]
        )

    def get_edge_attr(
        self,
        pos: Tensor,
        edge_index: Tensor,
        scalar_edge_feats: Tensor | None = None,  # [N_edges, feat_dim]
        vector_edge_feats: Tensor | None = None,  # [N_edges, feat_dim, 3]
    ) -> tuple[Tensor, Tensor]:
        s, V = coord2feat(pos, edge_index, self.eps)
        dist = s.clone()

        if self.dist_rbf_dim is not None:
            s = embed_rbf(s, dist_count=self.dist_rbf_dim)

        if self.fourier_embedding is not None:
            s = torch.cat((s, self.fourier_embedding(dist)), dim=-1)

        if scalar_edge_feats is not None:
            s = torch.cat([s, scalar_edge_feats], dim=-1)

        if vector_edge_feats is not None:
            V = torch.cat([V, vector_edge_feats], dim=1)
        return s, V

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        scalar_edge_feats: Tensor | None = None,
        vector_edge_feats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        edge_attr = self.get_edge_attr(
            x, edge_index, scalar_edge_feats, vector_edge_feats
        )
        edge_attr = self.edge_embedding(edge_attr)

        # [N_atoms, input_dim] -> [N_atom, hidden_dim], [N_atoms, hidden_dim, 3]
        h = self.embedding_in(h)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        h, x = self.embedding_out(h)
        return h, x


class GVPFinePL(nn.Module):
    def __init__(
        self,
        fine_input_dim: int,
        coarse_input_dim: int,
        coord_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 7,
        edge_dist_inter: float = 6,
        dist_rbf_dim: int | None = None,
        time_rbf_dim: int | None = None,
        attention: bool = False,
        fourier_feat: str | None = None,
        fourier_params: dict[str, int] | None = None,
        normalization_factor: int = 100,
        aggregation_method: str = "sum",
        self_condition: bool = False,
    ):
        super().__init__()
        self.edge_dist_inter = edge_dist_inter
        self.self_condition = self_condition
        self.dist_rbf_dim = dist_rbf_dim
        self.time_rbf_dim = time_rbf_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(fine_input_dim, 2 * fine_input_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * fine_input_dim),
            nn.Linear(2 * fine_input_dim, hidden_dim),
        )

        self.coarse_encoder = nn.Sequential(
            nn.Linear(coarse_input_dim, 2 * coarse_input_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * coarse_input_dim),
            nn.Linear(2 * coarse_input_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * fine_input_dim),
            nn.SiLU(),
            nn.Linear(2 * fine_input_dim, fine_input_dim),
        )

        time_dim = time_rbf_dim if time_rbf_dim is not None else 1
        dynamics_node_nf = hidden_dim + time_dim

        if fourier_feat == "basic":
            fourier_embedding = (
                BasicFourierEmbedding(**fourier_params)
                if fourier_params is not None
                else BasicFourierEmbedding()
            )
        elif fourier_feat == "gaussian":
            fourier_embedding = (
                GaussianFourierEmbedding(**fourier_params)
                if fourier_params is not None
                else GaussianFourierEmbedding()
            )
        elif fourier_feat is None:
            fourier_embedding = None
        else:
            raise NotImplementedError(
                f"Invalid value for fourier_feat: {fourier_feat}. Expected 'basic' or 'gaussian' or 'None'."
            )

        # 1: extra or not, 3: distinguish new edge types
        edge_feat_s_dim = edge_feat_dim + 4
        edge_feat_v_dim = 0
        if self_condition:
            extra_s_dim = dist_rbf_dim if dist_rbf_dim is not None else 1
            edge_feat_s_dim += extra_s_dim
            edge_feat_v_dim += 1

        self.gvp_module = GVPNetwork(
            in_node_dims=(dynamics_node_nf, 0),
            in_edge_dims=(edge_feat_s_dim, edge_feat_v_dim),
            hidden_dims=(hidden_dim, hidden_dim),
            out_node_dims=(hidden_dim, 1),
            n_layers=n_layers,
            dist_rbf_dim=dist_rbf_dim,
            attention=attention,
            fourier_embedding=fourier_embedding,
            normalization_factor=normalization_factor,
            aggr=aggregation_method,
        )
        self.node_nf = dynamics_node_nf
        self.coord_dim = coord_dim

    def forward(
        self,
        xh_f: Tensor,  # [N_atom, 3 + fine_feat_dim]
        xh_c: Tensor,  # [N_frag, 3 + coarse_feat_dim]
        c_pos: Tensor,  # [N_atom, 3]
        t: Tensor,  # [B_in, 1]
        b_idx_f: Tensor,  # [N_atom]
        b_idx_c: Tensor,  # [N_frag]
        res_idx_f: Tensor,  # [N_atom]
        res_idx_c: Tensor,  # [N_frag]
        rep_mask: Tensor,  # [N_atom]
        edge_idx: Tensor,  # [N_edges, 2]
        edge_feat: Optional[Tensor] = None,  # [N_edges, edge_feat_dim]
        xh_f_sc: Optional[Tensor] = None,
    ):
        device = xh_f.device

        # Split into x (coordinates) and h (features)
        x_f = xh_f[:, : self.coord_dim].clone()
        h_f = xh_f[:, self.coord_dim :].clone()

        x_c = xh_c[:, : self.coord_dim].clone()
        h_c = xh_c[:, self.coord_dim :].clone()

        # Embed atom and coarse features in a shared space
        h_f = self.atom_encoder(h_f)
        h_c = self.coarse_encoder(h_c)

        x_cat = torch.cat((x_f, x_c))
        h_cat = torch.cat((h_f, h_c))
        b_idx = torch.cat((b_idx_f, b_idx_c))
        res_idx = torch.cat((res_idx_f, res_idx_c))
        rep_mask = torch.cat(
            (rep_mask, torch.ones(len(res_idx_c), device=device).type(torch.bool))
        )
        c_pos = torch.cat((c_pos, torch.zeros_like(x_c)))

        # Time embedding condition
        h_time = t[b_idx]
        if self.time_rbf_dim:
            h_time = embed_rbf(h_time, dist_max=1.0, dist_count=self.time_rbf_dim)
        h_cat = torch.cat([h_cat, h_time], dim=-1)

        # Construct edges
        if xh_f_sc is not None:
            x_sc = torch.cat([xh_f_sc[:, : self.coord_dim], x_c])
            edges, edge_feats_s, edge_feats_v = self.get_edges(
                b_idx, x_cat, res_idx, rep_mask, edge_idx, edge_feat, x_sc
            )
        else:
            edges, edge_feats_s, edge_feats_v = self.get_edges(
                b_idx, x_cat, res_idx, rep_mask, edge_idx, edge_feat
            )
        assert torch.all(b_idx[edges[0]] == b_idx[edges[1]])
        # The first edge feature indicates whether it is a covalent bond or not
        matches = edges.T[:, None, :] == edge_idx[None, :, :]
        covalent_mask = torch.any(torch.all(matches, dim=-1), dim=1)
        assert edge_feats_s[covalent_mask, 0].sum() == len(edge_idx)

        h_final, x_final = self.gvp_module(
            h=h_cat,
            x=x_cat,
            edge_index=edges,
            scalar_edge_feats=edge_feats_s,
            vector_edge_feats=edge_feats_v,
        )
        vel = x_final.squeeze(dim=1) + c_pos

        # Decode atom features
        h_final_atoms = self.decoder(h_final)

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                print("Training: NaN detected in EGNN output. Velocity is set to zero!")
            else:
                raise ValueError("NaN detected in EGNN output")

        return torch.cat([vel, h_final_atoms], dim=-1)

    def get_edges(
        self,
        batch_idx: Tensor,
        pos: Tensor,
        res_idx: Tensor,
        rep_mask: Tensor,
        edge_idx: Tensor,
        edge_feats: Optional[Tensor] = None,
        pos_sc: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Constructs edge indices and features by concatenating the given edge
        index with fully-connected edges within the same residue and between residues
        within a threshold distance.

        Parameters
        ----------
        batch_idx : Tensor
            A tensor indicating the batch index for each node.
        pos : Tensor
            A tensor containing the positions of the nodes.
        res_idx : Tensor
            A tensor indicating the residue index for each node.
        edge_idx : Tensor
            A tensor containing the original edge indices.
        edge_feats : Optional[Tensor]
            A tensor containing the original edge features.

        Returns
        -------
        tuple[Tensor, Tensor]
            - new_edge_idx : Tensor
                The final edge index tensor after merging and deduplication.
            - new_edge_feats : Tensor
                The final edge feature tensor after concatenation and indexing.
        """

        def edge_onehot(adj: Tensor, class_idx: int, num_classes: int):
            class_idx = torch.tensor([class_idx], device=adj.device)
            onehot = torch.nn.functional.one_hot(class_idx, num_classes)
            return onehot.repeat(adj.numel(), 1).reshape(*adj.shape, -1)

        device = batch_idx.device

        # Fully-connected within the same residue index (adj_intra_res)
        # Inter-residue edges are formed according to edge_dist_inter
        # Representive nodes are fully-connected
        adj_b = batch_idx[:, None] == batch_idx[None, :]
        adj_intra_res = adj_b * (res_idx[:, None] == res_idx[None, :])
        adj_inter = adj_b * (
            (res_idx[:, None] != res_idx[None, :])
            & (torch.cdist(pos, pos) < self.edge_dist_inter)
        )
        adj_rep = adj_b * (rep_mask[:, None] * rep_mask[None, :])
        adj = adj_intra_res + adj_inter + adj_rep
        extra_edges = torch.stack(torch.where(adj), dim=0)
        n_extra_edges = extra_edges.shape[1]

        # NOTE: The concatenated edges and features are then indexed with
        # `first_indicies` to remove duplicates. Order: (given + new)[indexing]
        # Edge indices
        all_edges = torch.cat([edge_idx.T, extra_edges], dim=-1)
        _, idx, counts = torch.unique(
            all_edges, dim=1, sorted=True, return_inverse=True, return_counts=True
        )
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        new_edge_idx = all_edges[:, first_indicies]

        # Edge features
        new_edge_feats = torch.ones(len(edge_idx), device=device).reshape(-1, 1)
        new_edge_feats = torch.cat(
            [
                new_edge_feats,
                torch.zeros(n_extra_edges, device=device).reshape(-1, 1),
            ],
        )

        # To distinguish edge types for new edges (intra-residue, inter-residue, inter-rep)
        intra_res = edge_onehot(adj_intra_res, class_idx=0, num_classes=3)
        inter_res = edge_onehot(adj_inter, class_idx=1, num_classes=3)
        inter_rep = edge_onehot(adj_rep, class_idx=2, num_classes=3)
        res_e_feat = torch.zeros_like(intra_res, device=device)
        res_e_feat[torch.where(adj_intra_res)] = intra_res[torch.where(adj_intra_res)]
        res_e_feat[torch.where(adj_inter)] = inter_res[torch.where(adj_inter)]
        res_e_feat[torch.where(adj_rep)] = inter_rep[torch.where(adj_rep)]
        res_e_feat = torch.cat(
            [
                torch.zeros(len(edge_idx), res_e_feat.shape[-1], device=device),
                res_e_feat[torch.where(adj)],
            ]
        )
        new_edge_feats = torch.cat([new_edge_feats, res_e_feat], dim=-1)

        if edge_feats is not None:
            add_feat = torch.cat(
                [
                    edge_feats,
                    torch.zeros((n_extra_edges, edge_feats.shape[1]), device=device),
                ],
                dim=0,
            )
            new_edge_feats = torch.cat([new_edge_feats, add_feat], dim=-1)
        new_edge_feats = new_edge_feats[first_indicies, :]

        scalar_sc, vector_sc = None, None
        if self.self_condition:
            if pos_sc is not None:
                scalar_sc, vector_sc = coord2feat(pos_sc, new_edge_idx)
                if self.dist_rbf_dim is not None:
                    scalar_sc = embed_rbf(scalar_sc, dist_count=self.dist_rbf_dim)
            else:
                s_dim = self.dist_rbf_dim if self.dist_rbf_dim is not None else 1
                scalar_sc = torch.zeros(len(new_edge_feats), s_dim, device=device)
                vector_sc = torch.zeros(len(new_edge_feats), 1, 3, device=device)

        new_s_edge_feats = (
            torch.cat([new_edge_feats, scalar_sc], dim=-1)
            if scalar_sc is not None
            else new_edge_feats
        )
        new_v_edge_feats = vector_sc

        return new_edge_idx, new_s_edge_feats, new_v_edge_feats

    @property
    def name(self) -> str:
        return "gvp"


class GVPHybridFinePL(nn.Module):
    def __init__(
        self,
        fine_input_dim: int,
        coarse_input_dim: int,
        coord_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 6,
        edge_dist_inter: float = 6,
        dist_rbf_dim: int | None = None,
        time_rbf_dim: int | None = None,
        attention: bool = False,
        fourier_feat: str | None = None,
        fourier_params: dict[str, int] | None = None,
        normalization_factor: int = 100,
        aggregation_method: str = "sum",
        self_condition: bool = False,
    ):
        super().__init__()
        self.edge_dist_inter = edge_dist_inter
        self.self_condition = self_condition
        self.dist_rbf_dim = dist_rbf_dim
        self.time_rbf_dim = time_rbf_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(fine_input_dim, 2 * fine_input_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * fine_input_dim),
            nn.Linear(2 * fine_input_dim, hidden_dim),
        )

        self.coarse_encoder = nn.Sequential(
            nn.Linear(coarse_input_dim, 2 * coarse_input_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * coarse_input_dim),
            nn.Linear(2 * coarse_input_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * fine_input_dim),
            nn.SiLU(),
            nn.Linear(2 * fine_input_dim, fine_input_dim),
        )

        time_dim = time_rbf_dim if time_rbf_dim is not None else 1
        dynamics_node_nf = hidden_dim + time_dim

        if fourier_feat == "basic":
            fourier_embedding = (
                BasicFourierEmbedding(**fourier_params)
                if fourier_params is not None
                else BasicFourierEmbedding()
            )
        elif fourier_feat == "gaussian":
            fourier_embedding = (
                GaussianFourierEmbedding(**fourier_params)
                if fourier_params is not None
                else GaussianFourierEmbedding()
            )
        elif fourier_feat is None:
            fourier_embedding = None
        else:
            raise NotImplementedError(
                f"Invalid value for fourier_feat: {fourier_feat}. Expected 'basic' or 'gaussian' or 'None'."
            )

        # 1: extra or not, 3: distinguish new edge types
        edge_feat_s_dim = edge_feat_dim + 4
        edge_feat_v_dim = 0
        if self_condition:
            extra_s_dim = dist_rbf_dim if dist_rbf_dim is not None else 1
            edge_feat_s_dim += extra_s_dim
            edge_feat_v_dim += 1

        self.gvp_module = GVPNetwork(
            in_node_dims=(dynamics_node_nf, 0),
            in_edge_dims=(edge_feat_s_dim, edge_feat_v_dim),
            hidden_dims=(hidden_dim, hidden_dim),
            out_node_dims=(hidden_dim, 1),
            n_layers=n_layers,
            dist_rbf_dim=dist_rbf_dim,
            attention=attention,
            fourier_embedding=fourier_embedding,
            normalization_factor=normalization_factor,
            aggr=aggregation_method,
        )
        self.final_equiv_module = EquivariantUpdate(
            hidden_nf=hidden_dim,
            edges_in_d=edge_feat_s_dim,
            tanh=True,
            coords_range=15.0,
            normalization_factor=normalization_factor,
            aggregation_method="sum",
            reflection_equiv=False,
        )
        self.node_nf = dynamics_node_nf
        self.coord_dim = coord_dim

    def forward(
        self,
        xh_f: Tensor,  # [N_atom, 3 + fine_feat_dim]
        xh_c: Tensor,  # [N_frag, 3 + coarse_feat_dim]
        c_pos: Tensor,  # [N_atom, 3]
        t: Tensor,  # [B_in, 1]
        b_idx_f: Tensor,  # [N_atom]
        b_idx_c: Tensor,  # [N_frag]
        res_idx_f: Tensor,  # [N_atom]
        res_idx_c: Tensor,  # [N_frag]
        rep_mask: Tensor,  # [N_atom]
        edge_idx: Tensor,  # [N_edges, 2]
        edge_feat: Optional[Tensor] = None,  # [N_edges, edge_feat_dim]
        xh_f_sc: Optional[Tensor] = None,
    ):
        device = xh_f.device

        # Split into x (coordinates) and h (features)
        x_f = xh_f[:, : self.coord_dim].clone()
        h_f = xh_f[:, self.coord_dim :].clone()

        x_c = xh_c[:, : self.coord_dim].clone()
        h_c = xh_c[:, self.coord_dim :].clone()

        # Embed atom and coarse features in a shared space
        h_f = self.atom_encoder(h_f)
        h_c = self.coarse_encoder(h_c)

        x_cat = torch.cat((x_f, x_c))
        h_cat = torch.cat((h_f, h_c))
        b_idx = torch.cat((b_idx_f, b_idx_c))
        res_idx = torch.cat((res_idx_f, res_idx_c))
        rep_mask = torch.cat(
            (rep_mask, torch.ones(len(res_idx_c), device=device).type(torch.bool))
        )
        c_pos = torch.cat((c_pos, torch.zeros_like(x_c)))

        # Time embedding condition
        h_time = t[b_idx]
        if self.time_rbf_dim:
            h_time = embed_rbf(h_time, dist_max=1.0, dist_count=self.time_rbf_dim)
        h_cat = torch.cat([h_cat, h_time], dim=-1)

        # Construct edges
        if xh_f_sc is not None:
            x_sc = torch.cat([xh_f_sc[:, : self.coord_dim], x_c])
            edges, edge_feats_s, edge_feats_v = self.get_edges(
                b_idx, x_cat, res_idx, rep_mask, edge_idx, edge_feat, x_sc
            )
        else:
            edges, edge_feats_s, edge_feats_v = self.get_edges(
                b_idx, x_cat, res_idx, rep_mask, edge_idx, edge_feat
            )
        assert torch.all(b_idx[edges[0]] == b_idx[edges[1]])
        # The first edge feature indicates whether it is a covalent bond or not
        matches = edges.T[:, None, :] == edge_idx[None, :, :]
        covalent_mask = torch.any(torch.all(matches, dim=-1), dim=1)
        assert edge_feats_s[covalent_mask, 0].sum() == len(edge_idx)

        h_out, x_out = self.gvp_module(
            h=h_cat,
            x=x_cat,
            edge_index=edges,
            scalar_edge_feats=edge_feats_s,
            vector_edge_feats=edge_feats_v,
        )
        x_out = x_out.squeeze(dim=1)

        edge_feat, coord_diff = coord2diff(
            x_out, edges, norm_constant=0, return_norm=False
        )
        coord_cross = coord2cross(x_out, edges, b_idx, norm_constant=0)
        x_final = self.final_equiv_module(
            h_out,
            x_out,
            edges,
            coord_diff,
            coord_cross,
            edge_feats_s,
        )
        vel = x_final + c_pos

        # Decode atom features
        h_final = self.decoder(h_out)

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                print("Training: NaN detected in EGNN output. Velocity is set to zero!")
            else:
                raise ValueError("NaN detected in EGNN output")

        return torch.cat([vel, h_final], dim=-1)

    def get_edges(
        self,
        batch_idx: Tensor,
        pos: Tensor,
        res_idx: Tensor,
        rep_mask: Tensor,
        edge_idx: Tensor,
        edge_feats: Optional[Tensor] = None,
        pos_sc: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Constructs edge indices and features by concatenating the given edge
        index with fully-connected edges within the same residue and between residues
        within a threshold distance.

        Parameters
        ----------
        batch_idx : Tensor
            A tensor indicating the batch index for each node.
        pos : Tensor
            A tensor containing the positions of the nodes.
        res_idx : Tensor
            A tensor indicating the residue index for each node.
        edge_idx : Tensor
            A tensor containing the original edge indices.
        edge_feats : Optional[Tensor]
            A tensor containing the original edge features.

        Returns
        -------
        tuple[Tensor, Tensor]
            - new_edge_idx : Tensor
                The final edge index tensor after merging and deduplication.
            - new_edge_feats : Tensor
                The final edge feature tensor after concatenation and indexing.
        """

        def edge_onehot(adj: Tensor, class_idx: int, num_classes: int):
            class_idx = torch.tensor([class_idx], device=adj.device)
            onehot = torch.nn.functional.one_hot(class_idx, num_classes)
            return onehot.repeat(adj.numel(), 1).reshape(*adj.shape, -1)

        device = batch_idx.device

        # Fully-connected within the same residue index (adj_intra_res)
        # Inter-residue edges are formed according to edge_dist_inter
        # Representive nodes are fully-connected
        adj_b = batch_idx[:, None] == batch_idx[None, :]
        adj_intra_res = adj_b * (res_idx[:, None] == res_idx[None, :])
        adj_inter = adj_b * (
            (res_idx[:, None] != res_idx[None, :])
            & (torch.cdist(pos, pos) < self.edge_dist_inter)
        )
        adj_rep = adj_b * (rep_mask[:, None] * rep_mask[None, :])
        adj = adj_intra_res + adj_inter + adj_rep
        extra_edges = torch.stack(torch.where(adj), dim=0)
        n_extra_edges = extra_edges.shape[1]

        # NOTE: The concatenated edges and features are then indexed with
        # `first_indicies` to remove duplicates. Order: (given + new)[indexing]
        # Edge indices
        all_edges = torch.cat([edge_idx.T, extra_edges], dim=-1)
        _, idx, counts = torch.unique(
            all_edges, dim=1, sorted=True, return_inverse=True, return_counts=True
        )
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        new_edge_idx = all_edges[:, first_indicies]

        # Edge features
        new_edge_feats = torch.ones(len(edge_idx), device=device).reshape(-1, 1)
        new_edge_feats = torch.cat(
            [
                new_edge_feats,
                torch.zeros(n_extra_edges, device=device).reshape(-1, 1),
            ],
        )

        # To distinguish edge types for new edges (intra-residue, inter-residue, inter-rep)
        intra_res = edge_onehot(adj_intra_res, class_idx=0, num_classes=3)
        inter_res = edge_onehot(adj_inter, class_idx=1, num_classes=3)
        inter_rep = edge_onehot(adj_rep, class_idx=2, num_classes=3)
        res_e_feat = torch.zeros_like(intra_res, device=device)
        res_e_feat[torch.where(adj_intra_res)] = intra_res[torch.where(adj_intra_res)]
        res_e_feat[torch.where(adj_inter)] = inter_res[torch.where(adj_inter)]
        res_e_feat[torch.where(adj_rep)] = inter_rep[torch.where(adj_rep)]
        res_e_feat = torch.cat(
            [
                torch.zeros(len(edge_idx), res_e_feat.shape[-1], device=device),
                res_e_feat[torch.where(adj)],
            ]
        )
        new_edge_feats = torch.cat([new_edge_feats, res_e_feat], dim=-1)

        if edge_feats is not None:
            add_feat = torch.cat(
                [
                    edge_feats,
                    torch.zeros((n_extra_edges, edge_feats.shape[1]), device=device),
                ],
                dim=0,
            )
            new_edge_feats = torch.cat([new_edge_feats, add_feat], dim=-1)
        new_edge_feats = new_edge_feats[first_indicies, :]

        scalar_sc, vector_sc = None, None
        if self.self_condition:
            if pos_sc is not None:
                scalar_sc, vector_sc = coord2feat(pos_sc, new_edge_idx)
                if self.dist_rbf_dim is not None:
                    scalar_sc = embed_rbf(scalar_sc, dist_count=self.dist_rbf_dim)
            else:
                s_dim = self.dist_rbf_dim if self.dist_rbf_dim is not None else 1
                scalar_sc = torch.zeros(len(new_edge_feats), s_dim, device=device)
                vector_sc = torch.zeros(len(new_edge_feats), 1, 3, device=device)

        new_s_edge_feats = (
            torch.cat([new_edge_feats, scalar_sc], dim=-1)
            if scalar_sc is not None
            else new_edge_feats
        )
        new_v_edge_feats = vector_sc

        return new_edge_idx, new_s_edge_feats, new_v_edge_feats

    @property
    def name(self) -> str:
        return "gvp"
