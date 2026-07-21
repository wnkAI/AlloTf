"""A small O(3)-equivariant graph encoder (EGNN, Satorras et al. 2021), pure PyTorch, no e3nn/pyg.

Node embeddings are INVARIANT to rotation/translation (and, being distance-based, to reflection - so
this is O(3), not strictly SE(3); mirror geometries are indistinguishable unless a chirality feature
is added to the invariant node inputs). The coordinate update moves along relative-position vectors,
keeping the geometry equivariant. The SAME encoder (shared weights) runs on apo/lig/dna so any
difference in the embeddings comes from the state, not from three different models.

Inputs must be invariant scalars: x and edge_attr carry no raw coordinates (node type, chemistry,
region flags, distances - yes; absolute xyz - no), or invariance breaks. edge_index is [receiver i,
sender j]: message i <- j.
"""
import torch
import torch.nn as nn


def _scatter_sum(src, index, n):
    """Sum src rows into n buckets by index, without torch_scatter."""
    out = src.new_zeros((n,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


class EGNNLayer(nn.Module):
    def __init__(self, h_dim, edge_dim, m_dim=None, act=nn.SiLU, update_coords=True):
        super().__init__()
        m_dim = m_dim or h_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * h_dim + 1 + edge_dim, m_dim), act(),
            nn.Linear(m_dim, m_dim), act())
        self.node_mlp = nn.Sequential(
            nn.Linear(h_dim + m_dim, h_dim), act(),
            nn.Linear(h_dim, h_dim))
        self.update_coords = update_coords
        if update_coords:
            self.coord_mlp = nn.Sequential(nn.Linear(m_dim, m_dim), act(), nn.Linear(m_dim, 1))
            with torch.no_grad():
                self.coord_mlp[-1].weight.mul_(0.1)   # small step so early training keeps geometry sane

    def forward(self, h, pos, edge_index, edge_attr):
        i, j = edge_index                                  # i <- j messages
        rel = pos[i] - pos[j]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h[i], h[j], dist2, edge_attr], dim=-1))
        n = h.shape[0]
        if self.update_coords:                             # equivariant coordinate update
            coord_w = self.coord_mlp(m_ij)
            denom = _scatter_sum(torch.ones_like(coord_w), i, n).clamp_min(1.0)
            pos = pos + _scatter_sum(rel * coord_w, i, n) / denom
        m_i = _scatter_sum(m_ij, i, n)                     # invariant node update
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h, pos


class MultiStateEncoder(nn.Module):
    """One state graph -> (N, h_dim) invariant node embeddings. Shared across the three states."""

    def __init__(self, node_in, edge_in, h_dim=128, layers=4):
        super().__init__()
        self.embed = nn.Linear(node_in, h_dim)
        self.edge_in = edge_in
        # the last layer's coordinate update would be discarded (only invariant h is returned), so it
        # does not compute one - no dead parameters, no wasted scatter
        self.layers = nn.ModuleList(
            EGNNLayer(h_dim, edge_in, update_coords=(k < layers - 1)) for k in range(layers))
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, g):
        h = self.embed(g.x)
        pos = g.pos
        for layer in self.layers:
            h, pos = layer(h, pos, g.edge_index, g.edge_attr)
        return self.norm(h)
