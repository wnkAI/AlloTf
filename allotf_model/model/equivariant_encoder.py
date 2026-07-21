"""A small E(n)-equivariant graph encoder (EGNN, Satorras et al. 2021), pure PyTorch, no e3nn/pyg.

Chosen deliberately: the message and node updates act on invariant scalars while coordinate updates
move along relative-position vectors, so node embeddings are invariant and the geometry is equivariant
to rotation/translation - all we need, at a fraction of the weight and dependency cost of an
irreps-based network. The SAME encoder (shared weights) runs on apo/lig/dna so any difference in the
embeddings comes from the state, not from three different models.
"""
import torch
import torch.nn as nn


def _scatter_sum(src, index, n):
    """Sum src rows into n buckets by index, without torch_scatter."""
    out = src.new_zeros((n,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


class EGNNLayer(nn.Module):
    def __init__(self, h_dim, edge_dim, m_dim=None, act=nn.SiLU):
        super().__init__()
        m_dim = m_dim or h_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * h_dim + 1 + edge_dim, m_dim), act(),
            nn.Linear(m_dim, m_dim), act())
        self.node_mlp = nn.Sequential(
            nn.Linear(h_dim + m_dim, h_dim), act(),
            nn.Linear(h_dim, h_dim))
        self.coord_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim), act(), nn.Linear(m_dim, 1))
        # small coordinate step so early training does not blow the geometry up
        with torch.no_grad():
            self.coord_mlp[-1].weight.mul_(0.1)

    def forward(self, h, pos, edge_index, edge_attr):
        i, j = edge_index                                  # i <- j messages
        rel = pos[i] - pos[j]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h[i], h[j], dist2, edge_attr], dim=-1))
        # equivariant coordinate update
        coord_w = self.coord_mlp(m_ij)
        n = h.shape[0]
        denom = _scatter_sum(torch.ones_like(coord_w), i, n).clamp_min(1.0)
        pos = pos + _scatter_sum(rel * coord_w, i, n) / denom
        # invariant node update
        m_i = _scatter_sum(m_ij, i, n)
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h, pos


class MultiStateEncoder(nn.Module):
    """One state graph -> (N, h_dim) invariant node embeddings. Shared across the three states."""

    def __init__(self, node_in, edge_in, h_dim=128, layers=4):
        super().__init__()
        self.embed = nn.Linear(node_in, h_dim)
        self.edge_in = edge_in
        self.layers = nn.ModuleList(EGNNLayer(h_dim, edge_in) for _ in range(layers))
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, g):
        h = self.embed(g.x)
        pos = g.pos
        for layer in self.layers:
            h, pos = layer(h, pos, g.edge_index, g.edge_attr)
        return self.norm(h)
