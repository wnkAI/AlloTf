"""Small-molecule graph encoder, trained from scratch (no MolT5/Uni-Mol). Message passing over the
ligand bond graph -> per-atom embeddings and a pooled ligand vector. The atom embeddings feed the
pocket-ligand cross-attention; the pooled vector conditions the whole prediction on the target."""
import torch
import torch.nn as nn

from .equivariant_encoder import _scatter_sum


class LigandGNN(nn.Module):
    def __init__(self, atom_in, bond_in, h_dim=128, layers=3, act=nn.SiLU):
        super().__init__()
        self.embed = nn.Linear(atom_in, h_dim)
        self.msg = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * h_dim + bond_in, h_dim), act(), nn.Linear(h_dim, h_dim))
            for _ in range(layers))
        self.upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * h_dim, h_dim), act(), nn.Linear(h_dim, h_dim))
            for _ in range(layers))
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, g):
        h = self.embed(g.x)
        i, j = g.edge_index
        n = h.shape[0]
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([h[i], h[j], g.edge_attr], dim=-1))
            h = h + upd(torch.cat([h, _scatter_sum(m, i, n)], dim=-1))
        h = self.norm(h)
        return h, h.mean(0)                       # per-atom embeddings, pooled ligand vector
