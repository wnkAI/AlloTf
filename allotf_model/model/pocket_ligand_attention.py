"""Pocket-ligand cross-attention: let ligand atoms interact explicitly with pocket residues so the
model can learn which mutations recognise which ligand, which interactions relate to the response,
and why similar ligands give different outputs.

Pocket residues attend over ligand atoms (queries = pocket residues, keys/values = ligand atoms).
The output is a ligand-conditioned pocket representation - the seed that the allosteric propagation
carries to the DBD.
"""
import torch
import torch.nn as nn


class PocketLigandAttention(nn.Module):
    def __init__(self, h_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(h_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(h_dim)
        self.ff = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))

    def forward(self, h_pocket, h_ligand_atoms):
        """h_pocket: (n_pocket, h). h_ligand_atoms: (n_atom, h). -> (n_pocket, h) ligand-conditioned."""
        q = h_pocket.unsqueeze(0)                    # (1, n_pocket, h)
        kv = h_ligand_atoms.unsqueeze(0)             # (1, n_atom, h)
        ctx, _ = self.attn(q, kv, kv)                # pocket residues attend over ligand atoms
        h = self.norm(h_pocket + ctx.squeeze(0))
        return h + self.ff(h)
