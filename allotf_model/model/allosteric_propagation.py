"""Allosteric propagation: carry the ligand-conditioned pocket representation along the pocket->DBD
communication graph to the DNA-binding domain. This is what makes the network an ALLOSTERIC model
rather than a pocket scorer - the coupling lives in the model body, not in a separate head.

The physics path signals (resolvent gain, hinge integrity, contact churn, calibrated path, template
similarity) enter as EDGE FEATURES of the communication graph and as a gate on message passing, not
as pseudo-labels. Ligand information is injected at the pocket residues; a few propagation steps then
move it to the DBD. Residues are confidence-weighted so modelled geometry contributes less.
"""
import torch
import torch.nn as nn

from .equivariant_encoder import _scatter_sum


class AllostericPropagation(nn.Module):
    def __init__(self, h_dim, comm_edge_in, steps=3, act=nn.SiLU):
        super().__init__()
        self.steps = steps
        self.msg = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * h_dim + comm_edge_in, h_dim), act(), nn.Linear(h_dim, h_dim))
            for _ in range(steps))
        # edge gate from the physics path features: a broken path passes less signal
        self.gate = nn.ModuleList(
            nn.Sequential(nn.Linear(comm_edge_in, h_dim), nn.Sigmoid()) for _ in range(steps))
        self.upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * h_dim, h_dim), act(), nn.Linear(h_dim, h_dim))
            for _ in range(steps))
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, h_res, pocket_idx, pocket_ligand, comm_edge_index, comm_edge_attr,
                dbd_idx, confidence):
        h = h_res.clone()
        h[pocket_idx] = pocket_ligand                         # inject ligand-conditioned pocket rep
        h = h * confidence.unsqueeze(-1)                      # modelled residues contribute less
        i, j = comm_edge_index
        n = h.shape[0]
        for msg, gate, upd in zip(self.msg, self.gate, self.upd):
            m = msg(torch.cat([h[i], h[j], comm_edge_attr], dim=-1)) * gate(comm_edge_attr)
            h = h + upd(torch.cat([h, _scatter_sum(m, i, n)], dim=-1))
        h = self.norm(h)
        return h[dbd_idx].mean(0), h                          # DBD readout (ligand-conditioned), full field
