"""Target-conditioned Allosteric Coupling Network.

Protein encoder (E(3)-equivariant residue graph) + Ligand GNN + Pocket-ligand cross-attention +
Allosteric propagation + Joint functional heads. It is conditioned on (scaffold, mutation, ligand)
and predicts the two experimental states (no-ligand repression, with-ligand output) and the
functional class directly. The ranking score is P(functional_sensor) - not a product of separately
trained gates. Physics/resolvent/multi-state geometry enter as features and edge features, never as
the switching ground truth.
"""
import torch
import torch.nn as nn

from .equivariant_encoder import MultiStateEncoder
from .ligand_gnn import LigandGNN
from .pocket_ligand_attention import PocketLigandAttention
from .allosteric_propagation import AllostericPropagation
from .functional_heads import FunctionalHeads, CLASSES


def _conf_mean(h, idx, conf):
    w = conf[idx].unsqueeze(-1)
    return (h[idx] * w).sum(0) / w.sum().clamp_min(1e-6)


class AlloTF(nn.Module):
    def __init__(self, prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in, phys_dim,
                 h_dim=128, prot_layers=4, lig_layers=3, prop_steps=3):
        super().__init__()
        self.protein = MultiStateEncoder(prot_node_in, prot_edge_in, h_dim, prot_layers)
        self.ligand = LigandGNN(atom_in, bond_in, h_dim, lig_layers)
        self.cross = PocketLigandAttention(h_dim)
        self.prop = AllostericPropagation(h_dim, comm_edge_in, prop_steps)
        self.heads = FunctionalHeads(h_dim, phys_dim, h_dim)
        # auxiliary ligand-binding predictor from the ligand-conditioned pocket (a TASK, not a gate)
        self.aux_bind = nn.Linear(h_dim, 1)
        self.sensor_idx = CLASSES.index("functional_sensor")

    def forward(self, s):
        p = s.protein
        h_res = self.protein(p)                                   # (N_res, h) - ligand NOT in this graph
        h_atoms, ligand_vec = self.ligand(s.ligand)

        pocket_ligand = self.cross(h_res[p.pocket_idx], h_atoms)  # ligand-conditioned pocket
        lig_dbd, _ = self.prop(h_res, p.pocket_idx, pocket_ligand, s.comm_edge_index,
                               s.comm_edge_attr, p.dbd_idx, p.confidence)
        apo_dbd = _conf_mean(h_res, p.dbd_idx, p.confidence)      # no-ligand DBD readout

        out = self.heads(apo_dbd, lig_dbd, ligand_vec, s.physics)
        out["bind_logit"] = self.aux_bind(pocket_ligand.mean(0)).squeeze(-1)   # auxiliary
        probs = torch.softmax(out["class_logits"], dim=-1)
        out["class_probs"] = probs
        out["S_final"] = probs[self.sensor_idx]                   # P(functional_sensor)
        return out
