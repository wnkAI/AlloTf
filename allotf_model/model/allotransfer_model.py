"""AlloTransfer: redesign local ligand recognition while TRANSFERRING the native ligand-induced
DNA-release response to a non-native ligand.

The same conditioning machinery (protein encoder + ligand GNN + pocket-ligand cross-attention +
allosteric propagation) produces, for a given (scaffold, ligand): a null-ligand propagated field and
a ligand-conditioned propagated field. The ligand-induced change is their difference - BOTH go
through propagation, so the response reflects the ligand, not a propagation bias.

The native reference (WT + native ligand) is computed by a separate EMA TEACHER copy (no gradient,
slowly tracking the student), so it is a genuinely frozen, stable target rather than the student's own
drifting output. The native branch is also anchored to the known native release during training, so
the teacher encodes a real allosteric response, not an arbitrary latent.
"""
import copy

import torch
import torch.nn as nn

from .equivariant_encoder import MultiStateEncoder
from .ligand_gnn import LigandGNN
from .pocket_ligand_attention import PocketLigandAttention
from .allosteric_propagation import AllostericPropagation
from .dna_release_head import DNAReleaseHead
from .joint_function_head import JointFunctionHead, CLASSES


def _conf_mean(h, idx, conf):
    w = conf[idx].unsqueeze(-1)
    return (h[idx] * w).sum(0) / w.sum().clamp_min(1e-6)


class _Core(nn.Module):
    """The shared conditioning body. Kept as a submodule so the EMA teacher can copy exactly it."""

    def __init__(self, prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in,
                 h_dim, prot_layers, lig_layers, prop_steps):
        super().__init__()
        self.protein = MultiStateEncoder(prot_node_in, prot_edge_in, h_dim, prot_layers)
        self.ligand = LigandGNN(atom_in, bond_in, h_dim, lig_layers)
        self.cross = PocketLigandAttention(h_dim)
        self.prop = AllostericPropagation(h_dim, comm_edge_in, prop_steps)
        self.h_dim = h_dim

    def condition(self, protein, ligand, ce_idx, ce_attr, distal_idx):
        """-> apo_dbd, lig_dbd, dH_distal (ligand-induced), ligand_vec, pocket_ligand.

        BOTH apo and ligand states go through propagation; the difference is the ligand effect and
        carries no propagation bias. The null-ligand seed is the plain pocket embedding."""
        h_res = self.protein(protein)
        h_atoms, ligand_vec = self.ligand(ligand)
        pk = protein.pocket_idx
        pocket_ligand = self.cross(h_res[pk], h_atoms)

        lig_dbd, lig_field = self.prop(h_res, pk, pocket_ligand, ce_idx, ce_attr,
                                       protein.dbd_idx, protein.confidence)
        apo_dbd, apo_field = self.prop(h_res, pk, h_res[pk], ce_idx, ce_attr,   # null-ligand seed
                                       protein.dbd_idx, protein.confidence)
        conf = protein.confidence[distal_idx].unsqueeze(-1)
        dH_distal = ((lig_field - apo_field)[distal_idx] * conf).sum(0) / conf.sum().clamp_min(1e-6)
        return {"apo_dbd": apo_dbd, "lig_dbd": lig_dbd, "dH_distal": dH_distal,
                "ligand_vec": ligand_vec, "pocket_ligand": pocket_ligand.mean(0)}


class AlloTransfer(nn.Module):
    def __init__(self, prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in, phys_dim,
                 aux_dim=0, h_dim=128, prot_layers=4, lig_layers=3, prop_steps=3, ema=0.99):
        super().__init__()
        self.core = _Core(prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in,
                          h_dim, prot_layers, lig_layers, prop_steps)
        self.teacher = copy.deepcopy(self.core)               # frozen EMA copy for the native reference
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.ema = ema
        self.release = DNAReleaseHead(h_dim)
        self.bind = nn.Linear(h_dim, 1)                       # pocket-derived binding logit (recognition)
        self.joint = JointFunctionHead(h_dim, h_dim, phys_dim, aux_dim)
        self.aux_dim = aux_dim

    @torch.no_grad()
    def update_teacher(self):
        """Call once per optimiser step: teacher <- ema*teacher + (1-ema)*student."""
        for t, s in zip(self.teacher.parameters(), self.core.parameters()):
            t.mul_(self.ema).add_(s.detach(), alpha=1 - self.ema)
        for t, s in zip(self.teacher.buffers(), self.core.buffers()):
            t.copy_(s)

    def _aux(self, ts, ref):
        if self.aux_dim and ts.aux.numel() == self.aux_dim and ts.aux_confidence.numel() == self.aux_dim:
            a = torch.nan_to_num(ts.aux, nan=0.0) * ts.aux_confidence.clamp(0, 1)
            return a.to(ref.dtype)
        return ref.new_zeros(self.aux_dim)

    def forward(self, ts):
        d = ts.design
        if ts.distal_idx.numel() == 0:
            raise ValueError("distal_idx is empty: the response-matching region is undefined")
        tgt = self.core.condition(d.protein, d.ligand, d.comm_edge_index, d.comm_edge_attr, ts.distal_idx)
        with torch.no_grad():                                 # frozen EMA teacher
            nat = self.teacher.condition(ts.native_protein, ts.native_ligand, d.comm_edge_index,
                                         d.comm_edge_attr, ts.distal_idx)

        release = self.release(tgt["apo_dbd"], tgt["lig_dbd"])
        nat_release = self.release(nat["apo_dbd"], nat["lig_dbd"])   # anchored to known native release
        aux = self._aux(ts, tgt["apo_dbd"])
        joint = self.joint(tgt["apo_dbd"], tgt["lig_dbd"], tgt["dH_distal"], tgt["ligand_vec"],
                           release, d.physics, aux)

        out = {**release, **joint,
               "dH_target": tgt["dH_distal"], "dH_native": nat["dH_distal"],   # already no-grad
               "bind_logit": self.bind(tgt["pocket_ligand"]).squeeze(-1),
               "native_ddG": nat_release["ddG_coupling"]}
        return out
