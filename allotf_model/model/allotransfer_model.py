"""AlloTransfer: redesign local ligand recognition while TRANSFERRING the native ligand-induced
DNA-release response to a non-native ligand.

The same conditioning machinery (protein encoder + ligand GNN + pocket-ligand cross-attention +
allosteric propagation) is run twice: once for the DESIGN (mutant scaffold + target ligand) and once
for the NATIVE reference (WT scaffold + native ligand). The native distal response is the frozen
teacher (detached). The model predicts DNA release and the functional class directly; the ranking
score is P(functional_sensor). The generator is a separate design-time tool - it never trains this
model and never provides DNA-release truth; its physical features enter only as confidence-masked aux.
"""
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


class AlloTransfer(nn.Module):
    def __init__(self, prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in, phys_dim,
                 aux_dim=0, h_dim=128, prot_layers=4, lig_layers=3, prop_steps=3):
        super().__init__()
        self.protein = MultiStateEncoder(prot_node_in, prot_edge_in, h_dim, prot_layers)
        self.ligand = LigandGNN(atom_in, bond_in, h_dim, lig_layers)
        self.cross = PocketLigandAttention(h_dim)
        self.prop = AllostericPropagation(h_dim, comm_edge_in, prop_steps)
        self.release = DNAReleaseHead(h_dim)
        self.joint = JointFunctionHead(h_dim, h_dim, phys_dim, aux_dim)
        self.aux_dim = aux_dim

    def _condition(self, protein, ligand, ce_idx, ce_attr, distal_idx):
        h_res = self.protein(protein)
        h_atoms, ligand_vec = self.ligand(ligand)
        pocket_ligand = self.cross(h_res[protein.pocket_idx], h_atoms)
        lig_dbd, lig_field = self.prop(h_res, protein.pocket_idx, pocket_ligand, ce_idx, ce_attr,
                                       protein.dbd_idx, protein.confidence)
        apo_dbd = _conf_mean(h_res, protein.dbd_idx, protein.confidence)
        dH_distal = (lig_field - h_res)[distal_idx].mean(0)      # ligand-induced distal state change
        return {"apo_dbd": apo_dbd, "lig_dbd": lig_dbd, "dH_distal": dH_distal, "ligand_vec": ligand_vec}

    def forward(self, ts):
        d = ts.design
        tgt = self._condition(d.protein, d.ligand, d.comm_edge_index, d.comm_edge_attr, ts.distal_idx)
        nat = self._condition(ts.native_protein, ts.native_ligand, d.comm_edge_index,
                              d.comm_edge_attr, ts.distal_idx)
        dH_native = nat["dH_distal"].detach()                   # frozen teacher

        release = self.release(tgt["apo_dbd"], tgt["lig_dbd"])
        if self.aux_dim and ts.aux.numel() == self.aux_dim:
            aux = ts.aux * ts.aux_confidence                    # confidence-masked generator features
        else:
            aux = tgt["apo_dbd"].new_zeros(self.aux_dim)
        joint = self.joint(tgt["apo_dbd"], tgt["lig_dbd"], tgt["dH_distal"], tgt["ligand_vec"],
                           release, d.physics, aux)

        out = {**release, **joint, "dH_target": tgt["dH_distal"], "dH_native": dH_native}
        return out
