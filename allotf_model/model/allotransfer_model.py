"""AlloTransfer: redesign local ligand recognition while TRANSFERRING the native ligand-induced
DNA-release response to a non-native ligand.

The one native teacher is TransferSample.native_response - a FROZEN, per-residue, scaffold-level
physical descriptor loaded from disk (WT apo vs native-holo), aligned to the canonical residue index,
and detached. There is NO EMA/latent teacher: the model never recomputes the native response, so the
paper's "frozen experimental response template" is exactly what the code uses.

For a candidate the model runs its conditioning body ONCE (ligand vs null-ligand, both propagated),
projects the per-residue ligand-induced change into the teacher's physical channels, and matches it to
the frozen native response over the valid distal residues. It also predicts DNA release and the
functional class. Gradients flow only into the candidate branch and the projection heads.
"""
import types

import torch
import torch.nn as nn

from .equivariant_encoder import MultiStateEncoder
from .ligand_gnn import LigandGNN
from .pocket_ligand_attention import PocketLigandAttention
from .allosteric_propagation import AllostericPropagation
from .dna_release_head import DNAReleaseHead
from .joint_function_head import JointFunctionHead, CLASSES

D_TEACHER = 4          # native_reference.CHANNELS: ca_displacement, contact_change, n_neigh_apo, n_neigh_holo


def _conf_mean(h, idx, conf):
    w = conf[idx].unsqueeze(-1)
    return (h[idx] * w).sum(0) / w.sum().clamp_min(1e-6)


class _Core(nn.Module):
    def __init__(self, prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in,
                 h_dim, prot_layers, lig_layers, prop_steps):
        super().__init__()
        self.protein = MultiStateEncoder(prot_node_in, prot_edge_in, h_dim, prot_layers)
        self.ligand = LigandGNN(atom_in, bond_in, h_dim, lig_layers)
        self.cross = PocketLigandAttention(h_dim)
        self.prop = AllostericPropagation(h_dim, comm_edge_in, prop_steps)
        self.h_dim = h_dim

    def condition_from_sample(self, ts):
        prot = types.SimpleNamespace(x=ts.residue_features, pos=ts.residue_positions,
                                     edge_index=ts.protein_edge_index, edge_attr=ts.protein_edge_features)
        lig = types.SimpleNamespace(x=ts.ligand_atom_features, edge_index=ts.ligand_edge_index,
                                    edge_attr=ts.ligand_edge_features)
        h_res = self.protein(prot)
        h_atoms, ligand_vec = self.ligand(lig)
        pk = ts.pocket_mask.nonzero(as_tuple=True)[0]
        dbd = ts.dbd_mask.nonzero(as_tuple=True)[0]
        conf = torch.ones(h_res.shape[0], device=h_res.device)   # structural confidence rides in node features
        pocket_ligand = self.cross(h_res[pk], h_atoms)
        # both states go through propagation, so the difference is the ligand effect, not a bias
        lig_dbd, lig_field = self.prop(h_res, pk, pocket_ligand, ts.communication_edge_index,
                                       ts.communication_edge_features, dbd, conf)
        apo_dbd, apo_field = self.prop(h_res, pk, h_res[pk], ts.communication_edge_index,
                                       ts.communication_edge_features, dbd, conf)
        return {"apo_dbd": apo_dbd, "lig_dbd": lig_dbd, "apo_field": apo_field, "lig_field": lig_field,
                "ligand_vec": ligand_vec, "pocket_ligand": pocket_ligand.mean(0)}


class AlloTransfer(nn.Module):
    def __init__(self, prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in, phys_dim,
                 h_dim=128, prot_layers=4, lig_layers=3, prop_steps=3):
        super().__init__()
        self.core = _Core(prot_node_in, prot_edge_in, atom_in, bond_in, comm_edge_in,
                          h_dim, prot_layers, lig_layers, prop_steps)
        # project the per-residue latent change into the teacher's physical channels, by meaning:
        self.response_delta_head = nn.Linear(h_dim, 2)     # ca_displacement, contact_count_change
        self.apo_context_head = nn.Linear(h_dim, 1)        # n_neighbours_apo
        self.lig_context_head = nn.Linear(h_dim, 1)        # n_neighbours_holo
        self.release = DNAReleaseHead(h_dim)
        self.bind = nn.Linear(h_dim, 1)                    # pocket-derived binding logit (recognition)
        self.joint = JointFunctionHead(h_dim, h_dim, phys_dim)

    def forward(self, ts):
        if int(ts.distal_mask.sum()) == 0:
            raise ValueError("empty distal mask: response-matching region undefined (fail closed)")
        t = self.core.condition_from_sample(ts)
        dH_field = t["lig_field"] - t["apo_field"]                       # [N_res, h]

        predicted_native = torch.cat([self.response_delta_head(dH_field),
                                      self.apo_context_head(t["apo_field"]),
                                      self.lig_context_head(t["lig_field"])], dim=1)   # [N_res, 4]

        distal = ts.distal_mask
        w = ts.native_response_confidence[distal].clamp_min(0).unsqueeze(1) + 1e-6
        dH_distal = (dH_field[distal] * w).sum(0) / w.sum()

        release = self.release(t["apo_dbd"], t["lig_dbd"])
        physics_masked = ts.physics_aux * ts.physics_aux_confidence.clamp(0, 1) * ts.physics_aux_mask.float()
        joint = self.joint(t["apo_dbd"], t["lig_dbd"], dH_distal, t["ligand_vec"], release, physics_masked)

        return {**release, **joint,
                "bind_logit": self.bind(t["pocket_ligand"]).squeeze(-1),   # pocket-derived recognition
                "dH_target_field": dH_field,
                "predicted_native_response": predicted_native,
                "native_response": ts.native_response.detach(),          # frozen teacher, no grad
                "native_response_mask": ts.native_response_mask,
                "native_response_confidence": ts.native_response_confidence}
