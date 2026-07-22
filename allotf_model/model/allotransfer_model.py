"""AlloTransfer: reprogram local ligand recognition, align the resulting perturbation with the native
DNA-release response, and optimise allosteric transmission gain while preserving apo-state repression.

The distal response is not COPIED but decomposed against the FROZEN native teacher:

    dH_target_distal = alpha * dH_native_distal + eps_perp

alpha is the allosteric gain (useful in a band, not forced to 1), eps_perp is off-path perturbation
(penalised). The teacher (TransferSample.native_response) is loaded from disk, detached, never drifts.

The model emits seven judgments per candidate: target_binding, apo_DNA_competence, response_alignment,
allosteric_gain, off_path_response, DNA_release_margin, functional_sensor_probability. Gradients flow
only into the candidate branch and the small heads.
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
from .coupling_gain_head import CouplingGainHead
from ..losses.response_transfer import decompose

D_TEACHER = 4          # native_reference.CHANNELS: ca_displacement, contact_change, n_neigh_apo, n_neigh_holo
GAIN_REGIONS = ("pocket_exit_mask", "hinge_mask", "dimer_interface_mask", "dbd_mask")


def _pool(field, mask):
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return field.new_zeros(field.shape[1])
    return field[idx].mean(0)


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
        conf = torch.ones(h_res.shape[0], device=h_res.device)
        pocket_ligand = self.cross(h_res[pk], h_atoms)
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
        self.response_delta_head = nn.Linear(h_dim, 2)     # ca_displacement, contact_count_change
        self.apo_context_head = nn.Linear(h_dim, 1)        # n_neighbours_apo
        self.lig_context_head = nn.Linear(h_dim, 1)        # n_neighbours_holo
        self.release = DNAReleaseHead(h_dim)
        self.bind = nn.Linear(h_dim, 1)                    # target_binding (pocket-derived)
        self.apo_competence = nn.Linear(h_dim, 1)          # apo_DNA_competence (apo still represses)
        self.gain = CouplingGainHead(h_dim)
        self.joint = JointFunctionHead(h_dim, h_dim, phys_dim)

    def _region_gains(self, predicted, native, ts, base_mask, nconf):
        """Per-region allosteric gain alpha, restricted to each transduction region (analytic)."""
        out = {}
        predicted = predicted.detach()                 # attribution only, not a gradient path
        for name in GAIN_REGIONS:
            region = getattr(ts, name) & base_mask
            try:
                out[name.replace("_mask", "")] = float(decompose(predicted, native, region, nconf)["alpha"])
            except ValueError:
                out[name.replace("_mask", "")] = 0.0   # < 2 residues in region: no attribution
        return out

    def forward(self, ts):
        if int(ts.distal_mask.sum()) == 0:
            raise ValueError("empty distal mask: response-matching region undefined (fail closed)")
        t = self.core.condition_from_sample(ts)
        dH_field = t["lig_field"] - t["apo_field"]                       # [N_res, h]

        predicted_native = torch.cat([self.response_delta_head(dH_field),
                                      self.apo_context_head(t["apo_field"]),
                                      self.lig_context_head(t["lig_field"])], dim=1)   # [N_res, 4]

        # one mask everywhere: distal AND has a native response; teacher detached (never gets gradient)
        base_mask = ts.distal_mask & ts.native_response_mask
        native = ts.native_response.detach()
        nconf = ts.native_response_confidence.detach()
        d = decompose(predicted_native, native, base_mask, nconf)   # alpha * native + eps_perp
        offpath = (d["w"].unsqueeze(1) * d["eps_perp"] ** 2).sum() / (d["w"].sum() * d["eps_perp"].shape[1])

        dH_distal = _pool(dH_field, ts.distal_mask)
        hinge_repr = _pool(dH_field, ts.hinge_mask)
        phys_conf = (ts.physics_aux_confidence * ts.physics_aux_mask.float()).mean()
        region_contrib = self._region_gains(predicted_native, native, ts, base_mask, nconf)
        gain = self.gain(dH_distal, hinge_repr, t["lig_dbd"], phys_conf, region_contrib)

        release = self.release(t["apo_dbd"], t["lig_dbd"])
        physics_masked = ts.physics_aux * ts.physics_aux_confidence.clamp(0, 1) * ts.physics_aux_mask.float()
        joint = self.joint(t["apo_dbd"], t["lig_dbd"], dH_distal, t["ligand_vec"], release, physics_masked)

        return {**release, **joint,
                # the seven judgments
                "target_binding": self.bind(t["pocket_ligand"]).squeeze(-1),
                "apo_DNA_competence": torch.sigmoid(self.apo_competence(t["apo_dbd"]).squeeze(-1)),
                "response_alignment": d["alignment"],
                "allosteric_gain": d["alpha"],
                "gain_mean": gain["gain_mean"], "gain_uncertainty": gain["gain_uncertainty"],
                "gain_region_contributions": gain["gain_region_contributions"],
                "off_path_response": offpath,
                "DNA_release_margin": release["release_margin"],
                "functional_sensor_probability": joint["S_design"],
                # fields the losses consume
                "bind_logit": self.bind(t["pocket_ligand"]).squeeze(-1),
                "dH_target_field": dH_field,
                "predicted_native_response": predicted_native,
                "native_response": native,                       # already detached
                "native_response_mask": base_mask,               # loss uses the SAME combined mask
                "native_response_confidence": nconf}
