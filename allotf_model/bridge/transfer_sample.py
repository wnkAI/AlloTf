"""FROZEN data contract for the pipeline->model bridge. Every extractor writes into these exact
fields; the model reads them. Freezing this first is what stops the six extractors from each
inventing their own indexing, masks and missing-value conventions.

Three rules the contract enforces by shape:
  1. every per-residue tensor has length N_res, aligned to the canonical residue index (residue_mapping);
  2. native_response is a scaffold-level teacher object, loaded and aligned, never recomputed per candidate;
  3. every physics/teacher quantity carries value + confidence + availability mask - a real 0.0 is never
     confused with "not computed / failed / low-confidence model".
"""
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TransferSample:
    # Identity
    sample_id: str
    scaffold_id: str
    candidate_id: str
    ligand_id: str

    # Protein graph
    residue_features: torch.Tensor       # [N_res, F_res] invariant scalars
    residue_vectors: torch.Tensor        # [N_res, F_vec, 3] equivariant vector features (may be empty)
    residue_positions: torch.Tensor      # [N_res, 3] CA
    protein_edge_index: torch.Tensor     # [2, E_protein] [receiver, sender]
    protein_edge_features: torch.Tensor  # [E_protein, F_edge]

    # Mutation encoding
    wt_residue_ids: torch.Tensor         # [N_res] long (amino-acid id)
    mutant_residue_ids: torch.Tensor     # [N_res] long
    mutation_mask: torch.Tensor          # [N_res] bool

    # Region definitions (all [N_res] bool, canonical-index aligned)
    pocket_mask: torch.Tensor
    pocket_exit_mask: torch.Tensor
    hinge_mask: torch.Tensor
    dimer_interface_mask: torch.Tensor
    dbd_mask: torch.Tensor
    dna_contact_mask: torch.Tensor
    distal_mask: torch.Tensor

    # Ligand graph
    ligand_atom_features: torch.Tensor   # [N_atom, F_atom]
    ligand_edge_index: torch.Tensor      # [2, E_ligand]
    ligand_edge_features: torch.Tensor   # [E_ligand, F_bond]
    ligand_coordinates: torch.Tensor     # [N_atom, 3] optional pose (may be empty)

    # Protein-ligand geometry
    cross_edge_index: torch.Tensor       # [2, E_cross] ligand atom <-> pocket residue
    cross_edge_features: torch.Tensor    # [E_cross, F_cross]

    # Communication graph (channels: distance, contact, resolvent, calibrated_path,
    # contact_churn, hinge_weight, confidence)
    communication_edge_index: torch.Tensor
    communication_edge_features: torch.Tensor

    # Frozen pipeline auxiliary features (value + confidence + availability)
    physics_aux: torch.Tensor            # [F_phys]
    physics_aux_names: tuple
    physics_aux_confidence: torch.Tensor # [F_phys] in [0,1]
    physics_aux_mask: torch.Tensor       # [F_phys] bool, available?

    # Native response teacher (scaffold-level, frozen; aligned to canonical index)
    native_response: torch.Tensor        # [N_res, D_teacher]
    native_response_mask: torch.Tensor   # [N_res] bool (usually distal_mask)
    native_response_confidence: torch.Tensor  # [N_res] in [0,1]

    # Functional labels (scalar tensors; value meaningless where the mask is False)
    apo_repression: torch.Tensor
    ligand_response: torch.Tensor
    dna_kd_apo: torch.Tensor
    dna_kd_ligand: torch.Tensor
    ligand_binding: torch.Tensor
    functional_class: torch.Tensor       # long 0..4, or ignore index

    # Explicit label masks (bool)
    apo_repression_label_mask: torch.Tensor
    ligand_response_label_mask: torch.Tensor
    dna_kd_apo_label_mask: torch.Tensor
    dna_kd_ligand_label_mask: torch.Tensor
    ligand_binding_label_mask: torch.Tensor
    functional_class_label_mask: torch.Tensor

    # Provenance and debugging
    provenance: dict = field(default_factory=dict)

    @property
    def n_res(self):
        return self.residue_features.shape[0]

    PER_RESIDUE = ("residue_features", "residue_positions", "wt_residue_ids", "mutant_residue_ids",
                   "mutation_mask", "pocket_mask", "pocket_exit_mask", "hinge_mask",
                   "dimer_interface_mask", "dbd_mask", "dna_contact_mask", "distal_mask",
                   "native_response", "native_response_mask", "native_response_confidence")

    def to(self, device):
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if torch.is_tensor(v):
                setattr(self, f, v.to(device))
        return self
