"""Input contract for the target-conditioned model. ONE scaffold graph (with the variant's mutation
encoded on the nodes) + ONE ligand graph + a communication graph carrying the physics path features.

No per-variant three-state structure: inference needs only the scaffold, the mutation, the target
ligand, and confidence-masked derived features. Missing/predicted geometry rides a confidence mask so
it is never treated as equal to experiment.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ProteinGraph:
    x: torch.Tensor            # (N_res, Fp) residue features incl. WT->mutant encoding, region flags
    pos: torch.Tensor          # (N_res, 3) CA coordinates (scaffold; confidence-masked where modelled)
    edge_index: torch.Tensor   # (2, E) [receiver, sender]
    edge_attr: torch.Tensor    # (E, Fpe)
    pocket_idx: torch.Tensor   # (n_pocket,) residue indices lining the pocket
    dbd_idx: torch.Tensor      # (n_dbd,) DNA-binding-domain residue indices
    confidence: torch.Tensor   # (N_res,) in [0,1], structure reliability per residue

    def to(self, d):
        return ProteinGraph(self.x.to(d), self.pos.to(d), self.edge_index.to(d), self.edge_attr.to(d),
                            self.pocket_idx.to(d), self.dbd_idx.to(d), self.confidence.to(d))


@dataclass
class LigandGraph:
    x: torch.Tensor            # (N_atom, Fl) atom features: element, charge, aromatic, donor/acceptor...
    edge_index: torch.Tensor   # (2, El)
    edge_attr: torch.Tensor    # (El, Fle) bond order / aromaticity / in-ring

    def to(self, d):
        return LigandGraph(self.x.to(d), self.edge_index.to(d), self.edge_attr.to(d))


@dataclass
class Sample:
    protein: ProteinGraph
    ligand: LigandGraph
    comm_edge_index: torch.Tensor      # (2, Ec) pocket->DBD communication graph over residues
    comm_edge_attr: torch.Tensor       # (Ec, Fc) path features: resolvent gain, hinge, churn, calibrated path
    physics: torch.Tensor              # (Fphys,) pocket physics proxies (vdW/Coulomb/Hbond/strain/spec)
    tf_id: Optional[str] = None
    ligand_id: Optional[str] = None

    def to(self, d):
        return Sample(self.protein.to(d), self.ligand.to(d), self.comm_edge_index.to(d),
                      self.comm_edge_attr.to(d), self.physics.to(d), self.tf_id, self.ligand_id)


@dataclass
class TransferSample:
    """A design candidate PLUS the native response teacher. The native reference (WT scaffold + native
    ligand) is the SAME scaffold as the design, so it reuses the design's communication graph, pocket
    and DBD indices; only the residue features (WT vs mutant encoding) and the ligand differ.

    distal_idx marks the response-matching region (pocket exit -> hinge -> dimer interface -> DBD):
    the transfer loss is applied ONLY there, so the new ligand's pocket chemistry stays free.
    aux carries generator-derived physical features (pose confidence, strain, backbone displacement,
    docking consistency) with a per-feature confidence in [0,1]; the model may ignore low-confidence
    ones and they never serve as a functional label.
    """
    design: "Sample"                   # mutant scaffold + target ligand
    native_protein: ProteinGraph       # WT scaffold (same structure, WT encoding)
    native_ligand: LigandGraph         # native effector
    distal_idx: torch.Tensor           # residues where the native response is matched
    aux: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    aux_confidence: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    def to(self, d):
        return TransferSample(self.design.to(d), self.native_protein.to(d), self.native_ligand.to(d),
                              self.distal_idx.to(d), self.aux.to(d), self.aux_confidence.to(d))


def collate(samples):
    return list(samples)
