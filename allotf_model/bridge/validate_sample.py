"""Validate a TransferSample against the frozen contract and print a human-readable coverage report.

Runs on every sample the bridge produces: a shape or alignment mistake here is a silent poison
downstream, so it fails loudly and early.
"""
import torch

from .transfer_sample import TransferSample


def validate(s: TransferSample):
    n = s.n_res
    # every per-residue tensor must have length N_res, aligned to the canonical index
    for name in TransferSample.PER_RESIDUE:
        t = getattr(s, name)
        if t.shape[0] != n:
            raise AssertionError("%s has %d rows, N_res=%d" % (name, t.shape[0], n))
    if n == 0:
        raise AssertionError("empty protein graph (N_res=0)")
    if int(s.protein_edge_index.max()) >= n if s.protein_edge_index.numel() else False:
        raise AssertionError("protein_edge_index references a residue >= N_res")
    if s.communication_edge_index.numel() and int(s.communication_edge_index.max()) >= n:
        raise AssertionError("communication_edge_index references a residue >= N_res")
    if s.cross_edge_index.numel():
        if int(s.cross_edge_index[0].max()) >= s.ligand_atom_features.shape[0]:
            raise AssertionError("cross_edge_index[0] references a ligand atom out of range")
        if int(s.cross_edge_index[1].max()) >= n:
            raise AssertionError("cross_edge_index[1] references a residue >= N_res")
    if not torch.isfinite(s.residue_positions).all():
        raise AssertionError("residue_positions has non-finite values")
    if not s.distal_mask.any():
        raise AssertionError("distal_mask is empty: the response-matching region is undefined")
    for nm in ("physics_aux", "physics_aux_confidence", "physics_aux_mask"):
        if getattr(s, nm).shape[0] != len(s.physics_aux_names):
            raise AssertionError("%s length != number of physics_aux_names" % nm)
    return True


def report(s: TransferSample):
    """The human-readable coverage line the spec asks for."""
    n = s.n_res
    teacher_cov = int((s.native_response_mask & (s.native_response_confidence > 0)).sum())
    phys_avail = int(s.physics_aux_mask.sum())
    warnings = []
    if s.ligand_coordinates.numel() and float(s.provenance.get("pose_confidence", 1.0)) < 0.5:
        warnings.append("ligand pose confidence low")
    if teacher_cov < int(s.distal_mask.sum()):
        warnings.append("native teacher does not cover the full distal region")
    lines = [
        "Scaffold: %s   candidate: %s   ligand: %s" % (s.scaffold_id, s.candidate_id, s.ligand_id),
        "Residues mapped: %d" % n,
        "Mutations: %d" % int(s.mutation_mask.sum()),
        "Pocket residues: %d   distal residues: %d" % (int(s.pocket_mask.sum()), int(s.distal_mask.sum())),
        "Native teacher coverage: %d/%d" % (teacher_cov, int(s.distal_mask.sum())),
        "Protein edges: %d   communication edges: %d" %
        (s.protein_edge_index.shape[1] if s.protein_edge_index.numel() else 0,
         s.communication_edge_index.shape[1] if s.communication_edge_index.numel() else 0),
        "Physics features available: %d/%d" % (phys_avail, len(s.physics_aux_names)),
        "Warnings: %s" % ("; ".join(warnings) if warnings else "none"),
    ]
    return "\n".join(lines)
