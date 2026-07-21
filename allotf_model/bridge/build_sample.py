"""Assemble one validated TransferSample from the frozen pipeline outputs + a native reference + the
functional labels. This is the single place the six extractors are combined, so alignment is checked
once, here.
"""
import torch

from .protein_graph import build_protein_graph, REGIONS
from .ligand_graph import build_ligand_graph
from .communication_graph import build_communication_graph
from .pocket_physics import pack_physics, build_cross_graph
from .transfer_sample import TransferSample
from .validate_sample import validate

_LABELS = ("apo_repression", "ligand_response", "dna_kd_apo", "dna_kd_ligand",
           "ligand_binding", "functional_class")


def _label(v):
    """value -> (tensor, mask). None/NaN -> (nan, False)."""
    if v is None or (isinstance(v, float) and v != v):
        return torch.tensor(float("nan")), torch.tensor(False)
    return torch.tensor(float(v)), torch.tensor(True)


def build_transfer_sample(sample_id, candidate_id, ligand_id, scaffold_pdb, mapping, mutations,
                          regions, native_reference, ligand_smiles, physics=None, labels=None,
                          confidence=None, pair_features=None, provenance=None):
    if native_reference.mapping.scaffold_id != mapping.scaffold_id or native_reference.mapping.n_res != mapping.n_res:
        raise ValueError("native reference and candidate must share the same scaffold mapping")

    pg = build_protein_graph(scaffold_pdb, mapping, mutations, regions, confidence)
    lg = build_ligand_graph(ligand_smiles)
    comm_res = set().union(*[set((regions.get(r) or ())) for r in
                             ("pocket", "pocket_exit", "hinge", "dimer_interface", "dbd")])
    cg = build_communication_graph(pg["residue_positions"], comm_res, pair_features)
    xg = build_cross_graph(pg["residue_positions"], pg["pocket_mask"].nonzero(as_tuple=True)[0],
                           lg["ligand_coordinates"])
    ph = pack_physics(physics or {})

    labels = labels or {}
    lab_t, lab_m = {}, {}
    for k in _LABELS:
        t, m = _label(labels.get(k))
        lab_t[k] = t.long() if k == "functional_class" and m else t
        lab_m[k] = m

    s = TransferSample(
        sample_id=sample_id, scaffold_id=mapping.scaffold_id, candidate_id=candidate_id, ligand_id=ligand_id,
        **{k: pg[k] for k in pg},
        ligand_atom_features=lg["ligand_atom_features"], ligand_edge_index=lg["ligand_edge_index"],
        ligand_edge_features=lg["ligand_edge_features"], ligand_coordinates=lg["ligand_coordinates"],
        cross_edge_index=xg["cross_edge_index"], cross_edge_features=xg["cross_edge_features"],
        communication_edge_index=cg["communication_edge_index"],
        communication_edge_features=cg["communication_edge_features"],
        physics_aux=ph["physics_aux"], physics_aux_names=ph["physics_aux_names"],
        physics_aux_confidence=ph["physics_aux_confidence"], physics_aux_mask=ph["physics_aux_mask"],
        native_response=native_reference.response, native_response_mask=native_reference.distal_mask,
        native_response_confidence=native_reference.confidence,
        apo_repression=lab_t["apo_repression"], ligand_response=lab_t["ligand_response"],
        dna_kd_apo=lab_t["dna_kd_apo"], dna_kd_ligand=lab_t["dna_kd_ligand"],
        ligand_binding=lab_t["ligand_binding"], functional_class=lab_t["functional_class"],
        apo_repression_label_mask=lab_m["apo_repression"], ligand_response_label_mask=lab_m["ligand_response"],
        dna_kd_apo_label_mask=lab_m["dna_kd_apo"], dna_kd_ligand_label_mask=lab_m["dna_kd_ligand"],
        ligand_binding_label_mask=lab_m["ligand_binding"], functional_class_label_mask=lab_m["functional_class"],
        provenance=provenance or {})
    validate(s)
    return s
