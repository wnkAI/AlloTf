"""Pocket physics extractor: package the FROZEN pipeline's numbers as auxiliary features - PhysPocket
terms, six-state coupling, S_release, strain, decoy gap, pose consistency. Every quantity is carried
as value + confidence + availability so "not computed", "failed", "low-confidence model" and a real
0.0 are never confused. Produces NO labels; the model may down-weight or ignore low-confidence ones.
"""
import torch

# the standard pipeline quantities we surface (extend as the pipeline exposes more)
STANDARD = ("phys_pocket", "ddG_coupling", "S_release", "ligand_strain", "decoy_gap",
            "pose_consistency", "template_similarity")


def pack_physics(physics, names=STANDARD):
    """physics: {name: {"value": float, "confidence": float, "available": bool}} (missing keys -> not
    available). -> dict(physics_aux, physics_aux_names, physics_aux_confidence, physics_aux_mask)."""
    vals, confs, masks = [], [], []
    for nm in names:
        p = physics.get(nm) or {}
        ok = bool(p.get("available", "value" in p))
        v = float(p.get("value", 0.0)) if ok else 0.0
        vals.append(v if v == v else 0.0)                    # never let a NaN through as a value
        confs.append(float(p.get("confidence", 0.0)) if ok else 0.0)
        masks.append(ok and (v == v))
    return dict(physics_aux=torch.tensor(vals, dtype=torch.float32),
                physics_aux_names=tuple(names),
                physics_aux_confidence=torch.tensor(confs, dtype=torch.float32),
                physics_aux_mask=torch.tensor(masks, dtype=torch.bool))


def build_cross_graph(positions, pocket_idx, ligand_coords, cutoff=6.0):
    """Protein-ligand contact edges: ligand atom <-> pocket residue within cutoff. Empty when the
    ligand has no pose (co-folding/docking not run). -> dict(cross_edge_index [2,E], cross_edge_features)."""
    if ligand_coords is None or ligand_coords.numel() == 0 or len(pocket_idx) == 0:
        return dict(cross_edge_index=torch.zeros(2, 0, dtype=torch.long),
                    cross_edge_features=torch.zeros(0, 1))
    pk = positions[pocket_idx]
    d = torch.cdist(ligand_coords, pk)                       # [n_atom, n_pocket]
    at, pj = (d < cutoff).nonzero(as_tuple=False).t()
    edge_index = torch.stack([at, pocket_idx[pj]])           # [atom_idx, residue canonical index]
    return dict(cross_edge_index=edge_index, cross_edge_features=d[at, pj].unsqueeze(1))
