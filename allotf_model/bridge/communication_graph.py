"""Communication graph extractor: the pocket->hinge->DBD residue network the allosteric signal
travels along. Edges connect communication-relevant residues (pocket, pocket exit, hinge, dimer
interface, DBD) by CA proximity; each edge carries SEPARATE channels so three different notions are
never collapsed into one scalar:

    distance, contact            - static geometry of THIS scaffold;
    resolvent, calibrated_path   - static path-preservation signals (from the frozen pipeline);
    contact_churn, hinge_weight  - the native apo->holo change (from the frozen pipeline);
    confidence                   - whether the pipeline channels were actually available.

Candidate-specific, ligand-conditioned propagation is NOT baked in here - that is computed inside the
model. This graph is the fixed substrate it propagates over.
"""
import torch

CHANNELS = ("distance", "contact", "resolvent", "calibrated_path", "contact_churn",
            "hinge_weight", "confidence")
COMM_CUTOFF = 10.0


def build_communication_graph(positions, comm_residues, pair_features=None, cutoff=COMM_CUTOFF):
    """positions: [N_res, 3] CA (from protein_graph). comm_residues: iterable of canonical indices in
    the communication set. pair_features: optional {(i,j): {channel: value}} from the pipeline.
    -> dict(communication_edge_index, communication_edge_features [E, 7])."""
    comm = sorted(set(int(i) for i in comm_residues))
    if len(comm) < 2:
        raise ValueError("communication set has <2 residues; cannot build a pocket->DBD graph")
    idx = torch.tensor(comm, dtype=torch.long)
    sub = positions[idx]
    d = torch.cdist(sub, sub)
    ij = (d < cutoff).nonzero(as_tuple=False)
    ij = ij[ij[:, 0] != ij[:, 1]]
    pf = pair_features or {}
    src, dst, feats = [], [], []
    for a, b in ij.tolist():
        gi, gj = comm[a], comm[b]                             # back to canonical indices
        p = pf.get((gi, gj), {})
        have = len(p) > 0
        feats.append([float(d[a, b]), 1.0,
                      float(p.get("resolvent", 0.0)), float(p.get("calibrated_path", 0.0)),
                      float(p.get("contact_churn", 0.0)), float(p.get("hinge_weight", 0.0)),
                      1.0 if have else 0.0])
        src.append(gi); dst.append(gj)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_feat = torch.tensor(feats, dtype=torch.float32)
    return dict(communication_edge_index=edge_index, communication_edge_features=edge_feat)
