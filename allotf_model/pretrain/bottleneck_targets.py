"""Structural-pretraining target: bottleneck localisation. Which residues, if broken, collapse the
pocket->DBD signal? Computed as communicability betweenness restricted to pocket->DBD pairs on the
communication graph:

    score_v = sum_{p in pocket, d in DBD}  G[p, v] * G[v, d] / G[p, d],   G = exp(A)

Residues on many pocket->DBD communication paths score high. These are the candidate gain-tuning
residues (where a constrained transduction mutation can retune amplification) and the targets a
mask-and-predict pretraining task uses. Uses matrix exponential of the (small) communication adjacency
- no networkx dependency.
"""
import torch


def communicability_bottleneck(n_res, comm_edge_index, pocket_idx, dbd_idx):
    """-> bottleneck_score [n_res] in [0,1], normalised. Zero if no communication edges/regions."""
    score = torch.zeros(n_res)
    pk = torch.as_tensor(list(pocket_idx), dtype=torch.long)
    db = torch.as_tensor(list(dbd_idx), dtype=torch.long)
    if comm_edge_index.numel() == 0 or pk.numel() == 0 or db.numel() == 0:
        return score
    A = torch.zeros(n_res, n_res)
    A[comm_edge_index[0], comm_edge_index[1]] = 1.0
    A = torch.maximum(A, A.t())                         # undirected communication
    G = torch.matrix_exp(A)
    denom = G[pk][:, db] + 1e-6                          # [P, D] pocket->dbd communicability
    for v in range(n_res):
        # paths p -> v -> d through node v, summed over pocket/dbd pairs
        contrib = (G[pk, v].unsqueeze(1) * G[v, db].unsqueeze(0)) / denom
        score[v] = contrib.sum()
    score[pk] = 0.0; score[db] = 0.0                    # endpoints are not bottlenecks
    m = score.max()
    return score / m if m > 0 else score
