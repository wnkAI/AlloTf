"""Off-path loss: penalise the part of the design's distal response that does NOT lie along the native
direction. Without this, the model can score a strong response that is globally wrong - a large
structural perturbation in the wrong direction that would not release DNA (or would break the fold).

    L_offpath = sum_i w_i * ||eps_perp_i||^2   (per-residue confidence weighted, over distal residues)
"""
from .response_transfer import decompose


def offpath_loss(target, native, mask, confidence):
    d = decompose(target, native, mask, confidence)
    eps, w = d["eps_perp"], d["w"].unsqueeze(1)
    return (w * eps ** 2).sum() / (w.sum() * eps.shape[1])
