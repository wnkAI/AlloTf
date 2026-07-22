"""Structural-pretraining target: distal transmission gain. From a scaffold's apo->holo response field,

    g = ||response_distal|| / (||response_pocket|| + eps)

teaches the shared body how each scaffold AMPLIFIES or ATTENUATES a pocket perturbation on its way to
the DBD - the cross-scaffold prior the design-time gain head needs. No functional labels required.
"""
import torch


def distal_gain_target(response, distal_mask, pocket_idx, eps=1e-6):
    """response: [N_res, D] native apo->holo field. -> scalar g (nan if no pocket residues)."""
    dist = response[distal_mask.bool()].norm()
    pk_idx = torch.as_tensor(list(pocket_idx), dtype=torch.long)
    if pk_idx.numel() == 0:
        return float("nan")
    return float(dist / (response[pk_idx].norm() + eps))
