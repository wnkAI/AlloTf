"""Residue-level response-transfer loss: the ligand-induced distal response of the DESIGN, predicted
per residue in the native teacher's physical channels, must match the FROZEN native response over the
valid distal residues - in magnitude AND spatial pattern.

    L = weighted_Huber(pred, target) + lambda_pattern * (1 - mean_channel_correlation)

The teacher (TransferSample.native_response) is a per-residue physical descriptor with mixed units
(CA displacement in A, contact-count change, neighbour counts), so each channel is normalised by a
FIXED scale before comparison, weighted by the per-residue teacher confidence, and missing/zero-
confidence residues are fully masked. Huber pins the absolute magnitude; the correlation term pins
WHICH hinge/interface/DBD residues respond - the whole point of the canonical residue alignment.
Fails closed: raises if no valid teacher residue survives the mask.
"""
import torch
import torch.nn.functional as F

# fixed per-channel scales for native_reference.CHANNELS
# (ca_displacement A, contact_count_change, n_neighbours_apo, n_neighbours_holo)
CHANNEL_SCALES = (3.0, 5.0, 12.0, 12.0)


def _mean_channel_corr(p, t):
    pc, tc = p - p.mean(0, keepdim=True), t - t.mean(0, keepdim=True)
    den = pc.norm(dim=0) * tc.norm(dim=0) + 1e-6
    return ((pc * tc).sum(0) / den).mean()


def transfer_loss(pred, target, residue_mask, confidence, lambda_pattern=0.5, huber_delta=1.0):
    """pred/target: [N_res, D]. residue_mask/confidence: [N_res]. -> scalar (fails closed on empty)."""
    m = residue_mask.bool() & (confidence > 0)
    if int(m.sum()) < 2:                     # need >=2 residues for a pattern correlation
        raise ValueError("response-transfer has <2 valid teacher residues (empty distal mask or zero "
                         "confidence): fail closed rather than train on a meaningless target")
    scale = pred.new_tensor(CHANNEL_SCALES[:pred.shape[1]])
    p, t = pred[m] / scale, target[m] / scale
    w = confidence[m].unsqueeze(1)
    huber = (F.huber_loss(p, t, reduction="none", delta=huber_delta) * w).sum() / (w.sum() * p.shape[1])
    return huber + lambda_pattern * (1.0 - _mean_channel_corr(p, t))
