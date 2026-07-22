"""Shared response-transfer maths. The design's distal response is decomposed against the FROZEN
native response as

    dH_target_distal = alpha * dH_native_distal + eps_perp

in the teacher's fixed physical channels (each normalised by a fixed scale so mixed units - CA
displacement in A, contact-count change, neighbour counts - are comparable):

    alpha    = weighted projection of the design response onto the native direction / native power
               -> the allosteric transmission GAIN (how strongly the pocket perturbation reaches the
                  distal region, in the native direction);
    eps_perp = the component OFF the native direction (ineffective or destructive perturbation).

The three transfer losses (direction / gain / off-path, in their own modules) all read this
decomposition. Fails closed: needs >= 2 valid distal teacher residues.
"""
import torch
import torch.nn.functional as F

# fixed per-channel scales for native_reference.CHANNELS
# (ca_displacement A, contact_count_change, n_neighbours_apo, n_neighbours_holo)
CHANNEL_SCALES = (3.0, 5.0, 12.0, 12.0)


def decompose(target, native, mask, confidence):
    """target/native: [N_res, D]. mask/confidence: [N_res]. -> dict(alpha, eps_perp [M,D], alignment,
    w [M], T [M,D], N [M,D], idx). Fails closed on < 2 valid residues."""
    m = mask.bool() & (confidence > 0)
    if int(m.sum()) < 2:
        raise ValueError("response-transfer has < 2 valid distal teacher residues: fail closed")
    scale = target.new_tensor(CHANNEL_SCALES[:target.shape[1]])
    T, N, w = target[m] / scale, native[m] / scale, confidence[m]
    power = (w * (N * N).sum(1)).sum()                   # weighted native power
    if float(power) < 1e-8:
        raise ValueError("native response power ~ 0: direction/gain undefined, fail closed")
    alpha = (w * (T * N).sum(1)).sum() / power           # exact denom -> eps_perp is weighted-orthogonal
    eps = T - alpha * N
    tw = (w.sqrt().unsqueeze(1) * T).reshape(-1)
    nw = (w.sqrt().unsqueeze(1) * N).reshape(-1)
    alignment = (tw @ nw) / (tw.norm() * nw.norm() + 1e-6)
    return {"alpha": alpha, "eps_perp": eps, "alignment": alignment, "w": w, "T": T, "N": N, "idx": m}


def _mean_channel_corr(p, t):
    pc, tc = p - p.mean(0, keepdim=True), t - t.mean(0, keepdim=True)
    den = pc.norm(dim=0) * tc.norm(dim=0) + 1e-6
    return ((pc * tc).sum(0) / den).mean()


def transfer_loss(pred, target, residue_mask, confidence, lambda_pattern=0.5, huber_delta=1.0):
    """Legacy single response-copy loss (magnitude + pattern). Kept for the plain retrospective mode;
    the direction/gain/off-path split is the upgrade. Fails closed on empty."""
    m = residue_mask.bool() & (confidence > 0)
    if int(m.sum()) < 2:
        raise ValueError("response-transfer has <2 valid teacher residues: fail closed")
    scale = pred.new_tensor(CHANNEL_SCALES[:pred.shape[1]])
    p, t = pred[m] / scale, target[m] / scale
    w = confidence[m].unsqueeze(1)
    huber = (F.huber_loss(p, t, reduction="none", delta=huber_delta) * w).sum() / (w.sum() * p.shape[1])
    return huber + lambda_pattern * (1.0 - _mean_channel_corr(p, t))
