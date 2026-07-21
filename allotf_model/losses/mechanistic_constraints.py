"""Differentiable mechanistic constraints - physics the model may not violate, enforced as penalties.

  * functional probability may not exceed binding capability (no switch without a binder);
  * functional probability may not exceed path integrity (no switch through a broken path);
  * the switch margin must ALIGN with the observed DNA-compatibility difference between states:
    for a release TF, apo-more-compatible-than-lig should give a positive margin; for an enhanced-
    binding TF the direction flips. This is a directional penalty, so its minimum is the physically
    correct sign, not m=0.

The DNA-compatibility terms need both fractions; they are skipped (not faked) when either is absent.
topology_sign comes from the model output so there is one source of truth. c_* in [0,1].
"""
import torch


def mechanistic_penalty(out, dna_compat_apo=None, dna_compat_lig=None):
    p_bind, p_path, p_func = out["P_bind"], out["P_path"], out["P_functional"]
    m = out["M_switch"]
    topology_sign = out.get("topology_sign", 1)

    # no compensation: a switch cannot be called without a binder or through a broken path
    pen = torch.relu(p_func - p_bind) + torch.relu(p_func - p_path)

    if dna_compat_apo is not None and dna_compat_lig is not None:
        # m should share the sign of topology_sign*(c_apo - c_lig); penalise disagreement, weighted by
        # how clear the compatibility difference is
        target = topology_sign * (float(dna_compat_apo) - float(dna_compat_lig))
        pen = pen + torch.relu(-(m * target))
    return pen
