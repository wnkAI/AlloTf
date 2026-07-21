"""Differentiable mechanistic constraints - physics the model may not violate, enforced as penalties
rather than hoped for.

  * functional probability may not exceed binding capability (no switch without a binder);
  * functional probability may not exceed path integrity (no switch through a broken path);
  * for a release-type TF, a ligand state that still binds DNA well must not earn a positive switch
    margin (if the ligand state is DNA-compatible, it did not release);
  * an apo state that binds DNA well should earn a positive release margin.

The DNA-compatibility terms use pre-computed features when present and are skipped (not faked) when
absent. dna_compat_* are scalars in [0,1] (fraction of DNA-compatible conformations for that state).
"""
import torch


def mechanistic_penalty(out, dna_compat_apo=None, dna_compat_lig=None, topology_sign=1):
    p_bind, p_path, p_func = out["P_bind"], out["P_path"], out["P_functional"]
    m = out["M_switch"]

    # no compensation: a switch cannot be called without a binder or through a broken path
    pen = torch.relu(p_func - p_bind) + torch.relu(p_func - p_path)

    if topology_sign > 0:                     # release-type TF
        if dna_compat_lig is not None:
            # ligand state still DNA-compatible AND positive margin -> contradiction
            pen = pen + torch.relu(m) * dna_compat_lig
        if dna_compat_apo is not None:
            # apo state DNA-compatible should support a positive (release) margin
            pen = pen + torch.relu(-m) * dna_compat_apo
    return pen
