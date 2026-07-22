"""Gain losses: the allosteric transmission gain alpha (design distal response projected on the native
direction, normalised by native power) must be USEFUL - not forced to 1, not maximised without bound.

Two regimes:
  - structural pretraining (no functional labels): match the scaffold's native transmission efficiency,
        L_gain = Huber(predicted_gain, native_gain)
    so the model learns how each scaffold amplifies/attenuates a pocket perturbation.
  - functional training (reporter / DNA-affinity labels): keep alpha inside a useful band,
        L_gain = relu(alpha_min - alpha) + relu(alpha - alpha_max)
    too low -> no response; too high -> apo leakage / constitutive-ON / unfolding.
"""
import math

import torch
import torch.nn.functional as F


def gain_pretrain_loss(predicted_gain, native_gain, delta=1.0):
    pg = predicted_gain if torch.is_tensor(predicted_gain) else torch.tensor(float(predicted_gain))
    if not math.isfinite(float(native_gain)):
        return pg * 0.0                                  # no pocket / undefined target: skip, don't NaN
    return F.huber_loss(pg, pg.new_tensor(float(native_gain)), delta=delta)


def gain_band_loss(alpha, alpha_min=0.5, alpha_max=2.0):
    return torch.relu(alpha.new_tensor(alpha_min) - alpha) + torch.relu(alpha - alpha_max)
