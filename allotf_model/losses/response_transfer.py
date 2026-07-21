"""Response-transfer loss: the ligand-induced distal state change of the DESIGN must reproduce the
NATIVE ligand's distal change - in DIRECTION and MAGNITUDE, over the hinge->DBD region only.

    L = (1 - cos(dH_target, dH_native)) + lambda_mag * | ||dH_target|| - ||dH_native|| |

Cosine alone would let the model cheat by shrinking dH_target to ~0 (any direction matches a tiny
vector cheaply); the magnitude term forbids that. dH_native is the frozen teacher (detached), so the
model moves toward it, not the other way. Matching is distal ONLY - the new ligand's pocket chemistry
is free to differ.
"""
import torch


def transfer_loss(dH_target, dH_native, lambda_mag=0.5, eps=1e-6):
    """dH_target / dH_native: (D,) pooled distal response vectors (native already detached)."""
    cos = torch.nn.functional.cosine_similarity(dH_target, dH_native, dim=0, eps=eps)
    mag = (dH_target.norm() - dH_native.norm()).abs()
    return (1.0 - cos) + lambda_mag * mag
