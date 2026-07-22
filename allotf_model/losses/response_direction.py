"""Direction loss: the new ligand's distal perturbation must point ALONG the native DNA-release
response, regardless of magnitude.

    L_direction = 1 - cos(dH_target_distal, dH_native_distal)

Direction is what makes the response functional; magnitude is handled separately by the gain term.
"""
from .response_transfer import decompose


def direction_loss(target, native, mask, confidence):
    return 1.0 - decompose(target, native, mask, confidence)["alignment"]
