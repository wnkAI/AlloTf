"""One confidence convention for the whole bridge, so no extractor invents its own range.

Every confidence is a float in [0,1]: 1 = experimental / certain, 0 = absent / unusable. Sources
(structural, residue-mapping, ligand-pose, physics, label availability, teacher availability) are all
produced through here and combined the same way, so the model can down-weight modelled or missing
inputs uniformly.
"""
import torch

SOURCES = ("structural", "mapping", "pose", "physics", "label", "teacher")


def clamp01(x):
    return torch.as_tensor(x, dtype=torch.float32).clamp(0.0, 1.0)


def from_plddt(plddt):
    """pLDDT (0-100) -> [0,1] structural confidence."""
    return clamp01(torch.as_tensor(plddt, dtype=torch.float32) / 100.0)


def availability(values):
    """A per-element availability mask: finite -> 1, NaN/inf -> 0."""
    v = torch.as_tensor(values, dtype=torch.float32)
    return torch.isfinite(v).float()


def combine(*confidences):
    """Combine independent confidences multiplicatively (a weak link drags the product down)."""
    out = None
    for c in confidences:
        c = clamp01(c)
        out = c if out is None else out * c
    return out if out is not None else torch.ones(())
