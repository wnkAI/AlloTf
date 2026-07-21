"""Mechanistic regularisation via shuffle controls, with a DIRECTIONAL expectation (not just "must
differ", which has zero gradient at equality and accepts change in any direction).

  * shuffle the ligand -> the real target ligand must score as MORE of a sensor than a random ligand:
    S_design(real) should exceed S_design(shuffled) by a margin;
  * shuffle the pocket->DBD path -> breaking the path must REDUCE both the coupling and the sensor
    score: real should exceed shuffled by a margin.

Each term is a hinge on the signed difference, so it keeps a gradient even as the model approaches
the (wrong) shortcut of ignoring the ligand or the path. The trainer supplies the perturbed runs.
"""
import torch


def _exceed(better, worse, margin):
    """Penalise unless `better` beats `worse` by `margin` (hinge, gradient survives at equality)."""
    return torch.relu(margin - (better - worse))


def mechanistic_penalty(out_real, out_shuffled_ligand=None, out_shuffled_path=None, margin=0.2):
    pen = out_real["S_design"].new_zeros(())
    if out_shuffled_ligand is not None:
        pen = pen + _exceed(out_real["S_design"], out_shuffled_ligand["S_design"], margin)
    if out_shuffled_path is not None:
        pen = pen + _exceed(out_real["ddG_coupling"], out_shuffled_path["ddG_coupling"], margin)
        pen = pen + _exceed(out_real["S_design"], out_shuffled_path["S_design"], margin)
    return pen
