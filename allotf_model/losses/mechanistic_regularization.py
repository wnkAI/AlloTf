"""Mechanistic regularisation via shuffle controls - forces the model to actually USE the ligand
identity and the pocket->DBD path instead of reading the answer off the sequence.

The trainer re-runs the model on perturbed copies and passes the outputs here:
  * shuffle the ligand -> the coupling / design score MUST change (ligand specificity);
  * shuffle the pocket->DBD communication edges -> the coupling MUST change (the response really goes
    through the path).
A model that ignores the ligand or the path pays a penalty. The sequence-only negative control (no
ligand at all) is run by the evaluation harness.
"""
import torch


def _must_differ(a, b, tol):
    return torch.relu(tol - (a - b).abs())


def mechanistic_penalty(out_real, out_shuffled_ligand=None, out_shuffled_path=None, tol=0.5):
    pen = out_real["ddG_coupling"].new_zeros(())
    if out_shuffled_ligand is not None:
        pen = pen + _must_differ(out_real["ddG_coupling"], out_shuffled_ligand["ddG_coupling"], tol)
        pen = pen + _must_differ(out_real["S_design"], out_shuffled_ligand["S_design"], tol * 0.2)
    if out_shuffled_path is not None:
        pen = pen + _must_differ(out_real["ddG_coupling"], out_shuffled_path["ddG_coupling"], tol)
    return pen
