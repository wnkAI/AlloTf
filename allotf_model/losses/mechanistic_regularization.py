"""Mechanistic regularisation via shuffle controls - forces the model to actually USE the ligand
identity and the pocket->DBD path, instead of reading the answer off the sequence.

The trainer re-runs the model on perturbed copies of each sample and passes the outputs here:
  * shuffle the ligand  -> the prediction MUST change (ligand specificity);
  * shuffle the pocket->DBD communication edges -> the prediction MUST change (the coupling really
    goes through the path).
A model that ignores the ligand or the path pays a penalty. The sequence-only negative control (no
ligand input at all) is run by the evaluation harness, not penalised here.
"""
import torch


def _must_differ(a, b, tol):
    """Penalise when |a - b| is smaller than tol (i.e. the perturbation did nothing)."""
    return torch.relu(tol - (a - b).abs())


def mechanistic_penalty(out_real, out_shuffled_ligand=None, out_shuffled_path=None, tol=0.5):
    pen = out_real["M_switch"].new_zeros(())
    if out_shuffled_ligand is not None:
        pen = pen + _must_differ(out_real["M_switch"], out_shuffled_ligand["M_switch"], tol)
        pen = pen + _must_differ(out_real["S_final"], out_shuffled_ligand["S_final"], tol * 0.2)
    if out_shuffled_path is not None:
        pen = pen + _must_differ(out_real["M_switch"], out_shuffled_path["M_switch"], tol)
    return pen
