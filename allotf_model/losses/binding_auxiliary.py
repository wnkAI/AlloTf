"""Auxiliary ligand-binding supervision. Binding is a helper task, NOT a factor multiplied into the
final score: it shapes the pocket representation without vetoing candidates. Optional per sample."""
import torch.nn as nn


def binding_loss(bind_logit, y_bind):
    if bind_logit is None or y_bind is None or (isinstance(y_bind, float) and y_bind != y_bind):
        return None
    return nn.functional.binary_cross_entropy_with_logits(bind_logit, bind_logit.new_tensor(float(y_bind)))
