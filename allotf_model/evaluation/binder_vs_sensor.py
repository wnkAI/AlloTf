"""The metrics that actually test whether the model learned allostery, not just binding.

Plain accuracy is useless here: a binding-only scorer looks great on switch/non-switch if most
non-switches also fail to bind. The two metrics that matter:

  * AUPRC on the functional-sensor class (operational hit-rate for wet-lab picking);
  * AUC discriminating functional sensors from BINDER-ONLY variants (bind fine, never switch) -
    this is the pair a pocket scorer cannot separate and an allosteric model must.

Self-contained (no sklearn) so it runs in any env.
"""
import numpy as np


def auroc(scores, labels):
    """Rank-based AUC (Mann-Whitney). labels in {0,1}. Returns nan if only one class present."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = s.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for k, c in enumerate(counts):
        avg[k] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)


def auprc(scores, labels):
    """Area under precision-recall for the positive class (step/rectangle sum)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    if y.sum() == 0:
        return float("nan")
    order = s.argsort()[::-1]
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def functional_metrics(scores, functional_labels, is_binder=None):
    """scores: S_final per variant. functional_labels: 1 = functional sensor, 0 = not.
    is_binder: optional mask of variants that DO bind the ligand (to isolate sensor vs binder-only).
    """
    out = {"AUPRC_functional": auprc(scores, functional_labels),
           "AUROC_functional": auroc(scores, functional_labels)}
    if is_binder is not None:
        is_binder = np.asarray(is_binder, bool)
        fl = np.asarray(functional_labels, int)
        # among binders: sensor (1) vs binder-only (0)
        sub = is_binder
        if sub.sum() > 1 and 0 < fl[sub].sum() < sub.sum():
            out["AUC_sensor_vs_binderonly"] = auroc(np.asarray(scores)[sub], fl[sub])
        else:
            out["AUC_sensor_vs_binderonly"] = float("nan")
    return out
