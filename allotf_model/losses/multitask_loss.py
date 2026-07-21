"""One loss, defined once: task BCEs + within-TF ranking + state contrast + mechanistic constraints.

    L = lb*L_bind + lp*L_path + ls*L_switch + lr*L_rank + lc*L_contrast + lm*L_mech

Every task label is optional per sample (masked): a variant may have a functional switch label but no
binding assay. Missing labels contribute nothing rather than a fabricated zero.
"""
import torch
import torch.nn as nn

from .ranking_loss import pairwise_ranking
from .contrastive_loss import StateContrastLoss
from .mechanistic_constraints import mechanistic_penalty

DEFAULT_LAMBDAS = {"bind": 1.0, "path": 1.0, "switch": 1.0, "rank": 0.5, "contrast": 0.2, "mech": 0.5}


def _bce(logit, y):
    if y is None or (isinstance(y, float) and y != y):        # None or nan
        return None
    target = logit.new_tensor(float(y))
    return nn.functional.binary_cross_entropy_with_logits(logit, target)


class MultiTaskLoss(nn.Module):
    def __init__(self, h_dim, lambdas=None):
        super().__init__()
        self.l = dict(DEFAULT_LAMBDAS, **(lambdas or {}))
        self.contrast = StateContrastLoss(h_dim)

    def forward(self, outputs, labels):
        """outputs/labels: parallel lists (one per sample). labels keys: bind, path, switch, tier,
        tf_id, dna_compat_apo, dna_compat_lig, topology_sign."""
        dev = outputs[0]["S_final"].device
        acc = {k: torch.zeros((), device=dev) for k in ("bind", "path", "switch", "contrast", "mech")}
        cnt = {k: 0 for k in ("bind", "path", "switch")}

        for out, lab in zip(outputs, labels):
            for key, logit_key in (("bind", "bind"), ("path", "path"), ("switch", "func")):
                y = lab.get(key)
                lval = _bce(out["logits"][logit_key], y)
                if lval is not None:
                    acc[key] = acc[key] + lval
                    cnt[key] += 1
            acc["contrast"] = acc["contrast"] + self.contrast(out["pooled"])
            acc["mech"] = acc["mech"] + mechanistic_penalty(
                out, lab.get("dna_compat_apo"), lab.get("dna_compat_lig"))

        for k in ("bind", "path", "switch"):
            if cnt[k]:
                acc[k] = acc[k] / cnt[k]
        n = len(outputs)
        acc["contrast"] = acc["contrast"] / n
        acc["mech"] = acc["mech"] / n

        scores = torch.stack([o["S_final"] for o in outputs])
        tf_ids = [lab.get("tf_id") for lab in labels]        # None -> excluded from ranking
        tiers = [lab.get("tier") for lab in labels]
        acc["rank"] = pairwise_ranking(scores, tf_ids, tiers)

        total = sum(self.l[k] * acc[k] for k in acc)
        return total, {k: float(v) for k, v in acc.items()}
