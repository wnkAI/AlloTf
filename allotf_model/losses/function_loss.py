"""The main supervision, driven by the direct transcriptional-function labels (Sensor-seq F-score and
functional class), not by any Rosetta delta.

    L = lF*F-score reg + lC*class CE + lR*repression reg + lrank*within-ligand ranking
        + lB*binding aux + lphys*mechanistic

Every label is optional per sample (masked). F-score supervises M_switch = R_lig - R_apo; the class
head supervises {functional_sensor, constitutive_ON, constitutive_OFF, non_responsive}; the repression
terms anchor R_apo / R_lig individually so their difference is meaningful.
"""
import torch
import torch.nn as nn

from .ranking_loss import pairwise_ranking
from .binding_auxiliary import binding_loss

DEFAULT_LAMBDAS = {"fscore": 1.0, "class": 1.0, "repression": 0.5, "rank": 0.5, "bind": 0.3, "mech": 0.5}


def _mse(pred, y):
    if y is None or (isinstance(y, float) and y != y):
        return None
    return (pred - pred.new_tensor(float(y))) ** 2


class FunctionLoss(nn.Module):
    def __init__(self, lambdas=None):
        super().__init__()
        self.l = dict(DEFAULT_LAMBDAS, **(lambdas or {}))

    def forward(self, outputs, labels, mech=None):
        """outputs/labels: parallel lists. label keys: fscore, y_class(int), r_apo, r_lig, bind,
        ligand_id, tier. mech: optional pre-computed mechanistic penalty (scalar tensor)."""
        dev = outputs[0]["S_final"].device
        acc = {k: torch.zeros((), device=dev) for k in ("fscore", "class", "repression", "bind")}
        cnt = {k: 0 for k in acc}

        for out, lab in zip(outputs, labels):
            f = _mse(out["M_switch"], lab.get("fscore"))
            if f is not None:
                acc["fscore"] = acc["fscore"] + f; cnt["fscore"] += 1
            if lab.get("y_class") is not None:
                acc["class"] = acc["class"] + nn.functional.cross_entropy(
                    out["class_logits"].unsqueeze(0), out["class_logits"].new_tensor([lab["y_class"]], dtype=torch.long))
                cnt["class"] += 1
            for key, tgt in (("r_apo", "R_apo"), ("r_lig", "R_lig")):
                r = _mse(out[tgt], lab.get(key))
                if r is not None:
                    acc["repression"] = acc["repression"] + r; cnt["repression"] += 1
            b = binding_loss(out.get("bind_logit"), lab.get("bind"))
            if b is not None:
                acc["bind"] = acc["bind"] + b; cnt["bind"] += 1

        for k in acc:
            if cnt[k]:
                acc[k] = acc[k] / cnt[k]

        scores = torch.stack([o["S_final"] for o in outputs])
        groups = [lab.get("ligand_id") for lab in labels]        # rank WITHIN one ligand
        tiers = [lab.get("tier") for lab in labels]
        acc["rank"] = pairwise_ranking(scores, groups, tiers)
        acc["mech"] = mech if mech is not None else torch.zeros((), device=dev)

        total = sum(self.l[k] * acc[k] for k in acc)
        return total, {k: float(v) for k, v in acc.items()}
