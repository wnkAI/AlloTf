"""The AlloTransfer objective, defined once:

    L = lf*joint_function + la*apo_repression + lr*DNA_release + lt*response_transfer
        + lb*target_binding + lrank*within-ligand_rank + lphys*mechanistic

Main tasks: apo keeps DNA (low logKd_apo), target ligand binds, target lowers DNA affinity
(ddG_coupling>0), and the distal response reproduces the native ligand's. Every label is optional per
sample (masked); missing labels contribute nothing rather than a fabricated zero. Affinity data
(Kd/EMSA/MST/BLI/SPR) supervises logKd directly; reporter data supervises the class - reporter values
are never treated as absolute Kd. Binding is auxiliary, read off the class head as 1 - P(non_binder).
"""
import torch
import torch.nn as nn

from .ranking_loss import pairwise_ranking
from .response_transfer import transfer_loss
from ..model.joint_function_head import CLASSES

DEFAULT_LAMBDAS = {"func": 1.0, "apo": 0.7, "release": 1.0, "transfer": 0.5,
                   "bind": 0.3, "rank": 0.5, "mech": 0.5}
_NONBIND = CLASSES.index("non_binder")


def _mse(pred, y):
    if y is None or (isinstance(y, float) and y != y):
        return None
    return (pred - pred.new_tensor(float(y))) ** 2


class AlloTransferLoss(nn.Module):
    def __init__(self, lambdas=None):
        super().__init__()
        self.l = dict(DEFAULT_LAMBDAS, **(lambdas or {}))

    def forward(self, outputs, labels, mech=None):
        dev = outputs[0]["S_design"].device
        acc = {k: torch.zeros((), device=dev) for k in ("func", "apo", "release", "transfer", "bind")}
        cnt = {k: 0 for k in acc}

        for out, lab in zip(outputs, labels):
            if lab.get("y_class") is not None:
                acc["func"] = acc["func"] + nn.functional.cross_entropy(
                    out["class_logits"].unsqueeze(0),
                    out["class_logits"].new_tensor([lab["y_class"]], dtype=torch.long))
                cnt["func"] += 1
            a = _mse(out["logKd_apo"], lab.get("logKd_apo"))
            if a is not None:
                acc["apo"] = acc["apo"] + a; cnt["apo"] += 1
            r = _mse(out["ddG_coupling"], lab.get("ddG_coupling"))
            if r is not None:
                acc["release"] = acc["release"] + r; cnt["release"] += 1
            # response transfer is always defined (native teacher present)
            acc["transfer"] = acc["transfer"] + transfer_loss(out["dH_target"], out["dH_native"])
            cnt["transfer"] += 1
            yb = lab.get("bind")
            if yb is not None:
                p_bind = (1.0 - out["class_probs"][_NONBIND]).clamp(1e-6, 1 - 1e-6)
                acc["bind"] = acc["bind"] + nn.functional.binary_cross_entropy(
                    p_bind, p_bind.new_tensor(float(yb)))
                cnt["bind"] += 1

        for k in acc:
            if cnt[k]:
                acc[k] = acc[k] / cnt[k]

        scores = torch.stack([o["S_design"] for o in outputs])
        groups = [lab.get("ligand_id") for lab in labels]
        tiers = [lab.get("tier") for lab in labels]
        acc["rank"] = pairwise_ranking(scores, groups, tiers)
        acc["mech"] = mech if mech is not None else torch.zeros((), device=dev)

        total = sum(self.l[k] * acc[k] for k in acc)
        return total, {k: float(v) for k, v in acc.items()}
