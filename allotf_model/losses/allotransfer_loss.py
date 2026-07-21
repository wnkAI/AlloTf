"""The AlloTransfer objective, defined once:

    L = lf*joint_function + la*apo_repression + lr*DNA_release + lt*response_transfer + ln*native_anchor
        + lb*target_binding + lcons*consistency + lrank*within-ligand_rank + lphys*mechanistic

Main tasks: apo keeps DNA (low logKd_apo), target binds, target lowers DNA affinity (ddG>0), and the
distal response reproduces the native ligand's. native_anchor keeps the EMA teacher a REAL release
(native ligand releases DNA), preventing collapse to a trivial teacher. consistency ties a predicted
functional_sensor to low apo Kd and positive coupling, so binder-only cannot score high for free.

Affinity data (Kd/EMSA/MST/BLI/SPR) supervises logKd_apo/target directly; reporter data supervises the
class - reporter values are never treated as absolute Kd. Binding is a SEPARATE pocket-derived logit
(recognition), not read off the class head. Every label is optional per sample and NaN-safe.
"""
import torch
import torch.nn as nn

from .ranking_loss import pairwise_ranking
from .response_transfer import transfer_loss
from ..model.joint_function_head import CLASSES

DEFAULT_LAMBDAS = {"func": 1.0, "apo": 0.7, "release": 1.0, "transfer": 0.5, "native": 0.5,
                   "bind": 0.3, "cons": 0.3, "rank": 0.5, "mech": 0.5}
_SENSOR = CLASSES.index("functional_sensor")


def _has(y):
    if y is None:
        return False
    try:
        return not (isinstance(y, float) and y != y) and torch.isfinite(torch.tensor(float(y)))
    except (TypeError, ValueError):
        return False


def _mse(pred, y):
    return (pred - pred.new_tensor(float(y))) ** 2 if _has(y) else None


class AlloTransferLoss(nn.Module):
    def __init__(self, lambdas=None, native_margin=1.0, apo_thresh=0.0, ddg_margin=0.5):
        super().__init__()
        self.l = dict(DEFAULT_LAMBDAS, **(lambdas or {}))
        self.native_margin, self.apo_thresh, self.ddg_margin = native_margin, apo_thresh, ddg_margin

    def forward(self, outputs, labels, mech=None):
        dev = outputs[0]["S_design"].device
        keys = ("func", "apo", "release", "transfer", "native", "bind", "cons")
        acc = {k: torch.zeros((), device=dev) for k in keys}
        cnt = {k: 0 for k in keys}

        for out, lab in zip(outputs, labels):
            if _has(lab.get("y_class")):
                acc["func"] = acc["func"] + nn.functional.cross_entropy(
                    out["class_logits"].unsqueeze(0),
                    out["class_logits"].new_tensor([int(lab["y_class"])], dtype=torch.long))
                cnt["func"] += 1
            for tgt, key in (("logKd_apo", "logKd_apo"), ("logKd_target", "logKd_target")):
                m = _mse(out[tgt], lab.get(key))
                if m is not None:
                    bucket = "apo" if tgt == "logKd_apo" else "release"
                    acc[bucket] = acc[bucket] + m; cnt[bucket] += 1
            m = _mse(out["ddG_coupling"], lab.get("ddG_coupling"))
            if m is not None:
                acc["release"] = acc["release"] + m; cnt["release"] += 1

            acc["transfer"] = acc["transfer"] + transfer_loss(out["dH_target"], out["dH_native"])
            cnt["transfer"] += 1
            # native anchor: the teacher's native response must be a real release (ddG >= margin)
            acc["native"] = acc["native"] + torch.relu(self.native_margin - out["native_ddG"])
            cnt["native"] += 1

            if _has(lab.get("bind")):
                acc["bind"] = acc["bind"] + nn.functional.binary_cross_entropy_with_logits(
                    out["bind_logit"], out["bind_logit"].new_tensor(float(lab["bind"])))
                cnt["bind"] += 1
            # consistency: a predicted sensor must keep the apo repressed and give positive coupling
            if _has(lab.get("y_class")) and int(lab["y_class"]) == _SENSOR:
                acc["cons"] = acc["cons"] + torch.relu(out["logKd_apo"] - self.apo_thresh) \
                    + torch.relu(self.ddg_margin - out["ddG_coupling"])
                cnt["cons"] += 1

        for k in keys:
            if cnt[k]:
                acc[k] = acc[k] / cnt[k]

        scores = torch.stack([o["S_design"] for o in outputs])
        acc["rank"] = pairwise_ranking(scores, [lab.get("ligand_id") for lab in labels],
                                       [lab.get("tier") for lab in labels])
        acc["mech"] = mech if mech is not None else torch.zeros((), device=dev)

        total = sum(self.l[k] * acc[k] for k in acc)
        return total, {k: float(v) for k, v in acc.items()}
