"""The AlloTransfer objective, defined once:

    L = lf*joint_function + la*apo_repression + lr*DNA_release + lt*response_transfer
        + lb*target_binding + lcons*consistency + lrank*within-ligand_rank + lphys*mechanistic

The native anchor is gone: the teacher is the frozen per-residue native_response, matched by the
residue-level transfer loss, so nothing is left to anchor. Consistency ties a predicted
functional_sensor to low apo logKd and positive coupling, so a binder-only cannot score high for
free. Every label is optional per sample (read from the TransferSample masks) and NaN-safe.
"""
import torch
import torch.nn as nn

from .ranking_loss import pairwise_ranking
from .response_transfer import transfer_loss
from ..model.joint_function_head import CLASSES

DEFAULT_LAMBDAS = {"func": 1.0, "apo": 0.7, "release": 1.0, "transfer": 0.5,
                   "bind": 0.3, "cons": 0.3, "rank": 0.5, "mech": 0.5}
_SENSOR = CLASSES.index("functional_sensor")
_TIER = {"functional_sensor": 2, "binder_only": 1}          # else 0


def sample_labels(ts):
    """Extract a labels dict from a TransferSample, honouring the explicit label masks."""
    def val(field, mask):
        return float(getattr(ts, field)) if bool(getattr(ts, mask)) else None
    y_class = int(ts.functional_class) if bool(ts.functional_class_label_mask) else None
    tier = None if y_class is None else _TIER.get(CLASSES[y_class], 0)
    return {"y_class": y_class, "logKd_apo": val("dna_kd_apo", "dna_kd_apo_label_mask"),
            "logKd_target": val("dna_kd_ligand", "dna_kd_ligand_label_mask"),
            "bind": val("ligand_binding", "ligand_binding_label_mask"),
            "tier": tier, "ligand_id": ts.ligand_id}


def _mse(pred, y):
    if y is None or (isinstance(y, float) and y != y):
        return None
    return (pred - pred.new_tensor(float(y))) ** 2


class AlloTransferLoss(nn.Module):
    def __init__(self, lambdas=None, apo_thresh=0.0, ddg_margin=0.5):
        super().__init__()
        self.l = dict(DEFAULT_LAMBDAS, **(lambdas or {}))
        self.apo_thresh, self.ddg_margin = apo_thresh, ddg_margin

    def forward(self, outputs, labels, mech=None):
        dev = outputs[0]["S_design"].device
        keys = ("func", "apo", "release", "transfer", "bind", "cons")
        acc = {k: torch.zeros((), device=dev) for k in keys}
        cnt = {k: 0 for k in keys}

        for out, lab in zip(outputs, labels):
            if lab.get("y_class") is not None:
                acc["func"] = acc["func"] + nn.functional.cross_entropy(
                    out["class_logits"].unsqueeze(0),
                    out["class_logits"].new_tensor([lab["y_class"]], dtype=torch.long))
                cnt["func"] += 1
            m = _mse(out["logKd_apo"], lab.get("logKd_apo"))
            if m is not None:
                acc["apo"] = acc["apo"] + m; cnt["apo"] += 1
            m = _mse(out["logKd_target"], lab.get("logKd_target"))
            if m is not None:
                acc["release"] = acc["release"] + m; cnt["release"] += 1

            acc["transfer"] = acc["transfer"] + transfer_loss(
                out["predicted_native_response"], out["native_response"],
                out["native_response_mask"], out["native_response_confidence"])
            cnt["transfer"] += 1

            if lab.get("bind") is not None:
                acc["bind"] = acc["bind"] + nn.functional.binary_cross_entropy_with_logits(
                    out["bind_logit"], out["bind_logit"].new_tensor(float(lab["bind"])))
                cnt["bind"] += 1
            if lab.get("y_class") == _SENSOR:
                acc["cons"] = acc["cons"] + torch.relu(out["logKd_apo"] - self.apo_thresh) \
                    + torch.relu(self.ddg_margin - out["ddG_coupling"])
                cnt["cons"] += 1

        for k in keys:
            if cnt[k]:
                acc[k] = acc[k] / cnt[k]
        acc["rank"] = pairwise_ranking(torch.stack([o["S_design"] for o in outputs]),
                                       [lab.get("ligand_id") for lab in labels],
                                       [lab.get("tier") for lab in labels])
        acc["mech"] = mech if mech is not None else torch.zeros((), device=dev)
        total = sum(self.l[k] * acc[k] for k in acc)
        return total, {k: float(v.detach()) for k, v in acc.items()}
