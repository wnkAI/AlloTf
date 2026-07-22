"""The AlloTransfer objective, upgraded. Response transfer is no longer a single copy loss but a
direction + gain + off-path split, and apo DNA competence is supervised as a hard constraint against
constitutive-ON:

    L = lf*joint_function + la*apo_repression + lr*DNA_release
        + l_dir*direction + l_gain*gain_band + l_off*off_path
        + lb*target_binding + l_apo*apo_competence + lcons*consistency
        + lrank*within-ligand_rank + lmech*mechanistic

The teacher is the frozen per-residue native_response; direction pins WHERE the response goes, gain
keeps the transmission in a useful band (not forced to 1, not maximised), off-path kills strong
wrong-direction perturbations. Every label is optional per sample and NaN-safe.
"""
import torch
import torch.nn as nn

from .ranking_loss import pairwise_ranking
from .response_direction import direction_loss
from .offpath_loss import offpath_loss
from .gain_loss import gain_band_loss
from ..model.joint_function_head import CLASSES

DEFAULT_LAMBDAS = {"func": 1.0, "apo": 0.7, "release": 1.0, "dir": 0.5, "gain": 0.3, "gainhead": 0.2,
                   "off": 0.3, "bind": 0.3, "apocomp": 0.5, "cons": 0.3, "rank": 0.5, "mech": 0.5}
_SENSOR = CLASSES.index("functional_sensor")
_CONST_ON = CLASSES.index("constitutive_ON")
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
    def __init__(self, lambdas=None, apo_thresh=0.0, ddg_margin=0.5, alpha_min=0.5, alpha_max=2.0):
        super().__init__()
        self.l = dict(DEFAULT_LAMBDAS, **(lambdas or {}))
        self.apo_thresh, self.ddg_margin = apo_thresh, ddg_margin
        self.alpha_min, self.alpha_max = alpha_min, alpha_max

    def forward(self, outputs, labels, mech=None):
        dev = outputs[0]["S_design"].device
        keys = ("func", "apo", "release", "dir", "gain", "gainhead", "off", "bind", "apocomp", "cons")
        acc = {k: torch.zeros((), device=dev) for k in keys}
        cnt = {k: 0 for k in keys}

        for out, lab in zip(outputs, labels):
            if lab.get("y_class") is not None:
                acc["func"] = acc["func"] + nn.functional.cross_entropy(
                    out["class_logits"].unsqueeze(0),
                    out["class_logits"].new_tensor([lab["y_class"]], dtype=torch.long))
                cnt["func"] += 1
                # apo DNA competence: hard constraint - only constitutive_ON should lose apo repression
                target = out["apo_DNA_competence"].new_tensor(0.0 if lab["y_class"] == _CONST_ON else 1.0)
                acc["apocomp"] = acc["apocomp"] + nn.functional.binary_cross_entropy(
                    out["apo_DNA_competence"].clamp(1e-6, 1 - 1e-6), target)
                cnt["apocomp"] += 1
            m = _mse(out["logKd_apo"], lab.get("logKd_apo"))
            if m is not None:
                acc["apo"] = acc["apo"] + m; cnt["apo"] += 1
            m = _mse(out["logKd_target"], lab.get("logKd_target"))
            if m is not None:
                acc["release"] = acc["release"] + m; cnt["release"] += 1

            # response transfer: direction + off-path always; gain kept in a useful band
            args = (out["predicted_native_response"], out["native_response"],
                    out["native_response_mask"], out["native_response_confidence"])
            acc["dir"] = acc["dir"] + direction_loss(*args); cnt["dir"] += 1
            acc["off"] = acc["off"] + offpath_loss(*args); cnt["off"] += 1
            acc["gain"] = acc["gain"] + gain_band_loss(out["allosteric_gain"], self.alpha_min, self.alpha_max)
            cnt["gain"] += 1
            # supervise the coupling-gain head to predict the analytic gain (else it gets no gradient)
            acc["gainhead"] = acc["gainhead"] + nn.functional.huber_loss(
                out["gain_mean"], out["allosteric_gain"].detach())
            cnt["gainhead"] += 1

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
