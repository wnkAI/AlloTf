"""Constraint-based ranking, not a weighted sum. A very high binding score must NOT compensate for
lost apo DNA binding, a wrong response direction, or a broken pathway. So candidates are first FILTERED
to the feasible set:

    target_binding   > t_bind        (sigmoid of bind_logit)
    apo_DNA_competence > t_apo        (apo still represses)
    response_alignment > t_align      (response points the native way)
    allosteric_gain   > 0             (positive transmission)
    off_path_response < t_offpath     (little wrong-direction perturbation)

then the feasible set is ranked by functional_sensor_probability. Infeasible candidates are returned
with the reasons they failed, never silently dropped.
"""
import math

DEFAULT_THRESHOLDS = {"t_bind": 0.5, "t_apo": 0.5, "t_align": 0.2, "t_offpath": 1.0}


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def feasibility(out, thr=None):
    """out: one model output dict (floats or tensors). -> (is_feasible, reasons list)."""
    t = dict(DEFAULT_THRESHOLDS, **(thr or {}))
    f = lambda k: float(out[k])
    reasons = []
    if _sigmoid(f("target_binding")) <= t["t_bind"]:
        reasons.append("weak_target_binding")
    if f("apo_DNA_competence") <= t["t_apo"]:
        reasons.append("apo_DNA_binding_lost")
    if f("response_alignment") <= t["t_align"]:
        reasons.append("wrong_response_direction")
    if f("allosteric_gain") <= 0:
        reasons.append("no_positive_gain")
    if f("off_path_response") >= t["t_offpath"]:
        reasons.append("off_path_disturbance")
    return (len(reasons) == 0), reasons


def rank_designs(candidates, thr=None):
    """candidates: [(id, output_dict), ...]. -> dict(ranked=[(id, P_sensor), ...] feasible high->low,
    rejected=[(id, reasons), ...])."""
    feasible, rejected = [], []
    for cid, out in candidates:
        ok, reasons = feasibility(out, thr)
        if ok:
            feasible.append((cid, float(out["functional_sensor_probability"])))
        else:
            rejected.append((cid, reasons))
    feasible.sort(key=lambda x: -x[1])
    return {"ranked": feasible, "rejected": rejected}
