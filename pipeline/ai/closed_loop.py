"""Closed loop: plate feedback -> update GP -> select the next batch. Replaces active_learning.py.

experiment_io (read the plate) -> background-subtracted F(c,t) -> response_gp (refit on every
observation so far) -> bandit (Thompson-select the next batch from the untested pool). Project state
persists as JSON, so `design`, `feedback` and `select-next` are three separate CLI calls operating
on one project rather than one long-running process.
"""
import json
import os

import numpy as np

from . import experiment_io as eio
from . import response_gp as rgp
from . import bandit as bd

STATE_FILE = "project_state.json"


def _state_path(project):
    return os.path.join(project, STATE_FILE)


def _save(project, state):
    with open(_state_path(project), "w") as f:
        json.dump(state, f, indent=2)


def load(project):
    p = _state_path(project)
    if not os.path.exists(p):
        raise FileNotFoundError("no project at %s - run `design` first" % project)
    with open(p) as f:
        return json.load(f)


def init_project(project, candidates, conc, time, basal_max=None, initial_ids=None):
    """candidates: {cid: {sequence, mech, mut, scaffold}} - the WHOLE pool that passed the physical
    gates, not just the first plate. initial_ids marks which of them go on plate 1; the rest stay in
    the project as the untested pool select-next draws round two from.
    conc / time: the plate grid every candidate is measured on. -> the saved state."""
    os.makedirs(project, exist_ok=True)
    state = {"candidates": candidates,
             "initial_ids": list(initial_ids) if initial_ids else list(candidates),
             "observations": [],
             "grid": {"conc": [float(c) for c in conc], "time": [float(t) for t in time]},
             "basal_max": basal_max}
    _save(project, state)
    return state


def ingest_plate(project, plate_csv):
    """Read a plate, subtract background, append one observation per (candidate, dose, time).

    A candidate on the plate but not in the project (e.g. WT) is skipped for training - WT is a
    reference, not a design point. -> summary dict."""
    from . import fluorescence as fl
    state = load(project)
    surfaces, background, wt_id = eio.read_plate(plate_csv)
    added, skipped = 0, []
    for cid, s in surfaces.items():
        cand = state["candidates"].get(cid)
        if cand is None:
            skipped.append(cid)
            continue
        # same cleaning path as the readouts: per-(c,t) empty-vector subtraction, sorted grid
        conc, time, F = fl.clean(s["conc"], s["time"], s["F"],
                                 background=eio.background_for(background, s))
        for i in range(len(conc)):
            for j in range(len(time)):
                state["observations"].append({
                    "cid": cid, "mech": cand["mech"], "mut": cand["mut"],
                    "scaffold": cand["scaffold"],
                    "conc": float(conc[i]), "time": float(time[j]),
                    "F": float(F[i, j])})
                added += 1
    _save(project, state)
    bg_scalar = background.get("scalar", 0.0) if isinstance(background, dict) else float(background)
    bg_mode = ("surface" if isinstance(background, dict) and background.get("surface") is not None
               else "scalar")
    return {"added": added, "background": bg_scalar, "background_mode": bg_mode, "wt_id": wt_id,
            "skipped": skipped, "total_observations": len(state["observations"])}


def select_next(project, n, n_samples=64, min_seq_dist=2, seed=0):
    """Refit the GP on all observations, Thompson-select n from the still-untested candidates.
    -> bandit.select result (selected ids, front probabilities, predicted phenotypes)."""
    state = load(project)
    if not state["observations"]:
        raise RuntimeError("no observations yet - run `feedback` with the first plate first")
    scaffolds = sorted({c["scaffold"] for c in state["candidates"].values()})
    gp = rgp.ResponseGP(scaffolds).fit(state["observations"])
    bandit = bd.ThompsonBandit(gp, state["grid"]["conc"], state["grid"]["time"],
                               n_samples=n_samples, seed=seed)

    tested = {o["cid"] for o in state["observations"]}
    pool = [dict(candidate_id=cid, **cand)
            for cid, cand in state["candidates"].items() if cid not in tested]
    if not pool:
        return {"selected": [], "note": "every candidate has already been tested",
                "prob": {}, "phenotypes": {}}
    res = bandit.select(pool, n, basal_max=state.get("basal_max"), min_seq_dist=min_seq_dist)
    res["n_untested"] = len(pool)
    return res
