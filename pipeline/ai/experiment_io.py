"""Read a plate CSV into per-candidate F(c, t) surfaces.

Required columns: candidate_id, concentration, time_h, fluorescence, replicate.

The empty-vector wells set the background (autofluorescence + plate offset) that fluorescence.clean
subtracts; the WT wells are kept as the reference every candidate is normalised against downstream.
Replicates of the same (candidate, concentration, time) are averaged and their scatter recorded —
repeated reads of one well are replicates in time, not extra plasmids.
"""
import csv
from collections import defaultdict

import numpy as np

REQUIRED_COLUMNS = ["candidate_id", "concentration", "time_h", "fluorescence", "replicate"]


def _rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError("plate CSV missing columns: %s (has %s)"
                             % (missing, r.fieldnames))
        for row in r:
            yield row


def _surface(entries, cid):
    """entries: {(conc, time): [values]} -> (conc_grid, time_grid, F_mean, F_sd, n_rep).

    A missing (conc, time) cell is a hard error, not a silent NaN: a hole in the grid means the
    plate was read wrong, and averaging around it would fabricate a response.
    """
    concs = sorted({c for c, _ in entries})
    times = sorted({t for _, t in entries})
    F = np.full((len(concs), len(times)), np.nan)
    SD = np.zeros_like(F)
    N = np.zeros_like(F, dtype=int)
    ci = {c: i for i, c in enumerate(concs)}
    ti = {t: j for j, t in enumerate(times)}
    for (c, t), vals in entries.items():
        i, j = ci[c], ti[t]
        F[i, j] = float(np.mean(vals))
        SD[i, j] = float(np.std(vals))
        N[i, j] = len(vals)
    holes = [(concs[i], times[j]) for i in range(len(concs)) for j in range(len(times))
             if np.isnan(F[i, j])]
    if holes:
        raise ValueError("candidate %s has empty (conc,time) cells %s: incomplete plate"
                         % (cid, holes[:6]))
    return np.array(concs), np.array(times), F, SD, N


def read_plate(path, empty_vector_ids=("empty", "empty_vector", "EV"),
               wt_ids=("WT", "wt", "wild_type")):
    """-> (surfaces, background, wt_id)

    surfaces: {candidate_id: {'conc','time','F','F_sd','n_rep'}} for real candidates
    background: scalar autofluorescence from the empty-vector wells (0.0 if none present)
    wt_id: the WT candidate_id if one was on the plate, else None
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in _rows(path):
        cid = row["candidate_id"].strip()
        try:
            c = float(row["concentration"])
            t = float(row["time_h"])
            fl = float(row["fluorescence"])
        except (ValueError, KeyError) as exc:
            raise ValueError("bad numeric field in row %s: %s" % (row, exc))
        grouped[cid][(c, t)].append(fl)

    # empty vector -> a per-(c,t) background SURFACE when its grid is complete. Inducer
    # autofluorescence depends on dose and the plate drifts with time; one averaged scalar removes
    # neither. The scalar mean is kept only as a fallback for an incomplete control grid.
    ev_ids = [cid for cid in grouped if cid in empty_vector_ids]
    background = {"surface": None, "scalar": 0.0}
    if ev_ids:
        merged = defaultdict(list)
        for cid in ev_ids:
            for key, vals in grouped[cid].items():
                merged[key].extend(vals)
        flat = [v for vals in merged.values() for v in vals]
        background["scalar"] = float(np.mean(flat)) if flat else 0.0
        try:
            bc, bt, BF, _, _ = _surface(merged, "empty-vector")
            background["surface"] = {"conc": bc, "time": bt, "F": BF}
        except ValueError:
            pass          # incomplete control grid: fall back to the scalar

    wt_id = next((cid for cid in grouped if cid in wt_ids), None)

    surfaces = {}
    for cid, entries in grouped.items():
        if cid in empty_vector_ids:
            continue
        conc, time, F, SD, N = _surface(entries, cid)
        surfaces[cid] = {"conc": conc, "time": time, "F": F, "F_sd": SD, "n_rep": N}
    if not surfaces:
        raise ValueError("no candidate surfaces in %s (only empty-vector rows?)" % path)
    return surfaces, background, wt_id


def background_for(background, surface):
    """The empty-vector background to subtract from ONE candidate surface: the per-(c,t) control
    matrix when its grid matches, otherwise the scalar mean."""
    if not isinstance(background, dict):
        return background
    bg = background.get("surface")
    if (bg is not None
            and np.array_equal(np.asarray(bg["conc"]), np.asarray(surface["conc"]))
            and np.array_equal(np.asarray(bg["time"]), np.asarray(surface["time"]))):
        return np.asarray(bg["F"])
    return background.get("scalar", 0.0)


def readouts(path, **kw):
    """Convenience: plate CSV -> {candidate_id: phenotypes dict} with background subtracted."""
    from . import fluorescence as fl
    surfaces, background, wt_id = read_plate(path, **kw)
    out = {}
    for cid, s in surfaces.items():
        conc, time, F = fl.clean(s["conc"], s["time"], s["F"],
                                 background=background_for(background, s))
        out[cid] = fl.phenotypes(conc, time, F)
    return {"phenotypes": out, "background": background, "wt_id": wt_id}
