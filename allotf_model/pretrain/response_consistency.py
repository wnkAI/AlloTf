"""Response-consistency QC - the gate that decides whether a (scaffold, effector_cluster) may become a
frozen native teacher. Fixes from the Codex review:

- distal region = the OPERATOR-derived DBD residues (the DNA-release readout), NOT a heuristic span.
  No DBD -> `no_teacher_missing_DBD` (fail closed); the gate is never run on a fabricated mask.
- MAGNITUDE null is the EXACT enumeration of block-preserving apo/holo relabellings (not repeated
  random permutations), so the effective sample size is honest: n_unique_null, exact p-value and the
  minimum attainable p are reported, and the verdict is `unresolved_low_power` when 5% significance is
  unreachable (e.g. 2 apo + 2 holo -> 6 partitions -> min p = 0.167).
- consistency is measured in the TEACHER's stored channel space (per-residue displacement MAGNITUDE
  pattern), matching what the frozen teacher keeps; the signed Cartesian-vector cosine is reported
  SEPARATELY as an auxiliary QC, not the gate.
- structures are parsed protomer-aware and averaged per PDB (via ensemble_alignment.file_observations).
"""
from itertools import combinations
from math import comb

import numpy as np

from .ensemble_alignment import file_observations

_MAX_ENUM = 50000


def _mag_field(ga, gb, distal):
    """Per-residue displacement MAGNITUDE (teacher channel-0 space). -> {ci: scalar}."""
    f = {}
    for ci in distal:
        a = np.array([g[ci] for g in ga if ci in g]); b = np.array([g[ci] for g in gb if ci in g])
        if len(a) and len(b):
            f[ci] = float(np.linalg.norm(b.mean(0) - a.mean(0)))
    return f


def _corr(fa, fb):
    shared = [ci for ci in fa if ci in fb]
    if len(shared) < 3:
        return 0.0
    u = np.array([fa[ci] for ci in shared]); v = np.array([fb[ci] for ci in shared])
    if u.std() == 0 or v.std() == 0:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1])


def consistency(apo_paths, holo_paths, reference_seq, dbd_idx, seed=0):
    distal = sorted(set(int(i) for i in dbd_idx))
    if not distal:
        return {"verdict": "no_teacher_missing_DBD",
                "reason": "no operator-derived DBD residues - cannot gate the DNA-release response"}
    ref, A = file_observations(apo_paths, reference_seq, distal, distal)
    _, H = file_observations(holo_paths, reference_seq, distal, distal)
    na, nh = len(A), len(H)
    if na < 2 or nh < 2:
        return {"verdict": "unresolved", "reason": "need >=2 apo and >=2 holo files", "n_apo": na, "n_holo": nh}
    # is the DBD even covered by these apo/holo structures?
    if sum(1 for ci in distal if any(ci in a for a in A) and any(ci in h for h in H)) < 3:
        return {"verdict": "unresolved", "reason": "DBD not covered by apo/holo (needs operator ensemble)",
                "n_apo": na, "n_holo": nh}

    files = A + H
    real_mag = sum(_mag_field(A, H, distal).values())
    n_unique = comb(na + nh, na)
    combos = list(combinations(range(na + nh), na))
    if len(combos) > _MAX_ENUM:                           # too many: sample, and say so
        rng = np.random.default_rng(seed)
        combos = [tuple(sorted(rng.choice(na + nh, na, replace=False))) for _ in range(_MAX_ENUM)]
        combos = list({c for c in combos})
        enumerated = False
    else:
        enumerated = True
    null_mags = []
    for combo in combos:
        cs = set(combo)
        ga = [files[i] for i in combo]; gb = [files[i] for i in range(na + nh) if i not in cs]
        null_mags.append(sum(_mag_field(ga, gb, distal).values()))
    exact_p = sum(m >= real_mag for m in null_mags) / len(null_mags)
    min_p = 1.0 / len(null_mags)
    if min_p > 0.05:
        return {"verdict": "unresolved_low_power", "n_apo": na, "n_holo": nh,
                "n_unique_null": len(null_mags), "min_attainable_p": round(min_p, 3),
                "reason": "too few structures to reach 5%% significance (%d partitions)" % len(null_mags)}
    significant = exact_p < 0.05

    # direction/pattern consistency in TEACHER channel space (per-residue magnitude pattern)
    holo_fields = [_mag_field(A, [h], distal) for h in H]
    pair = [_corr(holo_fields[i], holo_fields[j]) for i in range(nh) for j in range(i + 1, nh)]
    mean_pattern_corr = float(np.mean(pair)) if pair else 0.0
    null_corr = [_corr(_mag_field(A, [A[i]], distal), _mag_field(A, [A[j]], distal))
                 for i in range(na) for j in range(i + 1, na)]
    corr_thr = float(np.percentile(null_corr, 95)) if null_corr else 0.0
    consistent = mean_pattern_corr > corr_thr

    verdict = ("consistent_direction" if significant and consistent else
               "heterogeneous_direction" if significant else "unresolved")
    return {"verdict": verdict, "n_apo": na, "n_holo": nh, "distal_residues": len(distal),
            "n_unique_null": len(null_mags), "null_enumerated_exactly": enumerated,
            "real_magnitude": round(real_mag, 3), "exact_p": round(exact_p, 4),
            "min_attainable_p": round(min_p, 4), "magnitude_significant": significant,
            "mean_pattern_correlation": round(mean_pattern_corr, 3),
            "null_p95_correlation": round(corr_thr, 3), "pattern_consistent": consistent}
