"""Response-consistency QC - the gate that decides whether a (scaffold, effector_cluster) may become a
frozen native teacher. It asks two questions with data-driven null distributions, NOT an arbitrary
cosine threshold:

  1. MAGNITUDE: is the real apo->holo distal response larger than crystal-form / label-shuffle noise?
     Nulls: apo-apo split, holo-holo split, apo/holo mismatch. Significant if the real distal
     displacement exceeds the 95th percentile of the pooled null.
  2. DIRECTION: do different holo crystal forms move the DISTAL region the SAME way? Compare the mean
     pairwise cosine of per-holo response fields against the apo-apo null cosine.

Verdict:
  consistent_direction    - significant magnitude AND direction agrees  -> may build a frozen teacher
  heterogeneous_direction - significant magnitude but direction disagrees -> split by chemotype
  unresolved              - not distinguishable from noise               -> auxiliary only, no teacher

All structures are parsed once into canonical CA (sequence alignment), fitted on the stable core
(excluding DBD and distal); nulls are computed in-memory.
"""
import numpy as np

from .ensemble_alignment import _canonical_ca, _fit


def _fit_group(structs, ref, frame):
    out = []
    for s in structs:
        fr = [i for i in frame if i in s]
        if len(fr) < 3:
            continue
        rot, tran = _fit(np.array([ref[i] for i in fr]), np.array([s[i] for i in fr]))
        out.append({i: s[i] @ rot + tran for i in s})
    return out


def _field(group_a, group_b, distal):
    """Per-distal-residue mean displacement group_b - group_a. -> ({canonical_index: vec}, magnitude).
    Keyed by residue so two fields covering different residue subsets can still be compared on their
    shared residues."""
    field = {}
    for ci in distal:
        a = np.array([g[ci] for g in group_a if ci in g])
        b = np.array([g[ci] for g in group_b if ci in g])
        if len(a) and len(b):
            field[ci] = b.mean(0) - a.mean(0)
    return field, float(sum(np.linalg.norm(v) for v in field.values()))


def consistency(apo_paths, holo_paths, reference_seq, dbd_idx, distal_idx, n_null=200, seed=0):
    rng = np.random.default_rng(seed)
    apo = [c for c in (_canonical_ca(p, reference_seq) for p in apo_paths) if c]
    holo = [c for c in (_canonical_ca(p, reference_seq) for p in holo_paths) if c]
    if len(apo) < 2 or len(holo) < 2:
        return {"verdict": "unresolved", "reason": "need >=2 apo and >=2 holo after alignment",
                "n_apo": len(apo), "n_holo": len(holo)}
    dbd, distal = set(int(i) for i in dbd_idx), sorted(set(int(i) for i in distal_idx))
    ref = apo[0]
    frame = [i for i in ref if i not in dbd and i not in distal]
    A, H = _fit_group(apo, ref, frame), _fit_group(holo, ref, frame)

    _, real_mag = _field(A, H, distal)

    def split_mag(pool):
        idx = rng.permutation(len(pool)); h = len(pool) // 2
        return _field([pool[i] for i in idx[:h]], [pool[i] for i in idx[h:]], distal)[1]
    nulls = ([split_mag(A) for _ in range(n_null)] + [split_mag(H) for _ in range(n_null)])
    pool = A + H
    for _ in range(n_null):
        idx = rng.permutation(len(pool))
        nulls.append(_field([pool[i] for i in idx[:len(A)]], [pool[i] for i in idx[len(A):]], distal)[1])
    thr = float(np.percentile(nulls, 95))
    significant = real_mag > thr

    # direction: mean pairwise cosine of each holo's own apo->holo field vs apo-apo null cosine
    def cos(fa, fb):
        shared = [ci for ci in fa if ci in fb]
        if len(shared) < 3:
            return 0.0
        u = np.concatenate([fa[ci] for ci in shared]); v = np.concatenate([fb[ci] for ci in shared])
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    holo_fields = [_field(A, [h], distal)[0] for h in H]
    pair_cos = [cos(holo_fields[i], holo_fields[j]) for i in range(len(H)) for j in range(i + 1, len(H))]
    mean_cos = float(np.mean(pair_cos)) if pair_cos else 0.0
    null_cos = []
    for _ in range(n_null):
        i, j = rng.integers(0, len(A)), rng.integers(0, len(A))
        if i != j:
            null_cos.append(cos(_field(A, [A[i]], distal)[0], _field(A, [A[j]], distal)[0]))
    cos_thr = float(np.percentile(null_cos, 95)) if null_cos else 0.0
    consistent_dir = mean_cos > cos_thr

    verdict = ("consistent_direction" if significant and consistent_dir else
               "heterogeneous_direction" if significant else "unresolved")
    return {"verdict": verdict, "n_apo": len(A), "n_holo": len(H),
            "real_magnitude": round(real_mag, 3), "null_p95_magnitude": round(thr, 3),
            "magnitude_significant": significant,
            "mean_holo_direction_cosine": round(mean_cos, 3), "null_p95_cosine": round(cos_thr, 3),
            "direction_consistent": consistent_dir}
