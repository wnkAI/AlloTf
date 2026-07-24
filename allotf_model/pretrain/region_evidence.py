"""Evidence-driven CANDIDATE regions from the QC-passed structures - NOT final regions. Contacts are
computed PROTOMER-AWARE (see structure_parse): each protein chain is paired only with its own bound
effector and its own operator DNA, so a homodimer never mixes protomer A's chain with protomer B's
ligand. Pocket = residues within POCKET_CUTOFF of the protomer's effector across the holo ensemble,
scored by contact FREQUENCY over protomers. DBD = residues within cutoff of the protomer's operator
DNA. Hinge/transduction = motion + churn + pocket->DBD communicability bottleneck. Everything is keyed
by the canonical reference position. Regions are finalised only after human curation.
"""
import glob
import os

import numpy as np

from .structure_parse import parse_protomers

POCKET_CUTOFF = 5.0


def _contact_freq(scaffold_dir, sid, reference_seq, state, ligand_key, cutoff):
    """Fraction of protomers whose residue is within cutoff of the protomer's ligand/DNA."""
    counts, n_protomer, effectors = {}, 0, {}
    for p in sorted(glob.glob(os.path.join(scaffold_dir, state, "*.cif"))):
        for pr in parse_protomers(p, sid, reference_seq):
            target = pr["effector"][1] if ligand_key == "effector" and pr["effector"] else (
                pr["dna"] if ligand_key == "dna" else None)
            if target is None or len(target) == 0:
                continue
            n_protomer += 1
            if ligand_key == "effector":
                effectors[pr["effector"][0]] = effectors.get(pr["effector"][0], 0) + 1
            for ci, atoms in pr["heavy_by_ci"].items():
                if atoms.size and np.linalg.norm(atoms[:, None, :] - target[None, :, :], axis=2).min() <= cutoff:
                    counts[ci] = counts.get(ci, 0) + 1
    freq = {ci: round(c / n_protomer, 3) for ci, c in counts.items()} if n_protomer else {}
    return dict(sorted(freq.items(), key=lambda kv: -kv[1])), n_protomer, effectors


def pocket_evidence(scaffold_dir, sid, reference_seq, cutoff=POCKET_CUTOFF):
    freq, n, effectors = _contact_freq(scaffold_dir, sid, reference_seq, "holo", "effector", cutoff)
    dna = bool(glob.glob(os.path.join(scaffold_dir, "dna", "*.cif")) +
               glob.glob(os.path.join(scaffold_dir, "ternary", "*.cif")))
    return {"pocket_frequency": freq, "n_protomers": n, "effectors": effectors,
            "pocket_high_confidence": sorted(ci for ci, f in freq.items() if f >= 0.5),
            "needs": ([] if dna else ["operator_structure_for_DBD"])
                     + (["single_effector_confirmation"] if len(effectors) > 3 else [])}


def dbd_evidence(scaffold_dir, sid, reference_seq, cutoff=POCKET_CUTOFF):
    """DBD candidates = protomer residues within cutoff of the protomer's operator DNA. HIGH confidence
    (same-scaffold experimental). No operator structure -> empty + a flag so the DBD-contact loss is
    MASKED, never faked from a homology/predicted complex."""
    freq_d, n_d, _ = _contact_freq(scaffold_dir, sid, reference_seq, "dna", "dna", cutoff)
    freq_t, n_t, _ = _contact_freq(scaffold_dir, sid, reference_seq, "ternary", "dna", cutoff)
    counts = {}
    for f, n in ((freq_d, n_d), (freq_t, n_t)):
        for ci, fr in f.items():
            counts[ci] = counts.get(ci, 0) + fr * n
    used = n_d + n_t
    if used == 0:
        return {"dbd_frequency": {}, "n_operator_protomers": 0, "confidence": "none",
                "note": "no_operator_structure - DBD-contact loss must be masked (family template only)"}
    freq = {ci: round(c / used, 3) for ci, c in counts.items()}
    return {"dbd_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])), "n_operator_protomers": used,
            "confidence": "high_same_scaffold_experimental",
            "dbd_contact_residues": sorted(ci for ci, f in freq.items() if f >= 0.5)}


def hinge_evidence(ensemble, pocket_idx, dbd_idx, n_res, contact_cutoff=10.0):
    """Transduction/hinge CANDIDATES between the pocket and the DBD: apo->holo motion + contact churn +
    pocket->DBD communicability bottleneck on the CA contact graph. Gated to residues ON the pocket->DBD
    path (bottleneck>0) when the DBD is in the ensemble; falls back to motion+churn with a flag when the
    DBD is absent. Outputs an evidence SCORE + components - NOT a final region."""
    from .bottleneck_targets import communicability_bottleneck
    import torch
    resp, ca = ensemble["response"], ensemble["ca_apo"]
    covered = sorted(ca)
    motion = resp[:, 0].numpy()
    churn = np.abs(resp[:, 1].numpy())
    coords = np.array([ca[i] for i in covered])
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    src, dst = [], []
    for a in range(len(covered)):
        for b in range(len(covered)):
            if a != b and d[a, b] < contact_cutoff:
                src.append(covered[a]); dst.append(covered[b])
    bott = communicability_bottleneck(n_res, torch.tensor([src, dst], dtype=torch.long),
                                      list(pocket_idx), list(dbd_idx)).numpy()
    pocket, dbd = set(int(i) for i in pocket_idx), set(int(i) for i in dbd_idx)
    dbd_covered = any(i in ca for i in dbd)
    note = None
    if dbd_covered:
        cand = [i for i in covered if i not in pocket and i not in dbd and bott[i] > 0]
    else:
        cand = [i for i in covered if i not in pocket and i not in dbd]
        note = "DBD absent from apo/holo ensemble - pocket->DBD path unavailable, needs operator structures"
    if not cand:
        return {"hinge_score": {}, "hinge_candidates": [], "note": note or "no candidate residues",
                "dbd_in_ensemble": dbd_covered}

    def z(vec):
        v = np.array([vec[i] for i in cand]); s = v.std()
        return (v - v.mean()) / s if s > 0 else v * 0.0
    zm, zc, zb = z(motion), z(churn), z(bott)
    w = (1.0, 1.0, 1.0) if dbd_covered else (1.0, 1.0, 0.0)
    score = {cand[k]: round(float(w[0] * zm[k] + w[1] * zc[k] + w[2] * zb[k]), 3) for k in range(len(cand))}
    ranked = sorted(score, key=lambda i: -score[i])
    comp = {cand[k]: {"motion": round(float(zm[k]), 2), "churn": round(float(zc[k]), 2),
                      "bottleneck": round(float(zb[k]), 2)} for k in range(len(cand))}
    return {"hinge_score": {i: score[i] for i in ranked}, "hinge_candidates": ranked[:20],
            "components": {i: comp[i] for i in ranked[:20]}, "dbd_in_ensemble": dbd_covered, "note": note}
