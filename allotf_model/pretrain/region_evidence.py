"""Evidence-driven CANDIDATE regions from the QC-passed structures - NOT final regions. The pocket is
the objective anchor: residues within POCKET_CUTOFF of the bound effector across the holo ENSEMBLE,
scored by contact FREQUENCY (so the pocket is not defined by a single crystal form). Everything is
indexed by the canonical reference position (alignment, not resseq), so it is comparable across PDBs.

Output per scaffold is a table {canonical_index: {frequency, n_holo, evidence}}; the pocket, hinge and
DBD are finalised only after human curation against the original papers / SI, and DBD needs operator
structures (dna/ternary states), which are flagged when absent. Nothing here is written as a final
region.
"""
import glob
import os

import numpy as np
from Bio.PDB import MMCIFParser

from .structure_qc import _align_identity, AA3TO1, MIN_IDENTITY, _MIN_CHAIN
from .build_manifest import IONS, ADDITIVES

_PARSER = MMCIFParser(QUIET=True)
POCKET_CUTOFF = 5.0
_NON_EFFECTOR = IONS | ADDITIVES          # comprehensive: a large PEG must not be picked as the effector


def _protein_and_effector(cif, sid):
    """-> (list of (aa1, [heavy-atom coords]) for the longest protein chain, effector-ligand heavy
    coords [K,3] or None, effector comp id)."""
    model = next(iter(_PARSER.get_structure(sid, cif)))
    chains, hets = {}, []
    for ch in model:
        res = []
        for r in ch:
            if r.id[0] == " " and r.get_resname() in AA3TO1 and r.has_id("CA"):
                coords = [a.coord for a in r if a.element != "H"]
                res.append((AA3TO1[r.get_resname()], np.array(coords)))
            elif r.id[0].startswith("H_"):
                nm = r.get_resname().strip().upper()
                if nm not in _NON_EFFECTOR:
                    hets.append((nm, np.array([a.coord for a in r if a.element != "H"])))
        if len(res) >= _MIN_CHAIN:
            chains[ch.id] = res
    if not chains:
        return None, None, None
    main = max(chains.values(), key=len)
    eff = max(hets, key=lambda h: len(h[1])) if hets else (None, None)
    return main, eff[1], eff[0]


_DNA_RES = {"DA", "DC", "DG", "DT", "DU"}


def _protein_and_dna(cif, sid):
    """-> (longest protein chain residues [(aa1, heavy coords)], DNA heavy-atom coords [K,3] or None)."""
    model = next(iter(_PARSER.get_structure(sid, cif)))
    chains, dna = {}, []
    for ch in model:
        res = []
        for r in ch:
            nm = r.get_resname().strip().upper()
            if r.id[0] == " " and nm in AA3TO1 and r.has_id("CA"):
                res.append((AA3TO1[nm], np.array([a.coord for a in r if a.element != "H"])))
            elif nm in _DNA_RES:
                dna.extend(a.coord for a in r if a.element != "H")
        if len(res) >= _MIN_CHAIN:
            chains[ch.id] = res
    if not chains:
        return None, None
    return max(chains.values(), key=len), (np.array(dna) if dna else None)


def dbd_evidence(scaffold_dir, sid, reference_seq, cutoff=POCKET_CUTOFF):
    """DBD candidates = protein residues within cutoff of operator DNA, across operator complexes.
    HIGH confidence (same-scaffold experimental operator). No operator structure -> empty + a flag so
    the DBD-contact loss is MASKED, never faked from a homology/predicted complex."""
    ops = sorted(glob.glob(os.path.join(scaffold_dir, "dna", "*.cif")) +
                 glob.glob(os.path.join(scaffold_dir, "ternary", "*.cif")))
    counts, used = {}, 0
    for p in ops:
        main, dna = _protein_and_dna(p, sid)
        if main is None or dna is None or len(dna) == 0:
            continue
        seq = "".join(a for a, _ in main)
        ident, _, ref_to_seq = _align_identity(seq, reference_seq)
        if ident < MIN_IDENTITY:
            continue
        used += 1
        for ci, pos in ref_to_seq.items():
            atoms = main[pos][1]
            if atoms.size and np.linalg.norm(atoms[:, None, :] - dna[None, :, :], axis=2).min() <= cutoff:
                counts[ci] = counts.get(ci, 0) + 1
    if used == 0:
        return {"dbd_frequency": {}, "n_operator": 0, "confidence": "none",
                "note": "no_operator_structure - DBD-contact loss must be masked (family template only)"}
    freq = {ci: round(c / used, 3) for ci, c in counts.items()}
    return {"dbd_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])), "n_operator": used,
            "confidence": "high_same_scaffold_experimental",
            "dbd_contact_residues": sorted(ci for ci, f in freq.items() if f >= 0.5)}


def hinge_evidence(ensemble, pocket_idx, dbd_idx, n_res, contact_cutoff=10.0):
    """Transduction/hinge CANDIDATES between the pocket and the DBD. Combines apo->holo motion, contact
    churn, and pocket->DBD communicability bottleneck on the CA contact graph. Pocket, DNA-contact
    (DBD) and uncovered residues are excluded. Outputs an evidence SCORE + component scores - NOT a
    final region (curation decides)."""
    from .bottleneck_targets import communicability_bottleneck
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
    import torch
    bott = communicability_bottleneck(n_res, torch.tensor([src, dst], dtype=torch.long),
                                      list(pocket_idx), list(dbd_idx)).numpy()
    pocket, dbd = set(int(i) for i in pocket_idx), set(int(i) for i in dbd_idx)
    dbd_covered = any(i in ca for i in dbd)
    note = None
    # a transduction residue must lie ON the pocket->DBD communication path (bottleneck > 0); this
    # excludes floppy termini that move a lot but carry no signal. If the DBD is absent from the
    # apo/holo ensemble (e.g. core-only LacI structures) the path cannot be computed - flag it and fall
    # back to motion+churn, never pretend a transduction path exists.
    if dbd_covered:
        cand = [i for i in covered if i not in pocket and i not in dbd and bott[i] > 0]
    else:
        cand = [i for i in covered if i not in pocket and i not in dbd]
        note = "DBD absent from apo/holo ensemble - pocket->DBD path unavailable, needs operator structures"
    if not cand:
        return {"hinge_score": {}, "hinge_candidates": [], "note": note or "no candidate residues"}

    def z(vec):
        v = np.array([vec[i] for i in cand]); s = v.std()
        return (v - v.mean()) / s if s > 0 else v * 0.0
    zm, zc, zb = z(motion), z(churn), z(bott)
    weights = (1.0, 1.0, 1.0) if dbd_covered else (1.0, 1.0, 0.0)
    score = {cand[k]: round(float(weights[0] * zm[k] + weights[1] * zc[k] + weights[2] * zb[k]), 3)
             for k in range(len(cand))}
    ranked = sorted(score, key=lambda i: -score[i])
    comp = {cand[k]: {"motion": round(float(zm[k]), 2), "churn": round(float(zc[k]), 2),
                      "bottleneck": round(float(zb[k]), 2)} for k in range(len(cand))}
    return {"hinge_score": {i: score[i] for i in ranked}, "hinge_candidates": ranked[:20],
            "components": {i: comp[i] for i in ranked[:20]},
            "dbd_in_ensemble": dbd_covered, "note": note}


def pocket_evidence(scaffold_dir, sid, reference_seq, cutoff=POCKET_CUTOFF):
    """-> dict(pocket_frequency {canonical_index: freq}, n_holo, effectors, needs)."""
    holo = sorted(glob.glob(os.path.join(scaffold_dir, "holo", "*.cif")))
    counts = {}
    effectors, used = {}, 0
    for p in holo:
        main, eff_coords, eff_name = _protein_and_effector(p, sid)
        if main is None or eff_coords is None or len(eff_coords) == 0:
            continue
        seq = "".join(a for a, _ in main)
        ident, _, ref_to_seq = _align_identity(seq, reference_seq)
        if ident < MIN_IDENTITY:
            continue                                     # variant outlier - not this scaffold's cluster
        used += 1
        effectors[eff_name] = effectors.get(eff_name, 0) + 1
        for ci, pos in ref_to_seq.items():
            res_atoms = main[pos][1]
            if res_atoms.size == 0:
                continue
            dmin = np.linalg.norm(res_atoms[:, None, :] - eff_coords[None, :, :], axis=2).min()
            if dmin <= cutoff:
                counts[ci] = counts.get(ci, 0) + 1
    freq = {ci: round(c / used, 3) for ci, c in counts.items()} if used else {}
    dna = bool(glob.glob(os.path.join(scaffold_dir, "dna", "*.cif")) +
               glob.glob(os.path.join(scaffold_dir, "ternary", "*.cif")))
    return {"pocket_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
            "n_holo_used": used, "effectors": effectors,
            "pocket_high_confidence": sorted(ci for ci, f in freq.items() if f >= 0.5),
            "needs": ([] if dna else ["operator_structure_for_DBD"])
                     + (["single_effector_confirmation"] if len(effectors) > 3 else [])}
