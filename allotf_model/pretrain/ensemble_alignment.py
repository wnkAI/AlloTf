"""Canonical ensemble alignment for a scaffold's apo/holo structures. Residues are keyed by the
CANONICAL reference position via SEQUENCE ALIGNMENT (never PDB resseq - different entries renumber,
carry insertion codes, missing loops and construct mutations), so a residue means the same thing
across every PDB. Variant outliers (identity < MIN_IDENTITY to the reference) are dropped.

Every structure is superposed onto the reference by rigid-body fit on the STABLE ligand-binding CORE
(shared canonical residues outside the DBD and the distal response region) - never on the DBD, so real
domain motion is preserved. Then:
  - per-residue native response = holo-ensemble mean vs apo-ensemble mean (native_reference CHANNELS);
  - confidence = coverage (vs the aligned structures) x inter-conformer consistency;
  - dbd_rigid_motion = the apo->holo rotation + translation of the DBD as a whole, reported separately,
    NOT used in the fit.
"""
import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer

from .structure_qc import _align_identity, AA3TO1, MIN_IDENTITY, _MIN_CHAIN
from ..bridge.native_reference import CHANNELS, D_TEACHER

_CIF, _PDB = MMCIFParser(QUIET=True), PDBParser(QUIET=True)
_CONTACT = 8.0


def _canonical_ca(path, ref):
    """{canonical_index: CA coord} for the longest protein chain, aligned to the reference sequence.
    None if the chain is a variant outlier (identity < MIN_IDENTITY)."""
    parser = _CIF if path.lower().endswith(".cif") else _PDB
    model = next(iter(parser.get_structure("S", path)))
    chains = {}
    for ch in model:
        res = [(AA3TO1[r.get_resname()], r["CA"].coord.astype(float)) for r in ch
               if r.id[0] == " " and r.get_resname() in AA3TO1 and r.has_id("CA")]
        if len(res) >= _MIN_CHAIN:
            chains[ch.id] = res
    if not chains:
        return None
    main = max(chains.values(), key=len)
    ident, _, r2s = _align_identity("".join(a for a, _ in main), ref)
    if ident < MIN_IDENTITY:
        return None
    return {ci: main[pos][1] for ci, pos in r2s.items()}


def _fit(ref_pts, mob_pts):
    svd = SVDSuperimposer(); svd.set(ref_pts, mob_pts); svd.run()
    return svd.get_rotran()


def _contacts(coords):
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return (d < _CONTACT).sum(1) - 1


def _rigid_motion(apo_xyz, holo_xyz, idx):
    shared = [i for i in idx if i in apo_xyz and i in holo_xyz]
    if len(shared) < 3:
        return {"rotation_deg": float("nan"), "translation": float("nan"), "n_dbd": len(shared)}
    rot, tran = _fit(np.array([holo_xyz[i] for i in shared]), np.array([apo_xyz[i] for i in shared]))
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    return {"rotation_deg": float(angle), "translation": float(np.linalg.norm(tran)), "n_dbd": len(shared)}


def align_ensemble(apo_paths, holo_paths, reference_seq, dbd_idx, distal_idx):
    """-> dict(response [N,D], confidence [N], dbd_rigid_motion, n_apo, n_holo). N = len(reference_seq)."""
    n = len(reference_seq)
    dbd, distal = set(int(i) for i in dbd_idx), set(int(i) for i in distal_idx)
    apo = [c for c in (_canonical_ca(p, reference_seq) for p in apo_paths) if c]
    holo = [c for c in (_canonical_ca(p, reference_seq) for p in holo_paths) if c]
    if not apo or not holo:
        raise ValueError("ensemble alignment needs >=1 apo and >=1 holo after sequence alignment")
    ref = apo[0]

    def fit_to_ref(s):
        frame = [i for i in (set(ref) & set(s)) if i not in dbd and i not in distal]
        if len(frame) < 3:
            return None
        rot, tran = _fit(np.array([ref[i] for i in frame]), np.array([s[i] for i in frame]))
        return {i: s[i] @ rot + tran for i in s}

    aligned = {"apo": [a for a in (fit_to_ref(s) for s in apo) if a],
               "holo": [h for h in (fit_to_ref(s) for s in holo) if h]}
    resp, conf = np.zeros((n, D_TEACHER)), np.zeros(n)
    apo_mean, holo_mean = {}, {}
    for ci in range(n):
        ap = np.array([a[ci] for a in aligned["apo"] if ci in a])
        hp = np.array([h[ci] for h in aligned["holo"] if ci in h])
        if len(ap) == 0 or len(hp) == 0:
            continue
        apo_mean[ci], holo_mean[ci] = ap.mean(0), hp.mean(0)
        resp[ci, 0] = np.linalg.norm(holo_mean[ci] - apo_mean[ci])
        spread = float(ap.std(0).sum() + hp.std(0).sum())
        cover = min(len(ap), len(hp)) / max(len(aligned["apo"]), len(aligned["holo"]), 1)
        conf[ci] = cover / (1.0 + spread)
    shared = sorted(set(apo_mean) & set(holo_mean))
    if shared:
        nc_a = dict(zip(shared, _contacts(np.array([apo_mean[i] for i in shared]))))
        nc_h = dict(zip(shared, _contacts(np.array([holo_mean[i] for i in shared]))))
        for i in shared:
            resp[i, 1] = float(nc_h[i] - nc_a[i]); resp[i, 2] = float(nc_a[i]); resp[i, 3] = float(nc_h[i])
    return {"response": torch.tensor(resp, dtype=torch.float32),
            "confidence": torch.tensor(conf, dtype=torch.float32),
            "dbd_rigid_motion": _rigid_motion(apo_mean, holo_mean, sorted(dbd)),
            "n_apo": len(aligned["apo"]), "n_holo": len(aligned["holo"]), "channels": CHANNELS}
