"""Canonical ensemble alignment for a scaffold's apo/holo structures. Every structure is superposed
onto one reference by rigid-body fit on the STABLE ligand-binding CORE (shared residues outside the
DBD and outside the distal response region) - never on the DBD, so real domain motion is preserved
(same principle as the pipeline's build_induced_on_dna). Then:

  - per-residue native response = holo-ensemble mean vs apo-ensemble mean (the CHANNELS of
    native_reference: ca_displacement, contact_count_change, n_neighbours_apo, n_neighbours_holo);
  - confidence = coverage x inter-conformer consistency (residues that move consistently across the
    ensemble are trusted; scattered ones are down-weighted, never fabricated);
  - dbd_rigid_motion = the apo->holo rotation angle + translation of the DBD as a whole (the readout the
    switch must reproduce), reported separately, NOT used in the fit.

Works on biological-assembly mmCIF (or PDB); residues are keyed by the frozen canonical mapping.
"""
import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser, Superimposer

from ..bridge.native_reference import CHANNELS, D_TEACHER, _contacts

_CIF, _PDB = MMCIFParser(QUIET=True), PDBParser(QUIET=True)


def _ca(path, mapping):
    parser = _CIF if path.lower().endswith(".cif") else _PDB
    model = next(iter(parser.get_structure(mapping.scaffold_id, path)))
    out = {}
    for ch in model:
        for r in ch:
            if r.id[0] == " " and r.has_id("CA"):
                ci = mapping.canonical(ch.id, r.id[1], r.id[2] or " ")
                if ci is not None:
                    out[ci] = r["CA"]
    return out


def _fit(ref, mob, frame):
    sup = Superimposer()
    sup.set_atoms([ref[i] for i in frame], [mob[i] for i in frame])
    return sup.rotran


def _rigid_motion(apo_xyz, holo_xyz, idx):
    """Rotation angle (deg) + translation (A) taking the apo DBD onto the holo DBD."""
    shared = [i for i in idx if i in apo_xyz and i in holo_xyz]
    if len(shared) < 3:
        return {"rotation_deg": float("nan"), "translation": float("nan")}
    from Bio.SVDSuperimposer import SVDSuperimposer
    svd = SVDSuperimposer()
    svd.set(np.array([holo_xyz[i] for i in shared]), np.array([apo_xyz[i] for i in shared]))
    svd.run()
    rot, tran = svd.get_rotran()
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    return {"rotation_deg": float(angle), "translation": float(np.linalg.norm(tran))}


def align_ensemble(apo_paths, holo_paths, mapping, dbd_idx, distal_idx):
    """-> dict(response [N,D_TEACHER], confidence [N], dbd_rigid_motion, n_apo, n_holo)."""
    n = mapping.n_res
    dbd, distal = set(int(i) for i in dbd_idx), set(int(i) for i in distal_idx)
    apo = [_ca(p, mapping) for p in apo_paths]
    holo = [_ca(p, mapping) for p in holo_paths]
    if not apo or not holo:
        raise ValueError("ensemble alignment needs >=1 apo and >=1 holo structure")
    ref = apo[0]

    def frame_for(struct):                                # stable core: shared, not DBD, not distal
        return [i for i in (set(ref) & set(struct)) if i not in dbd and i not in distal]

    aligned = {"apo": [], "holo": []}
    for group, structs in (("apo", apo), ("holo", holo)):
        for s in structs:
            fr = frame_for(s)
            if len(fr) < 3:
                continue
            rot, tran = (np.eye(3), np.zeros(3)) if s is ref else _fit(ref, s, fr)
            aligned[group].append({i: s[i].coord @ rot + tran for i in s})

    resp = torch.zeros(n, D_TEACHER)
    conf = torch.zeros(n)
    for ci in range(n):
        apo_pts = np.array([a[ci] for a in aligned["apo"] if ci in a])
        holo_pts = np.array([h[ci] for h in aligned["holo"] if ci in h])
        if len(apo_pts) == 0 or len(holo_pts) == 0:
            continue
        apo_mean, holo_mean = apo_pts.mean(0), holo_pts.mean(0)
        disp = float(np.linalg.norm(holo_mean - apo_mean))
        # inter-conformer spread: consistent residues (small spread relative to displacement) are trusted
        spread = float(apo_pts.std(0).sum() + holo_pts.std(0).sum())
        # coverage against the ALIGNED structures (some inputs may have been skipped on a tiny frame)
        cover = min(len(apo_pts), len(holo_pts)) / max(len(aligned["apo"]), len(aligned["holo"]), 1)
        conf[ci] = cover / (1.0 + spread)
        resp[ci, 0] = disp                                # ca_displacement (contacts filled below)

    # contact-based channels from the ensemble-mean coordinates
    apo_mean_xyz = {ci: np.mean([a[ci] for a in aligned["apo"] if ci in a], 0)
                    for ci in range(n) if any(ci in a for a in aligned["apo"])}
    holo_mean_xyz = {ci: np.mean([h[ci] for h in aligned["holo"] if ci in h], 0)
                     for ci in range(n) if any(ci in h for h in aligned["holo"])}
    shared = sorted(set(apo_mean_xyz) & set(holo_mean_xyz))
    if shared:
        nc_apo = dict(zip(shared, _contacts(np.array([apo_mean_xyz[i] for i in shared]))))
        nc_holo = dict(zip(shared, _contacts(np.array([holo_mean_xyz[i] for i in shared]))))
        for i in shared:
            resp[i, 1] = float(nc_holo[i] - nc_apo[i])
            resp[i, 2] = float(nc_apo[i])
            resp[i, 3] = float(nc_holo[i])

    motion = _rigid_motion(apo_mean_xyz, holo_mean_xyz, sorted(dbd))
    return {"response": resp, "confidence": conf, "dbd_rigid_motion": motion,
            "n_apo": len(aligned["apo"]), "n_holo": len(aligned["holo"]), "channels": CHANNELS}
