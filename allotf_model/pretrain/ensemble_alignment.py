"""Canonical ensemble alignment. Structures are parsed PROTOMER-AWARE (structure_parse): each protein
chain is aligned to the reference by SEQUENCE (never resseq). Every protomer is rigid-body fit onto the
reference on the STABLE ligand-binding CORE (shared canonical residues outside the DBD and the distal
region) - never the DBD, so domain motion is preserved - and the protomers of one PDB are AVERAGED into
a single observation (a homodimer contributes one observation, not two correlated ones).

Then per residue: native response = holo-ensemble mean vs apo-ensemble mean (native_reference CHANNELS);
confidence = coverage x inter-conformer consistency; dbd_rigid_motion = the apo->holo rotation +
translation of the DBD as a whole (reported, not used in the fit).
"""
from collections import defaultdict

import numpy as np
import torch
from Bio.SVDSuperimposer import SVDSuperimposer

from .structure_parse import parse_protomers
from ..bridge.native_reference import CHANNELS, D_TEACHER

_CONTACT = 8.0


def _fit(ref_pts, mob_pts):
    svd = SVDSuperimposer(); svd.set(ref_pts, mob_pts); svd.run()
    return svd.get_rotran()


def _file_obs(proto_list, ref, dbd, distal):
    """Fit each protomer onto ref (core frame) and AVERAGE them -> one {ci: CA} observation per file."""
    fitted = []
    for pr in proto_list:
        s = pr["canonical_ca"]
        frame = [i for i in (set(ref) & set(s)) if i not in dbd and i not in distal]
        if len(frame) < 3:
            continue
        rot, tran = _fit(np.array([ref[i] for i in frame]), np.array([s[i] for i in frame]))
        fitted.append({i: s[i] @ rot + tran for i in s})
    if not fitted:
        return None
    acc = defaultdict(list)
    for f in fitted:
        for ci, c in f.items():
            acc[ci].append(c)
    return {ci: np.mean(v, 0) for ci, v in acc.items()}


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


def file_observations(paths, reference_seq, dbd_idx, distal_idx):
    """-> (ref CA dict, [one averaged observation per file that aligns])."""
    protos = [pl for pl in (parse_protomers(p, "S", reference_seq) for p in paths) if pl]
    if not protos:
        return None, []
    ref = protos[0][0]["canonical_ca"]
    dbd, distal = set(int(i) for i in dbd_idx), set(int(i) for i in distal_idx)
    obs = [o for o in (_file_obs(pl, ref, dbd, distal) for pl in protos) if o]
    return ref, obs


def align_ensemble(apo_paths, holo_paths, reference_seq, dbd_idx, distal_idx):
    """-> dict(response [N,D], confidence [N], dbd_rigid_motion, ca_apo, n_apo, n_holo)."""
    n = len(reference_seq)
    dbd, distal = set(int(i) for i in dbd_idx), sorted(set(int(i) for i in distal_idx))
    ref, apo = file_observations(apo_paths, reference_seq, dbd, distal)
    _, holo = file_observations(holo_paths, reference_seq, dbd, distal)
    if not apo or not holo:
        raise ValueError("ensemble alignment needs >=1 apo and >=1 holo file after alignment")

    resp, conf = np.zeros((n, D_TEACHER)), np.zeros(n)
    apo_mean, holo_mean = {}, {}
    for ci in range(n):
        ap = np.array([a[ci] for a in apo if ci in a])
        hp = np.array([h[ci] for h in holo if ci in h])
        if len(ap) == 0 or len(hp) == 0:
            continue
        apo_mean[ci], holo_mean[ci] = ap.mean(0), hp.mean(0)
        resp[ci, 0] = np.linalg.norm(holo_mean[ci] - apo_mean[ci])
        spread = float(ap.std(0).sum() + hp.std(0).sum())
        conf[ci] = min(len(ap), len(hp)) / max(len(apo), len(holo), 1) / (1.0 + spread)
    shared = sorted(set(apo_mean) & set(holo_mean))
    if shared:
        nc_a = dict(zip(shared, _contacts(np.array([apo_mean[i] for i in shared]))))
        nc_h = dict(zip(shared, _contacts(np.array([holo_mean[i] for i in shared]))))
        for i in shared:
            resp[i, 1] = float(nc_h[i] - nc_a[i]); resp[i, 2] = float(nc_a[i]); resp[i, 3] = float(nc_h[i])
    return {"response": torch.tensor(resp, dtype=torch.float32),
            "confidence": torch.tensor(conf, dtype=torch.float32),
            "dbd_rigid_motion": _rigid_motion(apo_mean, holo_mean, sorted(dbd)),
            "ca_apo": apo_mean, "n_apo": len(apo), "n_holo": len(holo), "channels": CHANNELS}
