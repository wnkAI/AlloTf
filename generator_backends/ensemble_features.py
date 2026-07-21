"""Turn a conformer ensemble into the structural summary features the bridge packs as confidence-
masked aux - NOT into a functional score. These describe how consistent / uncertain the generated
structures are, so AlloTransfer can down-weight shaky candidates; they never claim DNA release.

Features (all from the ensemble, after aligning every conformer onto the first on CA):
    ligand_pose_spread        - mean ligand-atom RMSD across conformers (low = consistent pose)
    pocket_contact_frequency  - how often pocket residues contact the ligand across conformers
    backbone_rmsf             - mean per-residue CA fluctuation across conformers
    inter_conformer_variance  - overall CA positional variance
    mean_gen_confidence       - mean structural confidence reported by the generator
    n_conformers              - ensemble size
"""
import numpy as np

NAMES = ("ligand_pose_spread", "pocket_contact_frequency", "backbone_rmsf",
         "inter_conformer_variance", "mean_gen_confidence", "n_conformers")
_LIG_CONTACT = 4.5


def _kabsch(mobile, ref):
    """Rotation+translation aligning mobile onto ref (both [N,3]); returns the moved mobile."""
    mc, rc = mobile.mean(0), ref.mean(0)
    H = (mobile - mc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (mobile - mc) @ R.T + rc


def ensemble_features(conformers):
    """conformers: list[Conformer]. -> {names, values, confidence}. confidence reflects ensemble
    size and reported generator confidence, so a 1-conformer ensemble is flagged as uncertain."""
    if not conformers:
        raise ValueError("empty conformer ensemble")
    ref = conformers[0].ca_coords
    ca = np.stack([_kabsch(c.ca_coords, ref) if i else c.ca_coords
                   for i, c in enumerate(conformers)])            # [K, N, 3]
    lig = [c.ligand_coords for c in conformers]
    K = len(conformers)

    inter_var = float(ca.var(0).sum(-1).mean())                  # mean over residues of CA variance
    rmsf = float(np.sqrt(ca.var(0).sum(-1)).mean())
    # ligand pose spread: align ligands by the SAME protein transform is implicit (ligand is in the
    # protein frame per conformer); report mean atom-wise std if atom counts match, else nan
    if len({l.shape for l in lig}) == 1:
        lig_arr = np.stack(lig)
        lig_spread = float(np.sqrt(lig_arr.var(0).sum(-1)).mean())
    else:
        lig_spread = float("nan")
    # pocket contact frequency across conformers
    freqs = []
    for c in conformers:
        pk = c.ca_coords[c.pocket_idx]
        d = np.linalg.norm(pk[:, None, :] - c.ligand_coords[None, :, :], axis=2)
        freqs.append((d.min(1) < _LIG_CONTACT).mean())
    contact_freq = float(np.mean(freqs))
    mean_conf = float(np.mean([c.confidence for c in conformers]))

    values = np.array([lig_spread, contact_freq, rmsf, inter_var, mean_conf, float(K)])
    # more conformers + higher generator confidence -> more trustworthy summary
    ens_conf = min(1.0, K / 8.0) * mean_conf
    confidence = np.where(np.isfinite(values), ens_conf, 0.0)
    return {"names": NAMES, "values": values, "confidence": confidence.astype(float)}
