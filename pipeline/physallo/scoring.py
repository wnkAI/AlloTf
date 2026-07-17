"""FAST PREFILTER ONLY - clash and geometry screening. This is NOT a force field.

DEMOTED after it failed its first real test: on TtgR/quercetin the global optimum of these terms
was GGGGGGGGGGG - the search emptied the pocket, because side chains only ever cost energy here.
Two whole physical ingredients are missing:
  * reference energy (the unfolded/solution baseline per residue) - without it, "no side chain"
    is free;
  * hydrophobic burial REWARD - we penalise burying polar atoms but never reward burying apolar
    ones, so filling a pocket is pure cost.
Adding them properly means fitting ref terms over thousands of natives, real SASA solvation and
side-chain statistical potentials - i.e. rewriting Rosetta. We do not. PyRosetta ref2015 is the
atom-level energy backend (see rosetta_backend.py); these terms survive only as a cheap screen
that throws out clashing/impossible states before Rosetta is ever called.

    E = E_vdW + E_Coulomb + E_HB(directional) + E_solv + E_strain + E_unsat_polar

Use for: rejecting clashes, checking H-bond geometry, ranking rotamers within one residue type.
Never use for: comparing different residue types, absolute affinity, or any final decision.

Why write this at all instead of trusting a learned packer/designer:
  * a native-sequence prior optimises "what does the PDB usually put here", which is measurably
    insensitive to ligand chemistry (its own element-type ablation barely moved recovery);
  * it has no negative data, so it cannot say why a decoy should NOT bind - these terms can, by
    simply scoring the decoy;
  * directional H-bonds and salt bridges live on chi3/chi4 atoms, where learned packers are 19-28%
    accurate; here the geometry is explicit.
"""
import math
import numpy as np

# --- element parameters: (vdW radius A, LJ well depth kcal/mol) ------------------------------
VDW = {"C": (1.90, 0.086), "N": (1.80, 0.170), "O": (1.70, 0.210),
       "S": (2.00, 0.250), "P": (2.10, 0.200), "F": (1.50, 0.061),
       "CL": (1.95, 0.265), "BR": (2.10, 0.320), "I": (2.35, 0.400), "H": (1.10, 0.016)}
DEFAULT_VDW = (1.90, 0.100)

# formal-charge carriers on protein side chains (proxy partial charges on the key atoms)
PROT_Q = {("ARG", "NH1"): 0.5, ("ARG", "NH2"): 0.5, ("ARG", "NE"): 0.2,
          ("LYS", "NZ"): 1.0, ("HIS", "ND1"): 0.25, ("HIS", "NE2"): 0.25,
          ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
          ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
          ("SER", "OG"): -0.3, ("THR", "OG1"): -0.3, ("TYR", "OH"): -0.3,
          ("ASN", "OD1"): -0.4, ("ASN", "ND2"): 0.3,
          ("GLN", "OE1"): -0.4, ("GLN", "NE2"): 0.3,
          ("TRP", "NE1"): 0.2, ("CYS", "SG"): -0.2}

DONORS = {"N", "O"}          # heavy-atom donor/acceptor proxy (no explicit H in most PDB entries)
ACCEPTORS = {"O", "N", "S"}
HB_MIN, HB_OPT, HB_MAX = 2.6, 2.9, 3.5
HB_WELL = 2.0                # kcal/mol per ideal H-bond
SB_MAX = 4.0
SB_WELL = 2.5
DIEL_SLOPE = 4.0             # distance-dependent dielectric: eps(r) = 4r
BURIAL_R = 6.0               # neighbour count radius used as a burial proxy
BURIED_N = 16                # neighbours above which an atom counts as buried


def _p(el):
    return VDW.get((el or "C").upper(), DEFAULT_VDW)


def vdw(coords_a, els_a, coords_b, els_b, soft=True):
    """Softened LJ 12-6. Softening matters: a hard 1/r^12 makes the rotamer search fall off a
    cliff on any 0.1 A clash and the annealer never recovers."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0, 0
    A, B = np.asarray(coords_a), np.asarray(coords_b)
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    ra = np.array([_p(e)[0] for e in els_a])[:, None]
    rb = np.array([_p(e)[0] for e in els_b])[None, :]
    ea = np.array([_p(e)[1] for e in els_a])[:, None]
    eb = np.array([_p(e)[1] for e in els_b])[None, :]
    rmin = ra + rb
    eps = np.sqrt(ea * eb)
    d = np.clip(d, 0.7, None)
    if soft:
        d = np.sqrt(d ** 2 + 0.25)          # softening core
    x = rmin / d
    e = eps * (x ** 12 - 2 * x ** 6)
    e = np.clip(e, None, 10.0)              # cap per-pair repulsion
    clashes = int((np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) < 0.75 * rmin).sum())
    return float(e[d < 8.0].sum()), clashes


def coulomb(coords_a, q_a, coords_b, q_b):
    """Distance-dependent dielectric eps(r)=4r; screens long-range terms the way a solvated
    protein interior roughly does."""
    if not len(coords_a) or not len(coords_b):
        return 0.0
    qa, qb = np.asarray(q_a), np.asarray(q_b)
    nz_a, nz_b = np.abs(qa) > 1e-3, np.abs(qb) > 1e-3
    if not nz_a.any() or not nz_b.any():
        return 0.0
    A = np.asarray(coords_a)[nz_a]; B = np.asarray(coords_b)[nz_b]
    qa, qb = qa[nz_a], qb[nz_b]
    d = np.clip(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2), 1.5, None)
    e = 332.0 * (qa[:, None] * qb[None, :]) / (DIEL_SLOPE * d * d)
    return float(e[d < 12.0].sum())


def hbonds(don_xyz, don_root, acc_xyz, acc_root):
    """Directional H-bond term. Heavy-atom geometry only (PDB rarely has H):
        distance well around 2.9 A, plus an angular factor on root-donor...acceptor.
    A distance-only criterion counts H-bonds that point the wrong way - which is exactly how a
    design ends up 'making' bonds it cannot make.
    -> (energy, n_hbonds)
    """
    if not len(don_xyz) or not len(acc_xyz):
        return 0.0, 0
    D, A = np.asarray(don_xyz), np.asarray(acc_xyz)
    d = np.linalg.norm(D[:, None, :] - A[None, :, :], axis=2)
    ok = (d > HB_MIN) & (d < HB_MAX)
    if not ok.any():
        return 0.0, 0
    dist_f = np.exp(-((d - HB_OPT) ** 2) / (2 * 0.25 ** 2))
    ang_f = np.ones_like(d)
    if don_root is not None and len(don_root) == len(D):
        R = np.asarray(don_root)
        v1 = D - R
        n1 = np.linalg.norm(v1, axis=1, keepdims=True)
        n1[n1 == 0] = 1
        v1 = v1 / n1
        v2 = A[None, :, :] - D[:, None, :]
        n2 = np.linalg.norm(v2, axis=2, keepdims=True)
        n2[n2 == 0] = 1
        v2 = v2 / n2
        cosang = np.einsum("ik,ijk->ij", v1, v2)
        ang_f = np.clip(cosang, 0.0, 1.0) ** 2          # >90 deg off axis contributes nothing
    e = -HB_WELL * dist_f * ang_f * ok
    return float(e.sum()), int(((e < -0.5 * HB_WELL)).sum())


def salt_bridges(pos_xyz, neg_xyz):
    if not len(pos_xyz) or not len(neg_xyz):
        return 0.0, 0
    P, N = np.asarray(pos_xyz), np.asarray(neg_xyz)
    d = np.linalg.norm(P[:, None, :] - N[None, :, :], axis=2)
    ok = d < SB_MAX
    e = -SB_WELL * np.exp(-((d - 3.2) ** 2) / (2 * 0.5 ** 2)) * ok
    return float(e.sum()), int(ok.sum())


def burial(atom_xyz, env_xyz):
    """Neighbour count within BURIAL_R - a cheap stand-in for SASA."""
    if not len(atom_xyz) or not len(env_xyz):
        return np.zeros(len(atom_xyz))
    A, E = np.asarray(atom_xyz), np.asarray(env_xyz)
    d = np.linalg.norm(A[:, None, :] - E[None, :, :], axis=2)
    return (d < BURIAL_R).sum(1).astype(float)


def desolvation(polar_xyz, env_xyz, w=0.15):
    """Penalty for burying polar atoms: cost grows with neighbour count."""
    if not len(polar_xyz):
        return 0.0
    n = burial(polar_xyz, env_xyz)
    return float(w * np.clip(n - BURIED_N * 0.5, 0, None).sum())


def unsat_polar(polar_xyz, partner_xyz, env_xyz, penalty=2.0):
    """Buried polar atoms with no H-bond partner. This is the single most useful false-positive
    filter in ligand design: a pocket that buries an unpaired donor/acceptor is a pocket that
    does not bind, no matter how good the shape complementarity looks."""
    if not len(polar_xyz):
        return 0.0, 0
    P = np.asarray(polar_xyz)
    n_env = burial(P, env_xyz)
    if len(partner_xyz):
        dp = np.linalg.norm(P[:, None, :] - np.asarray(partner_xyz)[None, :, :], axis=2)
        has = (dp < HB_MAX).any(1)
    else:
        has = np.zeros(len(P), dtype=bool)
    bad = (n_env >= BURIED_N) & (~has)
    return float(penalty * bad.sum()), int(bad.sum())


def total(terms, weights=None):
    w = weights or dict(vdw=1.0, coulomb=1.0, hbond=1.0, salt=1.0,
                        desolv=1.0, unsat=1.0, strain=1.0)
    return float(sum(w.get(k, 1.0) * v for k, v in terms.items()))


if __name__ == "__main__":
    # sanity: the terms must behave, or the whole search is built on sand
    print("vdW distance scan (C...C):")
    for d in (2.0, 3.0, 3.8, 4.5, 6.0):
        e, cl = vdw([[0, 0, 0]], ["C"], [[d, 0, 0]], ["C"])
        print("   d=%.1f  E=%+7.3f  clashes=%d" % (d, e, cl))
    print("\nH-bond geometry (donor root at -1.4 A on x, acceptor moved along x):")
    for d in (2.6, 2.9, 3.2, 3.6):
        e, n = hbonds([[0, 0, 0]], [[-1.4, 0, 0]], [[d, 0, 0]], None)
        print("   d=%.1f aligned    E=%+6.3f  n=%d" % (d, e, n))
    e, n = hbonds([[0, 0, 0]], [[-1.4, 0, 0]], [[0, 2.9, 0]], None)
    print("   d=2.9 perpendicular E=%+6.3f  n=%d   <- must be ~0: wrong direction, no bond" % (e, n))
    print("\nsalt bridge:", "%.2f" % salt_bridges([[0, 0, 0]], [[3.2, 0, 0]])[0])
    print("buried unsatisfied polar:",
          unsat_polar([[0, 0, 0]], [], np.random.RandomState(0).randn(40, 3) * 2.0))
