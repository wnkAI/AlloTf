"""Contact / hbond / salt-bridge networks and their state-to-state changes."""
import numpy as np

CONTACT = 4.5
CLASH = 2.2
HB = 3.5
POS = {"ARG", "LYS", "HIS"}
NEG = {"ASP", "GLU"}


def heavy(res):
    return [a for a in res if a.element != "H"]


def contact_map(chain, resnums=None):
    res = [r for r in chain if r.id[0] == " " and (resnums is None or r.id[1] in resnums)]
    idx = [r.id[1] for r in res]
    A = [np.array([a.coord for a in heavy(r)]) for r in res]
    n = len(res)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 2, n):
            if len(A[i]) and len(A[j]):
                d = np.linalg.norm(A[i][:, None, :] - A[j][None, :, :], axis=2)
                C[i, j] = C[j, i] = (d < CONTACT).sum()
    return idx, C


def contact_delta(idx_a, Ca, idx_b, Cb):
    """dC_ij between two states over shared residues."""
    common = [r for r in idx_a if r in set(idx_b)]
    ia = [idx_a.index(r) for r in common]
    ib = [idx_b.index(r) for r in common]
    return common, Cb[np.ix_(ib, ib)] - Ca[np.ix_(ia, ia)]


def interface(atoms_a, atoms_b):
    """-> (contacts, clashes, min_dist)"""
    if not atoms_a or not atoms_b:
        return 0, 0, float("nan")
    A = np.array([a.coord for a in atoms_a])
    B = np.array([a.coord for a in atoms_b])
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return int((d < CONTACT).sum()), int((d < CLASH).sum()), float(d.min())


def hbonds(atoms_a, atoms_b):
    """TODO(D): angle-aware detection. V1 = donor/acceptor distance only."""
    da = [a for a in atoms_a if a.element in ("N", "O")]
    db = [a for a in atoms_b if a.element in ("N", "O")]
    if not da or not db:
        return 0
    A = np.array([a.coord for a in da])
    B = np.array([a.coord for a in db])
    return int((np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) < HB).sum())
