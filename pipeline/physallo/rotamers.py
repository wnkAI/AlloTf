"""Side-chain rotamer library and geometry builder.

Backbone-independent rotamers (Lovell/Dunbrack-style canonical chi values). Backbone-dependent
libraries are better, but they need an external data file; the search below is written so the
library can be swapped without touching the caller.

Design note: LigandMPNN's own numbers show side-chain accuracy collapsing outward
(chi1 84% -> chi4 19%), yet chi3/chi4 are exactly where Arg/Lys/Gln/Glu/Met place the atoms that
make salt bridges and directional H-bonds. So we do NOT trust a learned packer for those; we
enumerate rotamers explicitly and let physics score them.
"""
import math
import numpy as np

G_MINUS, TRANS, G_PLUS = -60.0, 180.0, 60.0
_C3 = (G_MINUS, TRANS, G_PLUS)

# chi angles per residue (IUPAC atom quadruples are handled by Biopython internal_coords)
N_CHI = {
    "ALA": 0, "GLY": 0,
    "SER": 1, "CYS": 1, "THR": 1, "VAL": 1, "PRO": 1,
    "ILE": 2, "LEU": 2, "ASP": 2, "ASN": 2, "HIS": 2, "PHE": 2, "TRP": 2, "TYR": 2,
    "MET": 3, "GLU": 3, "GLN": 3,
    "LYS": 4, "ARG": 4,
}

# canonical chi sets. Terminal chi of planar/symmetric groups is sampled coarsely on purpose.
_LIB = {
    "SER": [(x,) for x in _C3],
    "CYS": [(x,) for x in _C3],
    "THR": [(x,) for x in _C3],
    "VAL": [(x,) for x in _C3],
    "PRO": [(-30.0,), (30.0,)],                       # pucker, not a true rotamer
    "ILE": [(a, b) for a in _C3 for b in (G_MINUS, TRANS)],
    "LEU": [(a, b) for a in (G_MINUS, TRANS) for b in (G_PLUS, TRANS)],
    "MET": [(a, b, c) for a in _C3 for b in _C3 for c in (G_MINUS, TRANS, G_PLUS)],
    "ASP": [(a, b) for a in _C3 for b in (-30.0, 0.0, 30.0)],      # chi2 near-planar
    "ASN": [(a, b) for a in _C3 for b in (-60.0, 0.0, 60.0, 120.0)],
    "GLU": [(a, b, c) for a in _C3 for b in _C3 for c in (-30.0, 0.0, 30.0)],
    "GLN": [(a, b, c) for a in _C3 for b in _C3 for c in (-60.0, 0.0, 60.0)],
    "HIS": [(a, b) for a in _C3 for b in (-75.0, 75.0, 180.0)],
    "PHE": [(a, b) for a in _C3 for b in (-90.0, 90.0)],           # ring is symmetric
    "TYR": [(a, b) for a in _C3 for b in (-90.0, 90.0)],
    "TRP": [(a, b) for a in _C3 for b in (-105.0, -90.0, 90.0, 105.0)],
    "LYS": [(a, b, c, d) for a in _C3 for b in _C3 for c in _C3 for d in (G_MINUS, TRANS, G_PLUS)],
    "ARG": [(a, b, c, d) for a in _C3 for b in _C3 for c in _C3 for d in (-90.0, 90.0, 180.0)],
}
_LIB["ALA"] = [()]
_LIB["GLY"] = [()]

# rough prior: trans is the most common chi1 for most residues; used to break ties, never to gate
_CHI1_PRIOR = {G_MINUS: 0.40, TRANS: 0.40, G_PLUS: 0.20}


def rotamers(resname, max_n=None):
    """-> list of chi tuples (degrees). max_n keeps Lys/Arg (81 x 3) from exploding the search."""
    r = _LIB.get(resname.upper())
    if r is None:
        return [()]
    if max_n and len(r) > max_n:
        # keep the most likely chi1 shells first, then truncate deterministically
        r = sorted(r, key=lambda c: -_CHI1_PRIOR.get(c[0], 0.1) if c else 0)[:max_n]
    return r


def library_size(max_n=None):
    return {k: len(rotamers(k, max_n)) for k in sorted(_LIB)}


def strain(resname, chis):
    """Deviation from the nearest canonical rotamer, in kcal/mol-equivalent proxy units.
    Not a real energy - a penalty that keeps the search from parking side chains in eclipsed wells.
    """
    lib = _LIB.get(resname.upper())
    if not lib or not chis:
        return 0.0
    best = 1e9
    for cand in lib:
        d = 0.0
        for a, b in zip(chis, cand):
            x = math.radians(a - b)
            d += math.degrees(abs(math.atan2(math.sin(x), math.cos(x)))) ** 2
        best = min(best, d)
    return 0.002 * best          # ~2 kcal at 30 deg off a well


def set_chis(residue, chis):
    """Rebuild side-chain coordinates for the given chi angles, in place.

    Uses Biopython internal coordinates so we never hand-roll dihedral geometry.
    Returns False if the residue has no internal_coord (e.g. missing atoms).
    """
    ic = getattr(residue, "internal_coord", None)
    if ic is None:
        return False
    for i, v in enumerate(chis, start=1):
        try:
            ic.set_angle("chi%d" % i, float(v))
        except Exception:
            return False
    return True


def enumerate_states(residue, resname, max_n=None):
    """-> [(chis, strain)] for every rotamer of resname, cheapest bookkeeping only.
    Coordinates are built lazily by the search so we do not rebuild states we will reject."""
    return [(c, strain(resname, c)) for c in rotamers(resname, max_n)]


if __name__ == "__main__":
    sz = library_size()
    print("rotamer library (backbone-independent):")
    for k, v in sz.items():
        print("  %-4s chi=%d  rotamers=%3d" % (k, N_CHI[k], v))
    print("\ntotal states, unbounded :", sum(sz.values()))
    print("total states, max_n=20  :", sum(library_size(20).values()))
    print("\nstrain sanity (ARG):")
    for chis in [(-60, 180, -60, 90), (-55, 175, -65, 85), (-30, 150, -30, 60)]:
        print("   chi=%-24s strain=%.2f" % (str(chis), strain("ARG", chis)))
