"""Empirical interaction terms. RELATIVE proxies on one scaffold - never report as Kd."""
from utils.contacts import interface, hbonds


def interface_energy(atoms_a, atoms_b, w=None):
    w = w or dict(hbond=1.0, vdw=1.0, clash=5.0)
    c, cl, mn = interface(atoms_a, atoms_b)
    hb = hbonds(atoms_a, atoms_b)
    e = -(w["vdw"] * c * 0.05) - (w["hbond"] * hb * 0.5) + (w["clash"] * cl)
    return dict(energy=e, contacts=c, clashes=cl, hbonds=hb, min_dist=mn)


def buried_unsat_polar(structure, ligand_atoms):
    """TODO(D): buried polar atoms without a partner - key false-positive filter."""
    return 0


def fold_clash(chain):
    """TODO(D): internal clash count after repack."""
    return 0
