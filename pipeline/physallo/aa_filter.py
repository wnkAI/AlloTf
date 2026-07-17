"""Per-position allowed-amino-acid pre-filter.

Opening all 20 residues at 11 positions is 20^11 = 2e14 sequences - not a search space, a fantasy.
Physics and geometry cut it to 3-8 per position BEFORE any search, using only local facts:

    volume        does the side chain physically fit in the cavity at this position?
    burial        buried positions cannot afford a desolvation-costly charge
    ligand groups what does the ligand present at this position - carboxylate? aromatic? polyol?
    backbone      phi/psi that only Gly tolerates; Pro that would break the helix
    conservation  family-conserved positions are constrained (soft, never a hard veto alone)
    mask          recognition = free, transduction = limited, protected = fixed

This is where chemistry enters explicitly - exactly the information a native-sequence prior was
shown to be able to ignore (its element-type ablation barely moved recovery).
"""
import numpy as np
from .scoring import burial, BURIED_N

# side-chain heavy-atom volume (A^3) and properties
AA = {
    "GLY": dict(vol=0,   cls="tiny",  polar=False, charge=0,  arom=False, hbd=0, hba=0),
    "ALA": dict(vol=27,  cls="tiny",  polar=False, charge=0,  arom=False, hbd=0, hba=0),
    "SER": dict(vol=36,  cls="small", polar=True,  charge=0,  arom=False, hbd=1, hba=1),
    "CYS": dict(vol=45,  cls="small", polar=False, charge=0,  arom=False, hbd=1, hba=1),
    "THR": dict(vol=57,  cls="small", polar=True,  charge=0,  arom=False, hbd=1, hba=1),
    "PRO": dict(vol=58,  cls="small", polar=False, charge=0,  arom=False, hbd=0, hba=0),
    "VAL": dict(vol=71,  cls="med",   polar=False, charge=0,  arom=False, hbd=0, hba=0),
    "ASN": dict(vol=73,  cls="med",   polar=True,  charge=0,  arom=False, hbd=1, hba=1),
    "ASP": dict(vol=71,  cls="med",   polar=True,  charge=-1, arom=False, hbd=0, hba=2),
    "ILE": dict(vol=93,  cls="med",   polar=False, charge=0,  arom=False, hbd=0, hba=0),
    "LEU": dict(vol=93,  cls="med",   polar=False, charge=0,  arom=False, hbd=0, hba=0),
    "MET": dict(vol=94,  cls="med",   polar=False, charge=0,  arom=False, hbd=0, hba=1),
    "GLN": dict(vol=94,  cls="med",   polar=True,  charge=0,  arom=False, hbd=1, hba=1),
    "GLU": dict(vol=90,  cls="med",   polar=True,  charge=-1, arom=False, hbd=0, hba=2),
    "LYS": dict(vol=100, cls="large", polar=True,  charge=1,  arom=False, hbd=1, hba=0),
    "HIS": dict(vol=98,  cls="large", polar=True,  charge=0,  arom=True,  hbd=1, hba=1),
    "PHE": dict(vol=124, cls="large", polar=False, charge=0,  arom=True,  hbd=0, hba=0),
    "ARG": dict(vol=134, cls="large", polar=True,  charge=1,  arom=False, hbd=3, hba=0),
    "TYR": dict(vol=130, cls="large", polar=True,  charge=0,  arom=True,  hbd=1, hba=1),
    "TRP": dict(vol=163, cls="xlarge",polar=False, charge=0,  arom=True,  hbd=1, hba=0),
}
ALL = list(AA)

THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
                "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
                "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def one(resname):
    """Never use resname[0]: GLU->G, ASN->A, GLN->G, PHE->P are all wrong."""
    return THREE_TO_ONE.get((resname or "").upper(), "X")

# what the ligand presents nearby -> which residues complement it
COMPLEMENT = {
    "carboxylate": ["ARG", "LYS", "HIS", "SER", "THR", "ASN", "GLN", "TYR"],
    "amine":       ["ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR"],
    "hydroxyl":    ["ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR", "HIS"],
    "carbonyl":    ["ARG", "LYS", "SER", "THR", "ASN", "GLN", "TYR", "TRP", "HIS"],
    "aromatic":    ["PHE", "TYR", "TRP", "HIS", "LEU", "ILE", "VAL", "MET"],
    "aliphatic":   ["ALA", "VAL", "LEU", "ILE", "MET", "PHE"],
    "halogen":     ["PHE", "TYR", "TRP", "MET", "LEU", "ILE"],
    "cation":      ["ASP", "GLU", "TRP", "TYR", "PHE"],       # incl. cation-pi
    "anion":       ["ARG", "LYS", "HIS"],
}


def ligand_groups_near(pos_xyz, lig_atoms, cutoff=6.0):
    """lig_atoms: list of dict(xyz, element, charge, aromatic, is_donor, is_acceptor, in_ring)
    -> set of group labels presented to this position."""
    g = set()
    P = np.asarray(pos_xyz)
    for a in lig_atoms:
        if np.linalg.norm(np.asarray(a["xyz"]) - P) > cutoff:
            continue
        el, q = a.get("element", "C").upper(), a.get("charge", 0.0)
        if a.get("aromatic"):
            g.add("aromatic")
        if q <= -0.5:
            g.add("carboxylate" if el == "O" else "anion")
        elif q >= 0.5:
            g.add("cation" if el == "N" else "cation")
        if el == "O":
            g.add("hydroxyl" if a.get("is_donor") else "carbonyl")
        elif el == "N":
            g.add("amine" if a.get("is_donor") else "carbonyl")
        elif el in ("F", "CL", "BR", "I"):
            g.add("halogen")
        elif el == "C" and not a.get("aromatic"):
            g.add("aliphatic")
    return g


def cavity_volume(pos_xyz, env_xyz, probe=1.4, rmax=7.0):
    """Crude free volume around a design position: count empty probe-sized grid points."""
    P = np.asarray(pos_xyz)
    if not len(env_xyz):
        return 200.0
    E = np.asarray(env_xyz)
    E = E[np.linalg.norm(E - P, axis=1) < rmax + 3]
    if not len(E):
        return 200.0
    step = 1.0
    rng = np.arange(-rmax, rmax + step, step)
    grid = np.array([[x, y, z] for x in rng for y in rng for z in rng
                     if x * x + y * y + z * z <= rmax * rmax]) + P
    d = np.linalg.norm(grid[:, None, :] - E[None, :, :], axis=2).min(1)
    return float((d > probe + 1.5).sum() * step ** 3)


def allowed(pos_xyz, env_xyz, lig_atoms, wt_resname, mask_type,
            conservation=None, bb_phi=None, n_min=3, n_max=8):
    """-> list of allowed residue names for one design position.

    mask_type: 'recognition' (free) | 'transduction' (limited) | 'protected' (fixed to WT)
    conservation: 0..1 from the family alignment; high = constrained (soft)
    """
    wt = wt_resname.upper()
    if mask_type == "protected":
        return [wt]

    vol = cavity_volume(pos_xyz, env_xyz)
    nb = float(burial([pos_xyz], env_xyz)[0]) if len(env_xyz) else 0.0
    groups = ligand_groups_near(pos_xyz, lig_atoms)

    cand = []
    for aa, p in AA.items():
        if p["vol"] > vol * 0.85:                      # must physically fit
            continue
        if nb >= BURIED_N and abs(p["charge"]) > 0:    # buried charge: desolvation kills it
            if not (groups & {"carboxylate", "anion", "cation", "amine"}):
                continue                               # allowed only if the ligand pays it back
        if bb_phi is not None and bb_phi > 0 and aa not in ("GLY", "ASN", "ASP"):
            continue                                   # positive phi is Gly territory
        cand.append(aa)

    if groups:
        pref = set()
        for g in groups:
            pref.update(COMPLEMENT.get(g, []))
        ranked = [a for a in cand if a in pref] + [a for a in cand if a not in pref]
    else:
        # No ligand group within reach (position points away from, or is far from, the ligand).
        # Ranking by dict order here silently means "prefer the smallest residues", which strips
        # the pocket bare and even drops WT. Rank by chemical similarity to WT instead.
        wp = AA.get(wt, AA["ALA"])
        def sim(a):
            p2 = AA[a]
            return (p2["cls"] == wp["cls"]) * 2 + (p2["polar"] == wp["polar"])                    + (p2["charge"] == wp["charge"]) + (p2["arom"] == wp["arom"])                    - abs(p2["vol"] - wp["vol"]) / 200.0
        ranked = sorted(cand, key=lambda a: -sim(a))

    if conservation is not None and conservation > 0.8:
        # strongly conserved: WT plus its closest chemical neighbours only
        cls = AA[wt]["cls"]
        ranked = [a for a in ranked if a == wt or
                  (AA[a]["cls"] == cls and AA[a]["polar"] == AA[wt]["polar"])]

    if mask_type == "transduction":
        # Compensatory mutations only. Same size class AND same charge state: a transduction
        # position sits on the pocket->DBD path, and swapping a neutral side chain for a charged
        # one there buries a desolvation penalty inside the very packing that carries the signal.
        ranked = [a for a in ranked if a == wt or
                  (AA[a]["cls"] == AA[wt]["cls"] and AA[a]["charge"] == AA[wt]["charge"])]

    if wt in AA:
        # WT must survive the [:n_max] cut, not just exist somewhere down the list
        ranked = [wt] + [a for a in ranked if a != wt]
    out = ranked[:n_max]
    if len(out) < n_min:                               # never starve a position
        extra = [a for a in ALL if a not in out and AA[a]["vol"] <= max(vol * 0.85, 60)]
        out += extra[:n_min - len(out)]
    return out


def space_size(allowed_per_pos):
    n = 1
    for a in allowed_per_pos:
        n *= max(len(a), 1)
    return n


if __name__ == "__main__":
    rs = np.random.RandomState(0)
    env = rs.randn(80, 3) * 3.0 + np.array([3.0, 0, 0])
    carbox = [dict(xyz=[1.0, 0, 0], element="O", charge=-0.6, is_donor=False, is_acceptor=True),
              dict(xyz=[1.6, 1.0, 0], element="O", charge=-0.6, is_donor=False, is_acceptor=True)]
    arom = [dict(xyz=[1.2, 0, 0], element="C", charge=0.0, aromatic=True)]
    for tag, lig in (("carboxylate nearby", carbox), ("aromatic nearby", arom), ("no ligand", [])):
        a = allowed([0, 0, 0], env, lig, "LEU", "recognition")
        print("%-20s -> %d allowed: %s" % (tag, len(a), ", ".join(a)))
    print("\nmask behaviour (WT=LEU):")
    for m in ("recognition", "transduction", "protected"):
        a = allowed([0, 0, 0], env, carbox, "LEU", m)
        print("  %-13s -> %s" % (m, ", ".join(a)))
    print("\nsearch space, 11 positions x 6 aa :  %.2e sequences" % space_size([["A"] * 6] * 11))
    print("vs all 20 open                    :  %.2e" % space_size([["A"] * 20] * 11))
