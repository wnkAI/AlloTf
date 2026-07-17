"""Build side-chain heavy atoms from backbone + chi angles (Z-matrix / NeRF).

Needed because design mutates residue IDENTITY: Biopython's set_chis can re-place an existing side
chain, but it cannot turn LEU into ARG - the atoms do not exist. Every rotamer of every candidate
residue has to be constructed from the backbone.

V1 keeps the heavy-atom chain that carries the interactions (CB -> ... -> terminal polar/aromatic
atoms) and ignores hydrogens: PDB entries mostly lack them, and the H-bond term is heavy-atom
geometric with an explicit angular factor.

Ideal internal coordinates from standard amino-acid geometry. These are relative proxies for
ranking, not a refinement force field.
"""
import math
import numpy as np

# (name, parent_a, parent_b, parent_c, bond, angle_deg, dihedral_spec)
# dihedral_spec: ('chi', i) -> chi_i ;  float -> fixed offset from the preceding chi ;
#                ('fix', v) -> absolute
SC = {
    "ALA": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5))],
    "SER": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("OG", "N", "CA", "CB", 1.42, 111.0, ("chi", 1))],
    "CYS": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("SG", "N", "CA", "CB", 1.81, 114.0, ("chi", 1))],
    "THR": [("CB", "N", "C", "CA", 1.54, 111.5, ("fix", 122.5)),
            ("OG1", "N", "CA", "CB", 1.43, 109.5, ("chi", 1)),
            ("CG2", "N", "CA", "CB", 1.52, 110.5, ("off", 1, -120.0))],
    "VAL": [("CB", "N", "C", "CA", 1.54, 111.5, ("fix", 122.5)),
            ("CG1", "N", "CA", "CB", 1.52, 110.5, ("chi", 1)),
            ("CG2", "N", "CA", "CB", 1.52, 110.5, ("off", 1, 122.0))],
    "LEU": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.53, 116.0, ("chi", 1)),
            ("CD1", "CA", "CB", "CG", 1.52, 110.5, ("chi", 2)),
            ("CD2", "CA", "CB", "CG", 1.52, 110.5, ("off", 2, 122.0))],
    "ILE": [("CB", "N", "C", "CA", 1.54, 111.5, ("fix", 122.5)),
            ("CG1", "N", "CA", "CB", 1.53, 110.5, ("chi", 1)),
            ("CG2", "N", "CA", "CB", 1.52, 110.5, ("off", 1, -122.0)),
            ("CD1", "CA", "CB", "CG1", 1.52, 114.0, ("chi", 2))],
    "MET": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 114.0, ("chi", 1)),
            ("SD", "CA", "CB", "CG", 1.80, 112.7, ("chi", 2)),
            ("CE", "CB", "CG", "SD", 1.79, 100.9, ("chi", 3))],
    "ASP": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 112.6, ("chi", 1)),
            ("OD1", "CA", "CB", "CG", 1.25, 118.4, ("chi", 2)),
            ("OD2", "CA", "CB", "CG", 1.25, 118.4, ("off", 2, 180.0))],
    "GLU": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 114.0, ("chi", 1)),
            ("CD", "CA", "CB", "CG", 1.52, 112.6, ("chi", 2)),
            ("OE1", "CB", "CG", "CD", 1.25, 118.4, ("chi", 3)),
            ("OE2", "CB", "CG", "CD", 1.25, 118.4, ("off", 3, 180.0))],
    "ASN": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 112.6, ("chi", 1)),
            ("OD1", "CA", "CB", "CG", 1.23, 120.8, ("chi", 2)),
            ("ND2", "CA", "CB", "CG", 1.33, 116.4, ("off", 2, 180.0))],
    "GLN": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 114.0, ("chi", 1)),
            ("CD", "CA", "CB", "CG", 1.52, 112.6, ("chi", 2)),
            ("OE1", "CB", "CG", "CD", 1.23, 120.8, ("chi", 3)),
            ("NE2", "CB", "CG", "CD", 1.33, 116.4, ("off", 3, 180.0))],
    "LYS": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 114.0, ("chi", 1)),
            ("CD", "CA", "CB", "CG", 1.52, 111.3, ("chi", 2)),
            ("CE", "CB", "CG", "CD", 1.52, 111.3, ("chi", 3)),
            ("NZ", "CG", "CD", "CE", 1.49, 111.9, ("chi", 4))],
    "ARG": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.52, 114.0, ("chi", 1)),
            ("CD", "CA", "CB", "CG", 1.52, 111.3, ("chi", 2)),
            ("NE", "CB", "CG", "CD", 1.46, 112.0, ("chi", 3)),
            ("CZ", "CG", "CD", "NE", 1.33, 124.2, ("chi", 4)),
            ("NH1", "CD", "NE", "CZ", 1.33, 120.0, ("fix", 0.0)),
            ("NH2", "CD", "NE", "CZ", 1.33, 120.0, ("fix", 180.0))],
    "HIS": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.50, 113.8, ("chi", 1)),
            ("ND1", "CA", "CB", "CG", 1.38, 122.7, ("chi", 2)),
            ("CD2", "CA", "CB", "CG", 1.35, 131.0, ("off", 2, 180.0)),
            ("CE1", "CB", "CG", "ND1", 1.32, 109.0, ("fix", 180.0)),
            ("NE2", "CB", "CG", "CD2", 1.37, 107.0, ("fix", 180.0))],
    "PHE": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.50, 113.8, ("chi", 1)),
            ("CD1", "CA", "CB", "CG", 1.39, 120.8, ("chi", 2)),
            ("CD2", "CA", "CB", "CG", 1.39, 120.8, ("off", 2, 180.0)),
            ("CE1", "CB", "CG", "CD1", 1.39, 120.8, ("fix", 180.0)),
            ("CE2", "CB", "CG", "CD2", 1.39, 120.8, ("fix", 180.0)),
            ("CZ", "CG", "CD1", "CE1", 1.39, 120.0, ("fix", 0.0))],
    "TYR": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.51, 113.8, ("chi", 1)),
            ("CD1", "CA", "CB", "CG", 1.39, 120.8, ("chi", 2)),
            ("CD2", "CA", "CB", "CG", 1.39, 120.8, ("off", 2, 180.0)),
            ("CE1", "CB", "CG", "CD1", 1.39, 120.8, ("fix", 180.0)),
            ("CE2", "CB", "CG", "CD2", 1.39, 120.8, ("fix", 180.0)),
            ("CZ", "CG", "CD1", "CE1", 1.38, 120.0, ("fix", 0.0)),
            ("OH", "CD1", "CE1", "CZ", 1.38, 119.9, ("fix", 180.0))],
    "TRP": [("CB", "N", "C", "CA", 1.53, 110.5, ("fix", 122.5)),
            ("CG", "N", "CA", "CB", 1.50, 113.6, ("chi", 1)),
            ("CD1", "CA", "CB", "CG", 1.37, 127.0, ("chi", 2)),
            ("CD2", "CA", "CB", "CG", 1.43, 126.6, ("off", 2, 180.0)),
            ("NE1", "CB", "CG", "CD1", 1.38, 110.2, ("fix", 180.0)),
            ("CE2", "CB", "CG", "CD2", 1.41, 107.2, ("fix", 180.0))],
    "GLY": [],
    "PRO": [("CB", "N", "C", "CA", 1.53, 103.2, ("fix", 115.0)),
            ("CG", "N", "CA", "CB", 1.49, 104.5, ("chi", 1)),
            ("CD", "CA", "CB", "CG", 1.50, 105.5, ("fix", 30.0))],
}
POLAR = {"N", "O", "S"}


def place(a, b, c, bond, angle, dihedral):
    """NeRF: place atom D given A-B-C and internal coords (angle/dihedral in degrees)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ang, dih = math.radians(angle), math.radians(dihedral)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-8 else np.array([0.0, 0.0, 1.0])
    m = np.cross(n, bc)
    d2 = np.array([-bond * math.cos(ang),
                   bond * math.sin(ang) * math.cos(dih),
                   bond * math.sin(ang) * math.sin(dih)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def build(resname, bb, chis):
    """bb: {'N':xyz,'CA':xyz,'C':xyz}; chis: tuple of degrees.
    -> [(atom_name, xyz, element)] heavy side-chain atoms (CB onwards)."""
    resname = resname.upper()
    tmpl = SC.get(resname)
    if not tmpl:
        return []
    pos = {k: np.asarray(v, float) for k, v in bb.items()}
    out = []
    for name, pa, pb, pc, bond, angle, spec in tmpl:
        if pa not in pos or pb not in pos or pc not in pos:
            continue
        kind = spec[0]
        if kind == "chi":
            i = spec[1]
            if len(chis) < i:
                break                      # rotamer does not define this chi -> stop the chain
            dih = float(chis[i - 1])
        elif kind == "off":
            i, off = spec[1], spec[2]
            if len(chis) < i:
                break
            dih = float(chis[i - 1]) + off
        else:
            dih = float(spec[1])
        xyz = place(pos[pa], pos[pb], pos[pc], bond, angle, dih)
        pos[name] = xyz
        out.append((name, xyz, name[0]))
    return out


def polar_atoms(built):
    return [(n, x) for n, x, e in built if e in POLAR]


if __name__ == "__main__":
    # geometry sanity: bond lengths and chi angles must come back out as they went in
    bb = {"N": [0.0, 0.0, 0.0], "CA": [1.458, 0.0, 0.0], "C": [2.0, 1.42, 0.0]}

    def dihedral(p0, p1, p2, p3):
        # praxeolitic formula - note b0 points BACKWARDS (p0 - p1). Dropping that sign flips
        # every angle by exactly 180 deg, which looks like a geometry bug but is a checker bug.
        b0 = np.asarray(p0, float) - np.asarray(p1, float)
        b1 = np.asarray(p2, float) - np.asarray(p1, float)
        b2 = np.asarray(p3, float) - np.asarray(p2, float)
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))

    print("build ARG with chi=(-60, 180, -60, 90):")
    at = build("ARG", bb, (-60, 180, -60, 90))
    d = {n: x for n, x, _ in at}
    for n, x, e in at:
        print("   %-4s %-2s %s" % (n, e, np.round(x, 2)))
    print("\ngeometry check:")
    print("   CA-CB bond   = %.3f A   (ideal 1.53)" % np.linalg.norm(d["CB"] - np.array(bb["CA"])))
    print("   CB-CG bond   = %.3f A   (ideal 1.52)" % np.linalg.norm(d["CG"] - d["CB"]))
    chi1 = dihedral(bb["N"], bb["CA"], d["CB"], d["CG"])
    chi2 = dihedral(bb["CA"], d["CB"], d["CG"], d["CD"])
    print("   chi1 rebuilt = %+.1f deg  (asked -60)" % chi1)
    print("   chi2 rebuilt = %+.1f deg  (asked 180)" % chi2)
    print("\nrotamer spread of ARG NH1 (guanidinium reach):")
    for chis in [(-60, 180, -60, 90), (180, 180, 180, 180), (60, -60, 180, -90)]:
        a = {n: x for n, x, _ in build("ARG", bb, chis)}
        print("   chi=%-22s NH1 at %s  |CA->NH1| = %.2f A"
              % (str(chis), np.round(a["NH1"], 1), np.linalg.norm(a["NH1"] - np.array(bb["CA"]))))
    print("\npolar atoms of ARG:", [n for n, _ in polar_atoms(at)])
    print("built residue types:", len([k for k, v in SC.items() if v]))
