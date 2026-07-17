"""Explicit chemical identity for the target ligand.

The whole reason AlloSurrogate can beat a shape-first model is that it is TOLD the chemistry rather
than left to guess it from coordinates. LigandMPNN conditions on backbone + ligand atom positions
and recovers native-like sequence; it is weakly sensitive to whether an atom is a donor or an
acceptor, protonated or not, one tautomer or another. Those distinctions are exactly what decides
whether a redesigned pocket binds the target or an isosteric decoy, so they go in as features, not
as something to be inferred.

Every atom carries: element, formal charge, Gasteiger partial charge, degree, total valence,
hybridisation, aromaticity, ring membership, H-bond donor/acceptor role, and (implicitly, through
the protonation/tautomer state fixed upstream) its ionisation. Every bond carries: order,
aromaticity, conjugation, ring membership. 3D coordinates come from the pose, not from here - this
module defines the graph and its chemistry; pose.py places it.

The encoding is deterministic and versioned: a hash of (canonical SMILES, feature layout) travels
with the graph so a surrogate trained on one feature layout never silently scores graphs built by
another.
"""
import hashlib

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False

FEATURE_VERSION = "lig-feats-v1"

# fixed vocabularies -> one-hot slots. An out-of-vocab value lands in the trailing "other" slot
# rather than crashing or silently colliding with a real class.
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def _one_hot(value, vocab):
    v = [0.0] * (len(vocab) + 1)
    v[vocab.index(value) if value in vocab else len(vocab)] = 1.0
    return v


def _require():
    if not _RDKIT:
        raise RuntimeError("ligand encoding requires RDKit (conda install -c conda-forge rdkit)")


def prepare_mol(smiles, add_hs=True, embed=False, seed=0xA11):
    """Parse, sanitise, assign stereochemistry, compute Gasteiger charges.

    Protonation/tautomer state is taken AS GIVEN in the SMILES - it is a modelling decision made
    upstream (route/ligand_params), not re-guessed here, because the same molecule at two pH values
    is two different ligands to a pocket.
    """
    _require()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("unparsable SMILES: %s" % smiles)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    if add_hs:
        m = Chem.AddHs(m)
    AllChem.ComputeGasteigerCharges(m)
    if embed:
        AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def _gasteiger(atom):
    try:
        q = float(atom.GetDoubleProp("_GasteigerCharge"))
        return q if np.isfinite(q) else 0.0
    except (KeyError, ValueError):
        return 0.0


def atom_features(atom):
    """One atom -> feature vector. Chemistry first, then the scalar descriptors."""
    feats = []
    feats += _one_hot(atom.GetSymbol(), ELEMENTS)
    feats += _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
    feats += [
        float(atom.GetFormalCharge()),
        _gasteiger(atom),
        float(atom.GetDegree()),
        float(atom.GetTotalValence()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
    ]
    return feats


def bond_features(bond):
    feats = _one_hot(str(bond.GetBondType()), BOND_TYPES)
    feats += [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetIsAromatic()),
    ]
    return feats


# H-bond donor/acceptor by SMARTS: an atom's role, not just its element, decides isostere
# discrimination. Precompiled once. Evaluated on the IMPLICIT-H molecule (see encode) so H counts
# are the chemical ones, not confused by explicit hydrogens.
# A hydroxyl oxygen is BOTH a donor (O-H) and an acceptor (lone pairs) - the earlier acceptor
# pattern required H0 and wrongly excluded every -OH, the commonest recognition group there is.
_DONOR = Chem.MolFromSmarts("[$([N;!H0]),$([O,S;H1,H2])]") if _RDKIT else None
_ACCEPTOR = Chem.MolFromSmarts(
    "[$([O;v2]),$([S;v2]),$([N;v3;!$([N+]);!$(n)]),$([n;+0])]") if _RDKIT else None


def donor_acceptor_flags(mol):
    """-> {atom_idx: (is_donor, is_acceptor)}. Role features that a purely geometric model misses."""
    d = {a.GetIdx(): [0.0, 0.0] for a in mol.GetAtoms()}
    for (patt, slot) in ((_DONOR, 0), (_ACCEPTOR, 1)):
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            for idx in match:
                d[idx][slot] = 1.0
    return d


def encode(smiles, coords=None):
    """SMILES (+ optional 3D coords aligned to heavy-atom order) -> ligand graph dict.

    -> {
        'atom_features'  : (N, F) float array,
        'bond_index'     : (2, E) int array (both directions),
        'bond_features'  : (E, Fb) float array,
        'coords'         : (N, 3) or None,
        'n_atoms', 'smiles', 'feature_hash'
    }
    coords, when supplied, come from pose.py placing the ligand on a backbone - this module never
    invents a geometry.
    """
    _require()
    # donor/acceptor roles are read on the implicit-H molecule so SMARTS H-counts are chemical.
    # AddHs appends hydrogens AFTER the heavy atoms, so heavy-atom indices are stable and the flags
    # map straight onto the H-added graph; hydrogens themselves are neither donor nor acceptor here.
    m_heavy = prepare_mol(smiles, add_hs=False, embed=False)
    da = donor_acceptor_flags(m_heavy)
    m = Chem.AddHs(m_heavy)
    AllChem.ComputeGasteigerCharges(m)
    atoms = list(m.GetAtoms())
    X = np.array([atom_features(a) + da.get(a.GetIdx(), [0.0, 0.0]) for a in atoms],
                 dtype=np.float32)

    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ef += [f, f]
    bond_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)
    bond_features_arr = np.array(ef, dtype=np.float32) if ef else np.zeros((0, len(BOND_TYPES) + 4),
                                                                          np.float32)

    C = None
    if coords is not None:
        C = np.asarray(coords, dtype=np.float32)
        if C.shape[0] != X.shape[0]:
            raise ValueError("coords has %d atoms but the H-added graph has %d - align coordinates "
                             "to the same atom set (AddHs) before encoding" % (C.shape[0], X.shape[0]))

    return {"atom_features": X,
            "bond_index": bond_index,
            "bond_features": bond_features_arr,
            "coords": C,
            "n_atoms": int(X.shape[0]),
            "smiles": Chem.MolToSmiles(m),
            "feature_hash": feature_hash(smiles)}


def feature_dim():
    """Length of one atom feature vector - the surrogate's input width, computed not hardcoded."""
    return (len(ELEMENTS) + 1) + (len(HYBRIDIZATIONS) + 1) + 8 + 2


def bond_dim():
    return (len(BOND_TYPES) + 1) + 3


def feature_hash(smiles):
    """(canonical SMILES, feature layout) -> short hash. Travels with every graph so a surrogate
    cannot be fed graphs built under a different feature layout than it trained on."""
    m = prepare_mol(smiles, add_hs=False) if _RDKIT else None
    canon = Chem.MolToSmiles(m) if m is not None else smiles
    payload = "|".join([FEATURE_VERSION, canon,
                        ",".join(ELEMENTS), ",".join(HYBRIDIZATIONS), ",".join(BOND_TYPES),
                        str(feature_dim()), str(bond_dim())])
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
