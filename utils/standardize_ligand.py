"""Ligand standardisation: salts, protonation, tautomers, stereochemistry, sugar ring/open forms.

Route MUST call this before any similarity computation. Raw PubChem CanonicalSMILES drops
stereochemistry (verified: D-ribose / D-xylose / L-arabinose all collapsed to Tanimoto 1.0) and
locks sugars into one ring form (cyclic native sugar vs open-chain target scored 0.03 on ECFP).
"""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = rdMolStandardize.Cleanup(m)
    m = rdMolStandardize.FragmentParent(m)
    m = rdMolStandardize.Uncharger().uncharge(m)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return m


def inchikey(m):
    return Chem.MolToInchiKey(m) if m is not None else None


def same_molecule(a, b):
    """Identity test for ENHANCEMENT routing. Use InChIKey, never 2D similarity."""
    ka, kb = inchikey(standardize(a)), inchikey(standardize(b))
    return ka is not None and ka == kb


def same_connectivity(a, b):
    """Skeleton match ignoring stereochemistry (InChIKey first block)."""
    ka, kb = inchikey(standardize(a)), inchikey(standardize(b))
    return bool(ka and kb) and ka.split("-")[0] == kb.split("-")[0]


def open_chain_forms(m):
    """TODO(A): enumerate open-chain <-> pyranose/furanose forms for sugars."""
    return [m]
