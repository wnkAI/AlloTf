"""Ligand graph extractor: the molecule only - atoms, bonds, charge, aromaticity, chirality,
donor/acceptor, hybridisation. No protein, no labels. Cached per SMILES since one ligand recurs
across many candidates.
"""
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

_ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B")
_HYB = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2)
_CACHE = {}


def _elem(sym):
    v = torch.zeros(len(_ELEMENTS) + 1)
    v[_ELEMENTS.index(sym) if sym in _ELEMENTS else len(_ELEMENTS)] = 1.0
    return v


def build_ligand_graph(smiles, embed=True):
    """SMILES -> dict(ligand_atom_features, ligand_edge_index, ligand_edge_features, ligand_coordinates).
    Bonds are stored both directions. A pose is embedded (ETKDG) when possible, else empty coords."""
    if smiles in _CACHE:
        return _CACHE[smiles]
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("cannot parse ligand SMILES: %s" % smiles)
    m = Chem.AddHs(m)
    coords = torch.zeros(0, 3)
    if embed and AllChem.EmbedMolecule(m, randomSeed=1) == 0:
        AllChem.MMFFOptimizeMolecule(m)
        conf = m.GetConformer()
        coords = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())],
                              dtype=torch.float32)
    donor_acceptor = {}
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        is_don = sym in ("N", "O") and a.GetTotalNumHs() > 0
        is_acc = sym in ("N", "O") and a.GetTotalNumHs() == 0
        donor_acceptor[a.GetIdx()] = (is_don, is_acc)

    atom_feats = []
    for a in m.GetAtoms():
        hyb = torch.zeros(len(_HYB) + 1)
        hyb[_HYB.index(a.GetHybridization()) if a.GetHybridization() in _HYB else len(_HYB)] = 1.0
        don, acc = donor_acceptor[a.GetIdx()]
        atom_feats.append(torch.cat([
            _elem(a.GetSymbol()), hyb,
            torch.tensor([float(a.GetFormalCharge()), float(a.GetIsAromatic()),
                          float(a.IsInRing()), float(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
                          float(don), float(acc), float(a.GetTotalNumHs())])]))
    x = torch.stack(atom_feats)

    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = [b.GetBondTypeAsDouble(), float(b.GetIsAromatic()), float(b.IsInRing())]
        for a, bb in ((i, j), (j, i)):
            src.append(a); dst.append(bb); ef.append(feat)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    edge_feat = torch.tensor(ef, dtype=torch.float32) if ef else torch.zeros(0, 3)

    out = dict(ligand_atom_features=x, ligand_edge_index=edge_index,
               ligand_edge_features=edge_feat, ligand_coordinates=coords)
    _CACHE[smiles] = out
    return out
