"""Native RDKit -> Rosetta MutableResidueType, no molfile_to_params.py (absent from this wheel).

Rebuilds a Rosetta RWMol atom-by-atom from a python RDKit mol (coords + bond orders + formal
charges), runs RDMolToRestype.generate_restype(), and returns a MutableResidueType that can be
added to a pose's residue type set. Every one of the six states then loads the SAME residue type.
"""
import pyrosetta
from rdkit import Chem

_BONDNAME = None
def _bondname():
    global _BONDNAME
    if _BONDNAME is None:
        from pyrosetta.rosetta.core.chemical import BondName
        _BONDNAME = {1.0: BondName.SingleBond, 2.0: BondName.DoubleBond,
                     3.0: BondName.TripleBond, 1.5: BondName.AromaticBond}
    return _BONDNAME


def rdkit_to_restype(mol, name="LIG"):
    """python RDKit mol (3D, explicit Hs, one conformer) -> Rosetta MutableResidueType."""
    import pyrosetta.rosetta.RDKit as PRD
    import pyrosetta.rosetta.RDGeom as G
    from pyrosetta.rosetta.core.chemical.rdkit import RDMolToRestype, convert_to_rdkit_bondtype

    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no 3D conformer")
    conf = mol.GetConformer()

    rw = PRD.RWMol()
    for a in mol.GetAtoms():
        ra = PRD.Atom(a.GetAtomicNum())
        ra.setFormalCharge(a.GetFormalCharge())
        rw.addAtom(ra, True, False)
    coords = PRD.Conformer(mol.GetNumAtoms())
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        coords.setAtomPos(a.GetIdx(), G.Point3D(p.x, p.y, p.z))
    rw.addConformer(coords, True)

    names = _bondname()
    for b in mol.GetBonds():
        bt = convert_to_rdkit_bondtype(names.get(b.GetBondTypeAsDouble(), names[1.0]))
        rw.addBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bt)

    converter = RDMolToRestype(rw)
    converter.set_nbr(0)
    restype = converter.generate_restype()
    restype.name(name); restype.name3(name[:3]); restype.name1("Z")
    return restype


if __name__ == "__main__":
    from rdkit.Chem import AllChem
    pyrosetta.init("-mute all")
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@@H]1O)CO)O)O"))
    AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
    print("python rdkit mol:", m.GetNumAtoms(), "atoms", flush=True)
    rt = rdkit_to_restype(m, "TGT")
    print("MutableResidueType built: natoms=%d name=%s" % (rt.natoms(), rt.name()), flush=True)
