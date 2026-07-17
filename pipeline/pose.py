"""Target ligand pose generation in the scaffold pocket.

Poses must be produced on BOTH backbones - the whole method rests on asking whether the ligand
prefers the induced state:
    pose on X_D (DNA-compatible backbone)
    pose on X_I (induced backbone)

Two placement routes, and which one was used is RECORDED ON EVERY POSE, never inferred:

  mcs_transfer  RDKit constrained embedding onto the maximum common substructure with the native
                effector, whose crystallographic pose is the one piece of ground truth we have.
                Good when the target shares a scaffold with the native ligand - which is exactly
                the case Route selects for (xylitol found via xylose).
  smina         real docking, used when the target shares too little with the native ligand for a
                transfer to mean anything.

A pose with a tiny MCS is worse than no pose: it looks confident and is arbitrary. MIN_MCS_ATOMS
is the line, and falling below it without smina is a hard failure, not a silent degradation.

The same pose set must be generated on X_D and X_I with the SAME method and the SAME conformers -
if X_D poses came from docking and X_I poses from transfer, S_state measures the method, not the
ligand.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False

from utils.contacts import heavy

MIN_MCS_ATOMS = 6       # below this a transferred pose is arbitrary
POSE_RMS_CUT = 0.5      # prune near-duplicate conformers


def _require_rdkit():
    if not _RDKIT:
        raise RuntimeError("pose generation requires RDKit (conda install -c conda-forge rdkit). "
                           "There is no fallback: a hand-placed ligand would be an invented pose.")


def native_ligand_mol(pdb_path, ligand_resname):
    """Pull the crystallographic effector out of the holo structure, with its coordinates.

    Via load_assembly: a multi-MODEL biological assembly hides half the copies behind the first
    model, and in a dimer the effector we want may be the one in the second copy.
    """
    _require_rdkit()
    from .structure import load_assembly
    block = []
    for ch in load_assembly(pdb_path):
        for r in ch:
            if r.id[0] != " " and r.get_resname().strip() == ligand_resname:
                for a in heavy(r):
                    x, y, z = a.coord
                    block.append("HETATM%5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                                 % (len(block) + 1, a.get_name()[:4], ligand_resname, 1,
                                    x, y, z, a.element))
                break
        if block:
            break
    if not block:
        raise ValueError("effector %s not found in %s" % (ligand_resname, pdb_path))
    m = Chem.MolFromPDBBlock("\n".join(block) + "\nEND\n", sanitize=False, removeHs=True)
    if m is None:
        raise ValueError("RDKit could not read %s out of %s" % (ligand_resname, pdb_path))
    return m


def _template_with_bonds(native_pdb_mol, native_smiles):
    """PDB has no bond orders. Assign them from the known SMILES, or the MCS is nonsense."""
    ref = Chem.MolFromSmiles(native_smiles)
    if ref is None:
        raise ValueError("unparsable native SMILES: %s" % native_smiles)
    try:
        return AllChem.AssignBondOrdersFromTemplate(ref, native_pdb_mol)
    except Exception as exc:
        raise ValueError("could not map native SMILES onto the crystal ligand (%s). The SMILES "
                         "and the PDB chemical component disagree." % exc)


def mcs_transfer(target_smiles, template_mol, n_poses=10, seed=0xA110):
    """Constrained embedding of the target onto the native ligand's crystallographic pose.

    -> list of (mol_conformer_id, provenance dict). Raises if the shared substructure is too small
    to anchor anything.
    """
    _require_rdkit()
    tgt = Chem.MolFromSmiles(target_smiles)
    if tgt is None:
        raise ValueError("unparsable target SMILES: %s" % target_smiles)
    tgt = Chem.AddHs(tgt)

    mcs = rdFMCS.FindMCS([Chem.RemoveHs(tgt), template_mol],
                         ringMatchesRingOnly=True, completeRingsOnly=True,
                         timeout=30)
    if mcs.canceled or mcs.numAtoms < MIN_MCS_ATOMS:
        raise ValueError("MCS with the native effector is only %d atoms (need >=%d): a transferred "
                         "pose would be arbitrary. Dock instead."
                         % (mcs.numAtoms, MIN_MCS_ATOMS))
    patt = Chem.MolFromSmarts(mcs.smartsString)
    t_match = tgt.GetSubstructMatch(patt)
    r_match = template_mol.GetSubstructMatch(patt)
    if not t_match or not r_match:
        raise ValueError("MCS pattern did not match back onto both molecules")

    cmap = {}
    conf = template_mol.GetConformer()
    for ti, ri in zip(t_match, r_match):
        p = conf.GetAtomPosition(ri)
        cmap[ti] = p

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(tgt, numConfs=max(n_poses * 4, 20), params=params)
    if not len(cids):
        raise RuntimeError("ETKDG produced no conformers for %s" % target_smiles)

    kept = []
    for cid in cids:
        try:
            AllChem.ConstrainedEmbed(tgt, template_mol, coreConfId=-1, randomseed=seed + cid,
                                     getForceField=AllChem.UFFGetMoleculeForceField)
        except Exception:
            pass
        ff = AllChem.UFFGetMoleculeForceField(tgt, confId=cid)
        if ff is None:
            continue
        # pin the MCS atoms to the crystal coordinates, relax everything else
        for ti in cmap:
            idx = ff.AddExtraPoint(cmap[ti].x, cmap[ti].y, cmap[ti].z, fixed=True) - 1
            ff.AddDistanceConstraint(idx, ti, 0, 0.05, 100.0)
        ff.Minimize(maxIts=500)
        kept.append((cid, float(ff.CalcEnergy())))
    if not kept:
        raise RuntimeError("no conformer survived constrained minimisation")

    kept.sort(key=lambda t: t[1])
    out = _prune(tgt, kept, n_poses)
    prov = {"method": "mcs_transfer", "mcs_atoms": mcs.numAtoms,
            "mcs_smarts": mcs.smartsString, "anchored_atoms": len(cmap)}
    return tgt, out, prov


def _prune(mol, scored, n):
    """Drop conformers that are the same pose twice."""
    keep = []
    for cid, e in scored:
        dup = False
        for kid, _ in keep:
            if AllChem.GetConformerRMS(mol, cid, kid, prealigned=True) < POSE_RMS_CUT:
                dup = True
                break
        if not dup:
            keep.append((cid, e))
        if len(keep) >= n:
            break
    return keep


def dock_smina(target_smiles, receptor_pdb, pocket_center, n_poses=10, box=20.0, exe=None):
    """Real docking for targets too dissimilar to transfer. Requires smina on PATH."""
    _require_rdkit()
    exe = exe or shutil.which("smina") or shutil.which("vina")
    if not exe:
        raise RuntimeError("smina/vina not found on PATH and the target is too dissimilar to the "
                           "native effector for an MCS transfer. Install smina or pick a scaffold "
                           "whose native ligand shares a substructure with the target.")
    m = Chem.AddHs(Chem.MolFromSmiles(target_smiles))
    AllChem.EmbedMolecule(m, randomSeed=0xA110)
    AllChem.UFFOptimizeMolecule(m)
    d = tempfile.mkdtemp(prefix="allotf_dock_")
    lig = os.path.join(d, "lig.sdf")
    out = os.path.join(d, "out.sdf")
    Chem.SDWriter(lig).write(m)
    cx, cy, cz = pocket_center
    cmd = [exe, "-r", receptor_pdb, "-l", lig, "-o", out,
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(box), "--size_y", str(box), "--size_z", str(box),
           "--num_modes", str(n_poses), "--seed", "42"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("smina failed: %s" % (r.stderr[-500:] or r.stdout[-500:]))
    poses = [x for x in Chem.SDMolSupplier(out, removeHs=False) if x is not None]
    return poses, {"method": "smina", "exe": exe}


def pocket_center(pdb_path, pocket_resnums):
    from .structure import load_assembly
    A = [a.coord for ch in load_assembly(pdb_path) for r in ch
         if r.id[0] == " " and r.id[1] in set(pocket_resnums) for a in heavy(r)]
    if not A:
        raise ValueError("no pocket atoms found in %s" % pdb_path)
    return np.array(A).mean(axis=0)


def generate_poses(ligand_smiles, state_path, pocket_residues, n_poses=10,
                   native_pdb=None, native_resname=None, native_smiles=None, method="auto"):
    """-> list of poses (coords + score + provenance).

    method='auto' transfers when the MCS with the native effector is big enough, otherwise docks.
    Every pose records which route produced it: a transferred pose and a docked pose are not
    interchangeable evidence.
    """
    _require_rdkit()
    if method in ("auto", "mcs_transfer") and native_pdb and native_resname and native_smiles:
        try:
            tpl = _template_with_bonds(native_ligand_mol(native_pdb, native_resname), native_smiles)
            mol, kept, prov = mcs_transfer(ligand_smiles, tpl, n_poses)
            return [{"mol": mol, "conf_id": cid, "internal_energy": e,
                     "provenance": dict(prov, backbone=state_path)} for cid, e in kept]
        except ValueError as exc:
            if method == "mcs_transfer":
                raise
            reason = str(exc)
    else:
        reason = "no native ligand reference supplied"
    poses, prov = dock_smina(ligand_smiles, state_path,
                             pocket_center(state_path, pocket_residues), n_poses)
    return [{"mol": p, "conf_id": 0, "internal_energy": None,
             "provenance": dict(prov, backbone=state_path, transfer_declined=reason)}
            for p in poses]


def generate_on_both(ligand_smiles, x_d, x_i, pocket_residues, n_poses=10, **kw):
    """The only supported entry point for the pipeline: one method, both backbones.

    Refuses to hand back a set where the two backbones were posed differently - S_state would then
    be a difference of methods rather than of states.
    """
    pd = generate_poses(ligand_smiles, x_d, pocket_residues, n_poses, **kw)
    pi = generate_poses(ligand_smiles, x_i, pocket_residues, n_poses, **kw)
    md = {p["provenance"]["method"] for p in pd}
    mi = {p["provenance"]["method"] for p in pi}
    if md != mi:
        raise RuntimeError("X_D poses used %s but X_I poses used %s: S_state would measure the "
                           "placement method, not the ligand's state preference." % (md, mi))
    return {"X_D": pd, "X_I": pi, "method": sorted(md)[0] if md else None}


def run(ctx):
    """requires ctx['states'], ctx['masks']; produces ctx['poses']"""
    st = ctx["states"]
    rt = ctx["route"]
    poses = generate_on_both(rt["target_smiles"], st["paths"]["X_D"], st["paths"]["X_I"],
                             ctx["masks"]["recognition_mask"],
                             n_poses=ctx["cfg"]["design"].get("n_poses", 10),
                             native_pdb=st["paths"].get("X_I_lig") or st["paths"].get("X_D_lig"),
                             native_resname=st.get("effector_resname"),
                             native_smiles=rt.get("native_smiles"))
    return {"poses": poses}
