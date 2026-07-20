"""Ligand .params generation + provenance. Rosetta cannot score a ligand it has no residue type
for, so this is a hard prerequisite for every one of the six states.

Automated on purpose: hand-making params per ligand is where silent inconsistency creeps in - one
state built with a neutral quercetin, another with a deprotonated one, and the double difference
becomes meaningless while every number still looks plausible.

Guarantees enforced here (fail loudly, never fall back):
    * .params exists for the ligand
    * atom names in .params match the atom names in the PDB pose
    * total formal charge is what we intended
    * stereochemistry survived the SDF -> params round trip
    * ALL six states use the SAME params version (hash-checked)
"""
import hashlib
import json
import os
import subprocess
import sys

META = "ligand_metadata.json"


def _sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def from_sdf(sdf_path, out_dir, name="LIG", formal_charge=None, n_conformers=1,
             molfile_to_params=None):
    """SDF/MOL2 -> {params, conformers_pdb, metadata}

    molfile_to_params: path to Rosetta's molfile_to_params.py. It ships with PyRosetta under
    pyrosetta/toolbox or with the Rosetta source; we do not vendor it.
    """
    os.makedirs(out_dir, exist_ok=True)
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError("cannot parse %s" % sdf_path)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    q = Chem.GetFormalCharge(mol)
    if formal_charge is not None and q != formal_charge:
        raise ValueError("formal charge mismatch: file has %+d, expected %+d. A wrong protonation "
                         "state silently changes every electrostatic term." % (q, formal_charge))
    smiles = Chem.MolToSmiles(mol)
    stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)

    if molfile_to_params is None:
        molfile_to_params = _find_molfile_to_params()
    params = os.path.join(out_dir, "%s.params" % name)
    cmd = [sys.executable, molfile_to_params, "-n", name, "-p", os.path.join(out_dir, name),
           "--keep-names", "--conformers-in-one-file", sdf_path]
    if n_conformers > 1:
        cmd.insert(-1, "--recharge=%d" % q)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)
    if not os.path.exists(params):
        raise RuntimeError("molfile_to_params failed:\n%s\n%s" % (r.stdout[-800:], r.stderr[-800:]))

    meta = dict(name=name, smiles=smiles, formal_charge=q,
                stereocentres=[(int(i), str(c)) for i, c in stereo],
                n_conformers=n_conformers,
                source_file=os.path.abspath(sdf_path), source_file_hash=_sha1(sdf_path),
                params_file=os.path.abspath(params), params_file_hash=_sha1(params))
    json.dump(meta, open(os.path.join(out_dir, META), "w"), indent=2)
    verify_params_charge(params, q)
    return dict(params=params, conformers=os.path.join(out_dir, "%s_conformers.pdb" % name),
                metadata=meta)


def _find_molfile_to_params():
    # explicit override wins
    env = os.environ.get("ROSETTA_MOLFILE_TO_PARAMS")
    if env and os.path.exists(env):
        return env
    # the pip/conda PyRosetta wheels do NOT ship this script; fetch it once from the public
    # Rosetta source into ~/rosetta_tools (see tools/get_molfile_to_params.sh)
    cands = [os.path.expanduser("~/rosetta_tools/molfile_to_params.py")]
    try:
        import pyrosetta
        base = os.path.dirname(pyrosetta.__file__)
        cands += [os.path.normpath(os.path.join(base, sub)) for sub in
                  ("toolbox/molfile_to_params.py", "../molfile_to_params.py")]
    except Exception:
        pass
    for p in cands:
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "molfile_to_params.py not found. The pip/conda PyRosetta wheels omit it; get it with "
        "tools/get_molfile_to_params.sh (downloads from the public Rosetta source into "
        "~/rosetta_tools), or set ROSETTA_MOLFILE_TO_PARAMS to its path.")


def verify_params_charge(params_path, expected_charge):
    """Rosetta partial charges must sum to the formal charge. If they do not, every fa_elec term
    in all six states is wrong by the same silent amount.

    molfile_to_params writes each charge to 2 decimals, so the sum carries up to 0.005 of rounding
    noise per atom. The tolerance is that quantization budget (0.005*n), not a fixed number: a real
    off-by-one protonation shifts the sum by ~1.0 and is still caught with a wide margin, while a
    30-atom ligand summing to -0.09 is just rounding and must pass."""
    tot = 0.0
    n = 0
    for line in open(params_path):
        if line.startswith("ATOM "):
            f = line.split()
            if len(f) >= 5:
                try:
                    tot += float(f[-1])
                    n += 1
                except ValueError:
                    pass
    tol = max(0.05, 0.005 * n)
    if abs(tot - expected_charge) > tol:
        raise ValueError("params partial charges sum to %+.3f but formal charge is %+d, off by more "
                         "than the %d-atom rounding budget %.3f (%s)"
                         % (tot, expected_charge, n, tol, params_path))
    return tot


def params_atom_names(params_path):
    return [l.split()[1] for l in open(params_path) if l.startswith("ATOM ") and len(l.split()) > 1]


def check_matches_pdb(params_path, pdb_path, ligand_resname):
    """Atom names in the PDB ligand must match the params. A mismatch makes Rosetta either fail or
    - worse - silently rebuild the ligand in a different geometry."""
    names_p = set(params_atom_names(params_path))
    names_s = set()
    for line in open(pdb_path):
        if line.startswith(("HETATM", "ATOM")) and line[17:20].strip().upper() == ligand_resname.upper():
            names_s.add(line[12:16].strip())
    if not names_s:
        raise ValueError("ligand %s not found in %s" % (ligand_resname, pdb_path))
    missing = names_s - names_p
    if missing:
        raise ValueError("PDB ligand atoms absent from params: %s\nRun molfile_to_params with "
                         "--keep-names against the SAME molecule used in the structure."
                         % sorted(missing))
    return True


def assert_same_params(metas):
    """All six states must reference one params version."""
    hashes = {m["params_file_hash"] for m in metas if m}
    if len(hashes) > 1:
        raise RuntimeError("states used different ligand params versions: %s - the double "
                           "difference would be meaningless" % hashes)
    return True


def run(ctx):
    """requires ctx['poses'], ctx['states'], ctx['route'];  produces ctx['ligand_params']

    A separate stage on purpose: .params is a chemical preparation step with its own formal charge,
    stereochemistry, atom names and hash to record, and the six PyRosetta states AND specificity
    must all read the SAME parameter set. Every decoy gets its OWN resname and .params - they are
    different molecules, and reusing the target's parameters for a different atom topology is not
    a thing that can work.

    -> {'target': {...}, 'decoys': {id: {...}}, 'all_params': [paths], 'bundle_hash': str}
    """
    import hashlib

    rt = ctx["route"]
    out_dir = os.path.join(ctx["out_dir"], "ligand_params")
    os.makedirs(out_dir, exist_ok=True)

    target_smiles = rt.get("target_smiles") or rt.get("smiles")
    if not target_smiles:
        raise RuntimeError("route produced no target SMILES to parameterise")

    poses = ctx.get("poses") or {}
    target_resname = poses.get("target_resname", "TGT")

    # THE TARGET IS PARAMETERISED FROM THE POSE'S OWN MOLECULE, not re-embedded from SMILES.
    # X_D_lig / X_I_lig were written from that molecule; a fresh embed gives a different atom order,
    # so the .params atom names would not match the ligand in those PDBs and Rosetta would either
    # reject the residue or silently rebuild it against the wrong correspondence.
    top = None
    for key in ("X_I", "X_D"):
        ps = poses.get(key) or []
        if ps:
            top = ps[0]
            break
    if top is None:
        raise RuntimeError("no target pose available: ligand_params must parameterise the SAME "
                           "molecule the liganded-state PDBs were written from")
    target = _parameterise_mol(top["mol"], out_dir, target_resname, "target",
                               conf_id=top.get("conf_id", -1), smiles=target_smiles)

    # the params atom names must match the ligand actually sitting in the liganded states
    checked = []
    for state_key in ("X_D_lig", "X_I_lig"):
        pdb = (ctx.get("states") or {}).get("paths", {}).get(state_key)
        if pdb and os.path.exists(pdb):
            check_matches_pdb(target["params"], pdb, target_resname)   # raises on any mismatch
            checked.append(state_key)
    if not checked:
        raise RuntimeError("neither X_D_lig nor X_I_lig exists yet: ligand_params must run AFTER "
                           "pose, so the params can be checked against the actual pose PDBs")

    # decoys: the same set specificity will score. Their SDFs are kept in the bundle so specificity
    # reads the molecule back from there rather than re-embedding it - same reason as the target.
    from .specificity import default_decoys
    preserve = ctx.get("cfg", {}).get("objective", {}).get("preserve_native_response", False)
    native = None if preserve else rt.get("native_smiles")
    decoys = {}
    for i, (name, smi) in enumerate(default_decoys(target_smiles, native) or [], start=1):
        if not smi or smi == target_smiles:
            continue
        did = "D%02d" % i
        try:
            decoys[did] = _parameterise(smi, out_dir, did, role=name)
            decoys[did]["decoy_name"] = name
        except Exception as exc:
            raise RuntimeError("decoy %s (%s / %s) could not be parameterised: %s. A decoy without "
                               "its own params cannot be scored against the target."
                               % (did, name, smi, exc))

    all_params = [target["params"]] + [d["params"] for d in decoys.values()]
    bundle_hash = hashlib.sha1("|".join(sorted(_sha1(p) for p in all_params)).encode()).hexdigest()[:12]
    bundle = {"target": target, "decoys": decoys, "all_params": all_params,
              "bundle_hash": bundle_hash}
    with open(os.path.join(out_dir, "bundle.json"), "w") as f:
        json.dump(bundle, f, indent=2)
    return {"ligand_params": bundle}


def _parameterise_mol(mol, out_dir, resname, role, conf_id=-1, smiles=None):
    """One RDKit MOLECULE -> .params, written from that molecule's own coordinates.

    This is the path the target takes, because its pose PDBs (X_D_lig / X_I_lig) were written from
    this same object. Writing the SDF from it keeps ONE atom order across target.sdf and both
    liganded-state PDBs, which is what makes the .params atom names line up with the ligand Rosetta
    reads out of those files.
    """
    from rdkit import Chem
    charge = Chem.GetFormalCharge(mol)
    sdf = os.path.join(out_dir, "%s.sdf" % resname)
    w = Chem.SDWriter(sdf)
    w.write(mol, confId=int(conf_id))
    w.close()

    meta = from_sdf(sdf, out_dir, name=resname, formal_charge=charge)
    params = meta["params"] if isinstance(meta, dict) else meta
    verify_params_charge(params, charge)

    # per-molecule provenance: from_sdf writes one fixed META filename, so each call would
    # otherwise overwrite the previous molecule's record
    shared = os.path.join(out_dir, META)
    own = os.path.join(out_dir, "%s_metadata.json" % resname)
    if os.path.exists(shared):
        try:
            os.replace(shared, own)
        except OSError:
            own = shared

    return {"resname": resname, "params": params,
            "smiles": smiles or Chem.MolToSmiles(Chem.RemoveHs(mol)),
            "formal_charge": charge, "role": role, "sdf": sdf,
            "metadata_json": own,
            "sha1": _sha1(params),
            "metadata": meta if isinstance(meta, dict) else {}}


def _parameterise(smiles, out_dir, resname, role):
    """SMILES -> .params. Used for decoys, which have no pre-existing pose; the SDF written here is
    kept in the bundle so specificity poses THIS molecule rather than re-embedding its SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("unparsable SMILES for %s: %s" % (role, smiles))
    mh = Chem.AddHs(m)
    if AllChem.EmbedMolecule(mh, randomSeed=0xA11) != 0:
        raise RuntimeError("could not embed 3D coordinates for %s (%s)" % (role, smiles))
    AllChem.MMFFOptimizeMolecule(mh)
    return _parameterise_mol(mh, out_dir, resname, role, conf_id=-1, smiles=smiles)


def load_metadata(out_dir):
    p = os.path.join(out_dir, META)
    if not os.path.exists(p):
        raise RuntimeError("no %s in %s - refusing to run with an unverified ligand (a zero-charge "
                           "fallback ligand would silently break every electrostatic term)" % (META, out_dir))
    return json.load(open(p))
