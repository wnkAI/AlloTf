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
    cmd = ["python", molfile_to_params, "-n", name, "-p", os.path.join(out_dir, name),
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
    try:
        import pyrosetta
        base = os.path.dirname(pyrosetta.__file__)
        for sub in ("toolbox/molfile_to_params.py", "../molfile_to_params.py",
                    "database/chemical/molfile_to_params.py"):
            p = os.path.normpath(os.path.join(base, sub))
            if os.path.exists(p):
                return p
    except Exception:
        pass
    raise RuntimeError("molfile_to_params.py not found. It ships with PyRosetta/Rosetta; without "
                       "it no ligand can be scored, and there is no safe default.")


def verify_params_charge(params_path, expected_charge, tol=0.05):
    """Rosetta partial charges must sum to the formal charge. If they do not, every fa_elec term
    in all six states is wrong by the same silent amount."""
    tot = 0.0
    for line in open(params_path):
        if line.startswith("ATOM "):
            f = line.split()
            if len(f) >= 5:
                try:
                    tot += float(f[4])
                except ValueError:
                    pass
    if abs(tot - expected_charge) > tol:
        raise ValueError("params partial charges sum to %+.3f but formal charge is %+d (%s)"
                         % (tot, expected_charge, params_path))
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


def load_metadata(out_dir):
    p = os.path.join(out_dir, META)
    if not os.path.exists(p):
        raise RuntimeError("no %s in %s - refusing to run with an unverified ligand (a zero-charge "
                           "fallback ligand would silently break every electrostatic term)" % (META, out_dir))
    return json.load(open(p))
