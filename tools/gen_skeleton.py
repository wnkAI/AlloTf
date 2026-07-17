import os
F = {}

F["config/default.yaml"] = """# AlloTF V1 default config
target:
  name: null            # e.g. xylitol
  smiles: null          # resolved from name if absent

objective:
  mode: auto            # auto | enhancement | retargeting
  preserve_native_response: false
  operator_sequence: null

route:
  top_scaffolds: 3
  min_structure_tier: 2   # 1 = experimental apo/holo/DNA, 2 = DNA state homology-modelled

design:
  raw_designs: 10000
  max_mutations: 6
  design_both_backbones: true
  ligandmpnn:
    checkpoint: null
    temperature: 0.2
    batch: 32

state_builder:
  repack_shell: 8.0
  minimize: restrained    # sidechain repack + restrained local min, NO MD
  backbone_restraint: 0.5

output:
  final_designs: 96
  project_dir: outputs
"""

F["config/scoring.yaml"] = """# weights + hard gates. V1 reports RELATIVE proxies, never absolute Kd.
weights:
  target_binding:      0.25
  state_preference:    0.25   # S_state = E_L(DNA-compatible) - E_L(induced) > 0
  dna_release:         0.20   # S_release = E_DNA(induced) - E_DNA(DNA-compatible) > 0
  allosteric_template: 0.15
  specificity:         0.10
  fold_stability:      0.05

hard_gates:
  max_fold_clash:            0
  min_state_preference:      0.0
  max_apo_dna_energy:        0.0
  min_dna_release:           0.0
  min_allosteric_similarity: 0.5

diversity:
  cluster_identity: 0.9
  max_per_cluster: 3
"""

F["config/forcefield.yaml"] = """# empirical terms for utils/energy.py - relative proxies only
terms:
  hbond:         {weight: 1.0, max_dist: 3.5, max_angle: 60}
  salt_bridge:   {weight: 1.2, max_dist: 4.0}
  vdw:           {weight: 1.0, clash_dist: 2.2, contact_dist: 4.5}
  electrostatic: {weight: 0.8, dielectric: 4.0}
  solvation:     {weight: 0.5}
  buried_unsat_polar: {penalty: 2.0}
backend: internal    # internal | rosetta | foldx
"""

F["utils/standardize_ligand.py"] = '''"""Ligand standardisation: salts, protonation, tautomers, stereochemistry, sugar ring/open forms.

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
'''

F["utils/residue_mapping.py"] = '''"""Residue numbering map across PDB entries/constructs of the same TF.

Different entries of one TF use different author numbering. Every cross-structure comparison
(torsion, contacts, masks) must go through here.
"""
from Bio import pairwise2
from Bio.PDB.Polypeptide import three_to_index, index_to_one


def chain_sequence(chain):
    seq, nums = "", []
    for r in chain:
        if r.id[0] != " ":
            continue
        try:
            seq += index_to_one(three_to_index(r.get_resname()))
        except Exception:
            continue
        nums.append(r.id[1])
    return seq, nums


def map_chains(ref_chain, mob_chain):
    """-> {mobile_resnum: ref_resnum} by sequence alignment."""
    sr, nr = chain_sequence(ref_chain)
    sm, nm = chain_sequence(mob_chain)
    aln = pairwise2.align.globalms(sr, sm, 2, -1, -10, -0.5, one_alignment_only=True)
    if not aln:
        return {}
    a, b = aln[0].seqA, aln[0].seqB
    out, i, j = {}, 0, 0
    for x, y in zip(a, b):
        if x != "-" and y != "-":
            out[nm[j]] = nr[i]
        if x != "-":
            i += 1
        if y != "-":
            j += 1
    return out
'''

F["utils/contacts.py"] = '''"""Contact / hbond / salt-bridge networks and their state-to-state changes."""
import numpy as np

CONTACT = 4.5
CLASH = 2.2
HB = 3.5
POS = {"ARG", "LYS", "HIS"}
NEG = {"ASP", "GLU"}


def heavy(res):
    return [a for a in res if a.element != "H"]


def contact_map(chain, resnums=None):
    res = [r for r in chain if r.id[0] == " " and (resnums is None or r.id[1] in resnums)]
    idx = [r.id[1] for r in res]
    A = [np.array([a.coord for a in heavy(r)]) for r in res]
    n = len(res)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 2, n):
            if len(A[i]) and len(A[j]):
                d = np.linalg.norm(A[i][:, None, :] - A[j][None, :, :], axis=2)
                C[i, j] = C[j, i] = (d < CONTACT).sum()
    return idx, C


def contact_delta(idx_a, Ca, idx_b, Cb):
    """dC_ij between two states over shared residues."""
    common = [r for r in idx_a if r in set(idx_b)]
    ia = [idx_a.index(r) for r in common]
    ib = [idx_b.index(r) for r in common]
    return common, Cb[np.ix_(ib, ib)] - Ca[np.ix_(ia, ia)]


def interface(atoms_a, atoms_b):
    """-> (contacts, clashes, min_dist)"""
    if not atoms_a or not atoms_b:
        return 0, 0, float("nan")
    A = np.array([a.coord for a in atoms_a])
    B = np.array([a.coord for a in atoms_b])
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return int((d < CONTACT).sum()), int((d < CLASH).sum()), float(d.min())


def hbonds(atoms_a, atoms_b):
    """TODO(D): angle-aware detection. V1 = donor/acceptor distance only."""
    da = [a for a in atoms_a if a.element in ("N", "O")]
    db = [a for a in atoms_b if a.element in ("N", "O")]
    if not da or not db:
        return 0
    A = np.array([a.coord for a in da])
    B = np.array([a.coord for a in db])
    return int((np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) < HB).sum())
'''

F["utils/energy.py"] = '''"""Empirical interaction terms. RELATIVE proxies on one scaffold - never report as Kd."""
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
'''

F["pipeline/structure.py"] = '''"""Prepare comparable native states for one scaffold, and QC them.

V1 needs at least:  X_D (DNA-bound / DNA-compatible)  and  X_I (effector-induced).
Optional: X_A (apo).

QC is mandatory. "has a DNA structure" != "has a usable allosteric state":
  * is the bound ligand a real effector or a crystallisation additive?
  * is the DNA a real operator?
  * is the oligomeric state right (TetR/LacI act as dimers)?
  * can residue numbering be mapped across entries?
  * are hinge / DBD regions resolved?
  * is the construct truncated or carrying non-native mutations?
"""

def prepare(tf_name, out_dir):
    """-> dict of state paths + residue_mapping.json + operator.fasta

    TODO(B):
      1. read data/atf_structure_db.csv (uniprot, tier, pdb ids)
      2. download states, pick best entry per state (resolution, completeness, oligomer)
      3. run qc(); REJECT the scaffold on failure - never silently continue
      4. build residue_mapping.json via utils.residue_mapping
    """
    raise NotImplementedError


def qc(state_paths, native_ligand_smiles):
    """-> (ok: bool, report: dict)"""
    raise NotImplementedError
'''

F["pipeline/allostery.py"] = '''"""Extract the SYSTEM-SPECIFIC native allosteric template. No training, no MD.

Compares the scaffold's own experimental states X_D -> X_I and emits:
  * per-residue torsion redistribution   (utils/torsion.py, circular statistics)
  * contact / hbond / salt-bridge change (utils/contacts.py)
  * DBD output geometry change
  * allosteric path: pocket -> second shell -> hinge/dimer -> DBD (graph search)

and the three masks that constrain Design:
  recognition_mask   ligand first shell, free to redesign
  transduction_mask  second shell / hinge, limited compensatory mutations
  protected_mask     DNA recognition helix, dimer core, known allosteric nodes, fold core

Verified on TetR: torsion redistribution alone recovers alpha6 (103-109) and alpha4 (49, 62),
but also flags terminal residues (5, 205) that merely wobble. Terminal/surface residues that are
NOT on the pocket->DBD path must be discarded as noise.
"""

def native_template(state_paths, ligand_resname, out_json):
    """-> native_allosteric_template.json

    TODO(B):
      1. torsion fingerprint X_D vs X_I (utils.torsion - implemented)
      2. contact delta (utils.contacts.contact_delta)
      3. DBD geometry delta
      4. path search pocket -> DBD over the contact graph
      5. classify residues: recognition / transduction / output / noise
    """
    raise NotImplementedError


def build_masks(template_json, out_json):
    """-> masks.json. DBD is always protected in V1: it is the readout, not a design target."""
    raise NotImplementedError
'''

F["pipeline/pose.py"] = '''"""Target ligand pose generation in the scaffold pocket.

Poses must be produced on BOTH backbones - the whole method rests on asking whether the ligand
prefers the induced state:
    pose on X_D (DNA-compatible backbone)
    pose on X_I (induced backbone)
"""

def generate_poses(ligand_smiles, state_path, pocket_residues, n_poses=10):
    """-> list of poses (coords + score)

    TODO(C): RDKit ETKDG conformers -> dock (smina/vina) or transfer from the native holo ligand;
    keep multiple poses, never just top-1.
    """
    raise NotImplementedError
'''

F["pipeline/design.py"] = '''"""LigandMPNN candidate generation. It is a POCKET PROPOSAL GENERATOR, nothing more.

It answers P(S | X, L_target): which residues can hold this ligand here.
It does NOT know whether binding switches the DBD - that is decided downstream by
state_builder + ligand_score + dna_release + allosteric template matching.

Upgrade over stock usage (no retraining - the data to retrain does not exist):
  * sample on the induced backbone, then SCORE the same sequence on the DNA-compatible backbone
  * masks: protected fixed, transduction limited, recognition free
"""

def generate(state_paths, poses, masks, cfg):
    """-> raw_candidates.fasta

    TODO(C):
      1. call LigandMPNN with fixed/bias positions from masks
      2. generate on both backbones (cfg.design.design_both_backbones)
      3. dedupe + cluster, cap mutations at cfg.design.max_mutations
    """
    raise NotImplementedError
'''

F["pipeline/state_builder.py"] = '''"""Build the four static functional states per candidate. This is the core of the method.

  C_D  = S + DNA                 apo state must still bind the operator
  C_LD = S + L_target on X_D     does the target wrongly prefer the DNA-compatible state?
  C_LI = S + L_target on X_I     does the target stabilise the induced state?
  C_ID = S_induced + DNA         is the induced conformation already bad for the operator?

Sidechain repacking + restrained local minimisation only. NO MD.
"""

def build(candidate_seq, state_paths, poses, cfg):
    """-> {'C_D':path, 'C_LD':path, 'C_LI':path, 'C_ID':path}

    TODO(D): thread sequence onto each backbone, repack within cfg.state_builder.repack_shell,
    restrained minimisation with backbone restraints.
    """
    raise NotImplementedError
'''

F["pipeline/ligand_score.py"] = '''"""Does the target ligand PREFER the induced state?

    S_state = E_L(on X_D) - E_L(on X_I)     want > 0

Plus interpretable terms: hbonds, hydrophobic contacts, salt bridges, shape complementarity,
buried unsatisfied polars, ligand clash, buried surface area.
"""

def score(states, cfg):
    """-> dict(state_preference, target_binding, terms)
    TODO(D): utils.energy.interface_energy on ligand vs pocket in C_LD and C_LI."""
    raise NotImplementedError
'''

F["pipeline/specificity.py"] = '''"""Negative design against decoys.

Retargeting must include at least: the native ligand, close analogues of the target,
stereoisomers, and abundant host metabolites.

    S_specificity = E(best decoy) - E(target)

Whether the native response must be abolished is a user parameter
(objective.preserve_native_response).
"""

def score(candidate, states, decoys, cfg):
    """-> dict(specificity, per_decoy)
    TODO(D): dock each decoy into the designed pocket on X_I, compare with target."""
    raise NotImplementedError


def default_decoys(target_smiles, native_smiles):
    """TODO(A): analogues + stereoisomers + host metabolites."""
    return [native_smiles]
'''

F["pipeline/rank.py"] = '''"""Hard gates first, THEN multi-objective Pareto ranking, THEN diversity selection.

Never collapse everything into one weighted sum up front: a candidate that fails any gate is not
a sensor no matter how well it binds.
"""

def apply_gates(features, gates):
    return (features.get("fold_clash", 0) <= gates["max_fold_clash"]
            and features.get("state_preference", -1) > gates["min_state_preference"]
            and features.get("apo_dna_energy", 1) < gates["max_apo_dna_energy"]
            and features.get("dna_release", -1) > gates["min_dna_release"]
            and features.get("allosteric_similarity", 0) > gates["min_allosteric_similarity"])


def pareto_front(cands, objectives):
    """-> non-dominated subset. TODO(E)."""
    raise NotImplementedError


def rank(candidates, cfg):
    """-> ranked_candidates.csv + final_N.fasta, diversity-capped per sequence cluster.
    TODO(E): gates -> pareto -> cluster -> pick cfg.output.final_designs."""
    raise NotImplementedError
'''

F["pipeline/active_learning.py"] = '''"""Optional: fold wet-lab results back in. Target-specific, not a general model.

After round 1 you have (S_i, L_target) -> y_i for a few dozen designs. Those points are worth more
than the whole Sensor-seq corpus for THIS target, because they are on-target.
"""

def update(round_results, cfg):
    """TODO(E): fit f_target(S) on measured designs; propose round 2 by
    uncertainty + Pareto + diversity."""
    raise NotImplementedError
'''

F["tests/test_route.py"] = '''"""Test 1 - native ligand recovery: a native effector must return its own TF."""
import sys
sys.path.insert(0, r"E:\\DATA\\AlloTf")
from pipeline.route import route


def test_native_ligand_recovers_scaffold():
    r = route("naringenin", top_k=5)
    tfs = [h["tf"] for h in r["hits"]]
    assert "TtgR" in tfs, tfs
    assert r["mode"] == "ENHANCEMENT", r["mode"]


def test_chem_and_structure_reported_separately():
    r = route("quinine", top_k=5)
    h = r["hits"][0]
    assert "s_chem" in h and "s_struct" in h and "tier" in h
'''

F["tests/test_structure.py"] = '''"""QC must REJECT, not silently pass, when a state is unusable."""
import pytest


@pytest.mark.skip(reason="TODO(B): implement pipeline.structure.prepare")
def test_qc_rejects_additive_as_effector():
    ...


@pytest.mark.skip(reason="TODO(B)")
def test_qc_rejects_wrong_oligomer():
    ...
'''

F["tests/test_allostery.py"] = '''"""Test 2 - native switch recovery on a scaffold with a known mechanism."""
import pytest


@pytest.mark.skip(reason="TODO(B): wire pipeline.allostery.native_template")
def test_tetr_transduction_path_recovered():
    """TetR: torsion redistribution must flag alpha6 (~103-109) and alpha4 (~49-62), and must NOT
    keep terminal residues (5, 205) after the network filter."""
'''

F["tests/test_dna_release.py"] = '''"""Test 2b - WT direction: the native effector must weaken the predicted DNA interface.
A scaffold that cannot reproduce the WT direction must not enter design."""
import pytest


@pytest.mark.skip(reason="TODO(D): wire pipeline.dna_release on WT")
def test_wt_effector_weakens_dna_interface():
    ...
'''

F["tests/test_end_to_end.py"] = '''"""Test 3 - known functional vs constitutive mutants must be separated by the hard gates."""
import pytest


@pytest.mark.skip(reason="TODO(E): needs the full pipeline")
def test_known_dead_mutants_are_gated_out():
    ...
'''

for p, c in F.items():
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(c)
    print("wrote", p)

for d in ("pipeline", "utils", "tests"):
    open(os.path.join(d, "__init__.py"), "w").close()
print("\ntotal:", len(F) + 3, "files")
