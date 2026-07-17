"""PyRosetta ref2015 - the atom-level ENERGY CALCULATOR. Not a design method.

    ours      aa_filter, rotamers, sidechain, search, six states, allosteric template,
              linkage, DNA release, target/decoy negative design
    Rosetta   mutate + repack, restrained minimisation, protein-ligand / protein-DNA energy,
              and the reference energy that stops the search collapsing to poly-Gly

HARD DEPENDENCY. There is deliberately no automatic fallback: our scoring.py has no reference
energy and its global optimum on TtgR/quercetin was literally GGGGGGGGGGG. A silent fallback would
put poly-Gly back into the candidate pool. It can only be enabled explicitly, and never in
production:  mode: debug_geometry_only

ref in two different roles - do not conflate them
-------------------------------------------------
  sequence search (DIFFERENT sequences compared):  ref DOES matter. Composition changes, so ref
      changes, and it is exactly what makes poly-Gly/poly-Ala lose. Keep it.
  state linkage (ONE sequence in four states):     ref CANCELS. Composition is identical, so
      E_ref(I_L) = E_ref(D_L) = E_ref(I_0) = E_ref(D_0) and it drops out of
          ddG_coup = (E_IL - E_DL) - (E_I0 - E_D0)
      exactly, twice. The allosteric conclusion therefore never rests on ref, on Rosetta's
      absolute scale, or on any fitted total.

SIGN CONVENTION - fixed here, referenced everywhere, never redefined per module:
    ddG_coup < 0   the ligand shifts the population D -> I           (a sensor needs this)
    dG_apo   > 0   with no ligand the protein still prefers D        (not constitutive)
    S_release > 0  the induced state releases the operator           (inducible repressor)
                   corepressors invert S_release via topology sign, NOT by editing this file.
"""
import hashlib
import os

SCOREFXN = "ref2015"
# terms kept even though ref cancels in linkage: a state-preparation mistake shows up as a
# nonsensical split (e.g. fa_rep exploding in one state only) long before it shows up in the total
TERMS = ["total_score", "fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec",
         "hbond_sc", "hbond_bb_sc", "fa_dun", "ref"]

# ddG_coup < 0 wanted; dG_apo > 0 wanted; S_release > 0 wanted (inducible repressor)
SIGN = dict(ddG_coup=-1, dG_apo=+1, S_release=+1)


def available():
    try:
        import pyrosetta  # noqa: F401
        return True
    except Exception:
        return False


class PyRosettaBackend:
    """Every state in one candidate MUST go through one instance of this class, so that
    scorefunction, protocol, protonation and ligand params are identical by construction.
    A double difference between states prepared differently is meaningless."""

    def __init__(self, score_function=SCOREFXN, ligand_params=None, extra_flags=""):
        if not available():
            raise RuntimeError(
                "PyRosetta is required for production multistate scoring. No automatic fallback "
                "is allowed (our scoring.py has no reference energy; its optimum is poly-Gly).\n"
                "  licence: https://els2.comotion.uw.edu/product/pyrosetta\n"
                "  install: pip install pyrosetta-installer && python -c "
                "'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'")
        import pyrosetta
        self.params = list(ligand_params or [])
        flags = ("-mute all -ex1 -ex2aro -use_input_sc -ignore_unrecognized_res false "
                 "-load_PDB_components false ")
        if self.params:
            flags += "-extra_res_fa " + " ".join(self.params) + " "
        pyrosetta.init(flags + extra_flags)
        from pyrosetta import create_score_function
        self.sfxn_name = score_function
        self.sfxn = create_score_function(score_function)
        self.params_hash = self._hash(self.params)

    @staticmethod
    def _hash(paths):
        h = hashlib.sha1()
        for p in sorted(paths or []):
            h.update(open(p, "rb").read())
        return h.hexdigest()[:12] if paths else "no-ligand"

    # ---- pose -------------------------------------------------------------------------------
    def prepare_pose(self, pdb_path, ligand_params=None):
        if ligand_params and self._hash(ligand_params) != self.params_hash:
            raise RuntimeError("ligand params differ from the ones this backend was initialised "
                               "with - all six states must share one params version")
        from pyrosetta.rosetta.core.pose import Pose
        from pyrosetta.rosetta.core.import_pose import pose_from_file
        pose = Pose()
        pose_from_file(pose, pdb_path)
        return pose

    def _pose_index(self, pose, resnum, chain="A"):
        i = pose.pdb_info().pdb2pose(chain, int(resnum))
        if i == 0:
            raise ValueError("residue %s%s absent from pose" % (chain, resnum))
        return i

    # ---- mutate + repack --------------------------------------------------------------------
    def mutate_and_repack(self, pose, sequence, design_positions, repack_positions=(), chain="A",
                          symmetric_chains=None):
        """sequence: {pdb_resnum: 'ALA'} for design_positions.
        Repacking is restricted to design + repack positions; everything else is frozen, so the
        six states differ only where we intend them to.

        symmetric_chains: for a HOMODIMER the pocket exists in every subunit and every subunit must
        carry the mutation. Mutating only chain A leaves one mutant subunit packed against one WT
        subunit - a chimera that exists nowhere and whose interface energy is meaningless (GPT-5.6).
        Pass e.g. ['A','B'] to apply the same design to both. Defaults to [chain] (monomer/asym).
        """
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            InitializeFromCommandline, RestrictToRepacking, PreventRepacking,
            OperateOnResidueSubset)
        from pyrosetta.rosetta.core.select.residue_selector import (
            ResidueIndexSelector, NotResidueSelector)

        chains = list(symmetric_chains) if symmetric_chains else [chain]
        p = pose.clone()
        movable = set()
        for ch in chains:
            for rn in design_positions:
                aa = sequence[rn]
                idx = self._pose_index(p, rn, ch)
                MutateResidue(idx, aa.upper()).apply(p)
                movable.add(idx)
            for r in repack_positions:
                movable.add(self._pose_index(p, r, ch))
        movable = sorted(movable)
        sel = ResidueIndexSelector(",".join(str(i) for i in movable))
        tf = TaskFactory()
        tf.push_back(InitializeFromCommandline())
        tf.push_back(RestrictToRepacking())
        tf.push_back(OperateOnResidueSubset(PreventRepacking(), NotResidueSelector(sel)))
        PackRotamersMover(self.sfxn, tf.create_task_and_apply_taskoperations(p)).apply(p)
        return p

    # ---- restrained minimisation -------------------------------------------------------------
    def restrained_minimize(self, pose, design_positions, second_shell_positions=(),
                            allow_ligand_torsions=True, ligand_rigid_body=False,
                            freeze_dna=True, chain="A"):
        """Backbone frozen, jumps frozen, chi free only where we say.

        Two traps this guards against:
          * free backbone -> minimisation relaxes the imposed X_D and X_I toward each other and
            the state difference we are measuring evaporates;
          * free ligand rigid body -> the ligand re-docks itself in each state, so the double
            difference silently reports a docking result instead of an allosteric one.
        Ligand INTERNAL torsions stay free: a rigid ligand is unphysical strain.
        """
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover

        mm = MoveMap()
        mm.set_bb(False)
        mm.set_chi(False)
        mm.set_jump(False)                       # ligand + DNA rigid bodies stay put
        for rn in list(design_positions) + list(second_shell_positions):
            mm.set_chi(self._pose_index(pose, rn, chain), True)
        if allow_ligand_torsions:
            for i in range(1, pose.total_residue() + 1):
                if pose.residue(i).is_ligand():
                    mm.set_chi(i, True)          # internal torsions only; jump stays frozen
        if ligand_rigid_body:
            raise NotImplementedError("V1 keeps the ligand rigid body frozen on purpose - "
                                      "letting it float turns the state difference into docking")
        p = pose.clone()
        MinMover(mm, self.sfxn, "lbfgs_armijo_nonmonotone", 0.01, True).apply(p)
        return p

    # ---- energies ----------------------------------------------------------------------------
    def score_terms(self, pose):
        """-> dict of decomposed terms. Kept even though ref cancels in linkage: an inconsistent
        state preparation shows up here (one state with exploding fa_rep) long before the total
        looks wrong."""
        from pyrosetta.rosetta.core.scoring import ScoreType
        self.sfxn(pose)
        e = pose.energies().total_energies()
        out = {"total_score": float(self.sfxn(pose))}
        for t in TERMS[1:]:
            try:
                out[t] = float(e[getattr(ScoreType, t)])
            except Exception:
                out[t] = None
        out["_sfxn"] = self.sfxn_name
        out["_params_hash"] = self.params_hash
        return out

    def pose_composition(self, pose):
        """What is actually in this pose: ligand residue count and DNA residue count.

        Used to VERIFY a state matches its label - nothing else checks that D0 really has no
        ligand or that a DNA state really has DNA. A mislabelled template silently poisons the
        double difference (GPT-5.6: state labels are never validated).
        """
        n_lig = n_dna = 0
        for i in range(1, pose.total_residue() + 1):
            r = pose.residue(i)
            if r.is_ligand():
                n_lig += 1
            if r.is_DNA():
                n_dna += 1
        return {"n_ligand": n_lig, "n_dna": n_dna}

    def ligand_jump(self, pose):
        """The jump whose downstream side is the ligand. NOT assumed to be jump 1.

        In a dimer with ligand and/or DNA the fold tree has several jumps and jump 1 is usually a
        protein-protein or protein-DNA partition. Separating across the wrong jump measures the
        wrong interface. Found by GPT-5.6. We locate the jump that actually moves the ligand by
        checking which jump's downstream partition contains a ligand residue.
        """
        from pyrosetta.rosetta.core.kinematics import FoldTree
        ft = pose.fold_tree()
        lig = [i for i in range(1, pose.total_residue() + 1) if pose.residue(i).is_ligand()]
        if not lig:
            raise ValueError("no ligand residue in pose: interface_energy has nothing to separate")
        for j in range(1, ft.num_jump() + 1):
            stop = ft.downstream_jump_residue(j)
            # a jump directly onto a ligand residue, or whose subtree holds one
            if stop in lig:
                return j
        # fall back: the jump whose downstream residue is closest (in fold-tree order) to a ligand
        for j in range(1, ft.num_jump() + 1):
            if any(abs(ft.downstream_jump_residue(j) - L) == 0 for L in lig):
                return j
        raise ValueError("could not identify the ligand jump in the fold tree; refusing to guess "
                         "jump=1, which would measure the wrong interface")

    def interface_energy(self, pose, jump=None):
        """E(complex) - E(separated) across the LIGAND jump (auto-detected unless given).

        jump defaults to None -> ligand_jump(pose). Passing an explicit jump is allowed for the DNA
        interface, but the default no longer silently assumes jump 1 is the ligand.
        """
        from pyrosetta.rosetta.protocols.rigid import RigidBodyTransMover
        if jump is None:
            jump = self.ligand_jump(pose)
        complexed = float(self.sfxn(pose))
        p = pose.clone()
        t = RigidBodyTransMover(p, jump)
        t.step_size(500.0)
        t.apply(p)
        return complexed - float(self.sfxn(p))

    def residue_pair_terms(self, pose, residues_a, residues_b, chain="A"):
        """Per-pair decomposition - used to see WHICH contacts change between states."""
        from pyrosetta.rosetta.core.scoring import ScoreType
        self.sfxn(pose)
        emap = pose.energies()
        out = {}
        for ra in residues_a:
            ia = self._pose_index(pose, ra, chain)
            for rb in residues_b:
                ib = self._pose_index(pose, rb, chain)
                try:
                    out[(ra, rb)] = float(emap.residue_pair_energies(ia, ib, self.sfxn).sum())
                except Exception:
                    out[(ra, rb)] = None
        return out


def linkage(energies):
    """energies: {'I_L','D_L','I_0','D_0'} totals for ONE sequence -> ddG_coup (want < 0).
    ref and every composition-dependent bias cancel exactly in this double difference."""
    need = ("I_L", "D_L", "I_0", "D_0")
    if any(k not in energies or energies[k] is None for k in need):
        return None                                   # fail closed
    return (energies["I_L"] - energies["D_L"]) - (energies["I_0"] - energies["D_0"])


def apo_bias(energies):
    """dG_apo = E(I_0) - E(D_0), want > 0: with no ligand the protein must still prefer D."""
    if any(k not in energies or energies[k] is None for k in ("I_0", "D_0")):
        return None
    return energies["I_0"] - energies["D_0"]


def dna_release(e_dna_induced, e_dna_compatible, topology_sign=+1):
    """S_release, want > 0 for an inducible repressor; corepressors pass topology_sign=-1."""
    if e_dna_induced is None or e_dna_compatible is None:
        return None
    return topology_sign * (e_dna_induced - e_dna_compatible)
