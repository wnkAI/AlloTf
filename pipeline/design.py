"""Candidate generation. The production designer is OURS; LigandMPNN is a paper baseline.

Why we are not building on LigandMPNN
------------------------------------
It optimises   P(S | X_backbone, X_ligand)   under plain native-amino-acid cross-entropy:
    L = -sum_i log P(a_i_native | X, L)
i.e. "which residue does the PDB usually put in this geometry?" - not "which sequence binds this
target and switches the output".
  * ITS OWN ABLATION: removing ligand element type barely moves near-ligand sequence recovery.
    If erasing chemical identity does not move the metric, the metric is not measuring chemistry.
  * No negative data at all. The PDB holds successful complexes only, so it never learned why a
    sequence should NOT bind, nor how to reject a decoy or the wrong stereoisomer.
  * Fixed backbone + fixed pose: a wrong pose still yields a confident sequence FOR THE WRONG POSE.
  * Side-chain accuracy decays outward (chi1 84%, chi2 64%, chi3 28%, chi4 19%) - exactly the
    terminal atoms that make salt bridges and directional H-bonds.
  * No notion of apo-vs-holo, DNA-bound-vs-free, or switching.

So it is a fixed-backbone sequence prior, and it is kept ONLY as a benchmark:
  1. does our physics-grounded designer beat it? (paper comparison)
  2. does our designer emit sequences so unnatural that a native-sequence prior rejects them?
It never ranks, filters or selects a production candidate.

Wording discipline: an empirical force field is PHYSICS-GROUNDED, not first-principles in the
quantum sense. Do not write 'ab initio'.
"""
from abc import ABC, abstractmethod


class DesignBackend(ABC):
    name = "abstract"
    production = False          # may this backend decide real candidates?

    @abstractmethod
    def propose(self, ctx, n):
        """-> list[dict(seq, mutations, rotamers, backbone, energy_terms, meta)]"""

    @abstractmethod
    def score(self, seq, state_path, pose, cfg):
        """-> float, comparable within one backend only."""

    def supports_negative_design(self):
        return False


class PhysAlloDesignBackend(DesignBackend):
    """THE production designer: physics-grounded, multistate, allostery-aware.

    Pipeline inside propose():
        target pose ensemble
          -> per-position allowed-AA pre-filter   (pocket volume, polarity, charge, burial,
                                                   conservation, backbone phi/psi, ligand groups)
          -> rotamer search (chi1..chi4) scored by explicit terms
                 E_local = E_vdW + E_Coulomb + E_HB(directional) + E_solv + E_strain + E_unsat
          -> joint sequence+rotamer search (SA / MC / beam / GA)
          -> thread each candidate into D0 / I0 / DL / IL
          -> state preference + linkage:
                 ddG_coup = (E_IL - E_DL) - (E_I0 - E_D0)      want < 0
          -> DNA interface:  S_release = E_DNA(I) - E_DNA(D)   sign from topology
          -> native torsion/contact template check (distal only; the pocket is meant to differ)
          -> native-ligand + decoy negative design
        Hard gates BEFORE any weighted sum. Weights only break ties inside the Pareto front.

    Unlike a sequence prior, this backend can do negative design: a decoy is simply scored with
    the same terms.
    """
    name = "physallo"
    production = True

    def propose(self, ctx, n):
        raise NotImplementedError("TODO(C/D): see pipeline/physallo/ - AA prefilter, rotamer "
                                  "search, multistate linkage")

    def score(self, seq, state_path, pose, cfg):
        raise NotImplementedError

    def supports_negative_design(self):
        return True


class PhysicsDesignBackend(DesignBackend):
    """Ablation of PhysAllo: same physics, NO allosteric template and NO multistate linkage.
    Its job in the paper is to show how much of the gain comes from allostery rather than from
    plain binding physics."""
    name = "physics"
    production = True

    def propose(self, ctx, n):
        raise NotImplementedError("TODO(C/D): single-state physics ablation")

    def score(self, seq, state_path, pose, cfg):
        raise NotImplementedError

    def supports_negative_design(self):
        return True


class LigandMPNNBenchmarkBackend(DesignBackend):
    """BENCHMARK ONLY. Frozen checkpoint, frozen params, frozen inputs. Never production.

    supports_negative_design() is False by construction: it never saw a negative example.
    """
    name = "ligandmpnn_benchmark"
    production = False

    def propose(self, ctx, n):
        raise NotImplementedError("TODO: wrap the frozen checkpoint for benchmarking only")

    def score(self, seq, state_path, pose, cfg):
        raise NotImplementedError("TODO: pocket-only log-likelihood, see autopsy/probes.py")


BACKENDS = {b.name: b for b in (PhysAlloDesignBackend, PhysicsDesignBackend,
                                LigandMPNNBenchmarkBackend)}


def get_backend(name):
    if name not in BACKENDS:
        raise SystemExit("unknown design backend '%s' (have: %s)" % (name, ", ".join(BACKENDS)))
    return BACKENDS[name]()


def run(ctx):
    """requires ctx['states'], ctx['poses'], ctx['masks'];  produces ctx['candidates']"""
    name = ctx["cfg"]["design"]["backend"]
    be = get_backend(name)
    if not be.production:
        raise SystemExit("backend '%s' is benchmark-only and must not generate production "
                         "candidates. Set design.backend to a production backend." % name)
    return dict(ctx, candidates=be.propose(ctx, ctx["cfg"]["design"]["raw_designs"]))
