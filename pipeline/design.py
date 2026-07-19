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


def _wt_chain_residues(pdb_path, chain_id):
    """-> [(resnum, resname)] of the design chain, in sequence order.

    Goes through load_assembly, not next(iter(structure)): a biological assembly stored as several
    MODEL blocks would otherwise collapse to its first chain.
    """
    from .structure import load_assembly
    from .physallo.aa_filter import one

    model = load_assembly(pdb_path)

    def protein_len(c):
        return sum(1 for r in c if r.id[0] == " " and one(r.get_resname().upper()))

    chain = {c.id: c for c in model}.get(chain_id)
    if chain is None or protein_len(chain) == 0:
        chain = max(model, key=protein_len)
    return [(r.id[1], r.get_resname().upper()) for r in chain
            if r.id[0] == " " and one(r.get_resname().upper())]


def _apply_residues(wt_chain, residues):
    """WT chain with the designed positions substituted -> full-length one-letter sequence, i.e.
    something that can actually be ordered as a gene."""
    from .physallo.aa_filter import one
    return "".join(one(residues.get(num, wt)) or "X" for num, wt in wt_chain)


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
        """Generate n pocket candidates on the induced-state scaffold.

        aa_filter (in prepare) -> joint sequence+rotamer annealing (backend.design, fast in-house
        energy) -> fast clash reject relative to WT -> unified candidate records. The in-house
        energy is a PREFILTER only: it decides which candidates are worth the expensive six-state
        PyRosetta pass, never the final ranking. Its optimum is poly-Gly, so nothing here is
        allowed to survive as a score - state_builder re-scores every survivor under ref2015.
        """
        import numpy as np
        from .physallo import backend as pb

        st = ctx["states"]
        masks = ctx["masks"]
        dcfg = ctx["cfg"]["design"]
        # design on the induced holo backbone: that is the conformation the ligand must stabilise
        scaffold = st["paths"].get("X_I_lig") or st["paths"].get("X_I")
        if not scaffold:
            raise RuntimeError("no induced-state scaffold (X_I_lig / X_I) to design on")
        ligand_resname = st.get("effector_resname")
        chain = st.get("chain", "A")

        # recognition is free; transduction is designed under a tighter mask; protected is fixed
        design_positions = list(masks["recognition_mask"])
        mask_of = {p: "recognition" for p in design_positions}
        for p in masks.get("transduction_mask", []):
            mask_of[p] = "transduction"
            if p not in design_positions:
                design_positions.append(p)
        if not design_positions:
            raise RuntimeError("empty recognition mask: nothing to design")

        pctx = pb.prepare(scaffold, design_positions, ligand_resname, chain, masks=mask_of)
        raw, space, efn = pb.design(
            pctx,
            n_candidates=max(n * 3, n + 10),
            n_steps=dcfg.get("search_steps", 4000),
            n_restarts=dcfg.get("search_restarts", 8),
            seed=dcfg.get("seed", 0))

        wt_e = efn(space.wt_state(np.random.RandomState(0)))
        clash_margin = dcfg.get("fast_clash_margin", 25.0)
        wt_chain = _wt_chain_residues(scaffold, chain)

        out = []
        for c in raw:
            if c["energy"] > wt_e + clash_margin:      # fast clash / strain reject vs WT
                continue
            residues = {p: c["state"][p][0] for p in c["state"]}
            out.append({
                "candidate_id": "cand_%04d" % len(out),
                # the pocket string is what the search optimises; the FULL chain is what gets
                # synthesised. Emitting only the 11-mer would hand the wet lab an unorderable gene.
                "design_site_sequence": c["seq"],
                "full_sequence": _apply_residues(wt_chain, residues),
                "sequence": _apply_residues(wt_chain, residues),
                "mutations": c["mutations"],           # [(pos, wt, mut)] relative to WT
                "residues": residues,                  # {resnum: resname} for state_builder
                "rotamers": {p: c["state"][p][1] for p in c["state"]},
                "fast_score": c["energy"],
                "structure_path": scaffold,
            })
            if len(out) >= n:
                break
        if not out:
            raise RuntimeError("PhysAllo produced no candidate within %.1f of WT (%d searched): "
                               "the pocket may be too constrained, or the clash margin too tight"
                               % (clash_margin, len(raw)))
        return out

    def score(self, seq, state_path, pose, cfg):
        """Fast in-house energy of one sequence on a backbone. PREFILTER / benchmark comparison
        only - not a production ranking (see propose). Needs cfg['design_positions'] and, if the
        pocket has a ligand, cfg['ligand_resname']; refuses rather than guess the design set."""
        from .physallo import backend as pb
        dp = cfg.get("design_positions")
        if not dp:
            raise ValueError("score needs cfg['design_positions'] to know which positions vary")
        pctx = pb.prepare(state_path, list(dp), cfg.get("ligand_resname"),
                          cfg.get("chain", "A"))
        efn = pb.make_energy_fn(pctx)
        from .physallo.aa_filter import one
        state = {}
        for i, p in enumerate(pctx.positions):
            want = seq[i] if i < len(seq) else None
            aa = next((a for a in pctx.allowed[p] if one(a) == want), pctx.allowed[p][0])
            rots = pb.rotamers.rotamers(aa) or [()]
            state[p] = (aa, rots[0])
        return efn(state)

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
    dcfg = ctx["cfg"]["design"]
    name = dcfg.get("backend", dcfg.get("generator", "physallo"))
    be = get_backend(name)
    if not be.production:
        raise SystemExit("backend '%s' is benchmark-only and must not generate production "
                         "candidates. Set design.generator to a production backend." % name)
    n = dcfg.get("initial_designs", dcfg.get("raw_designs", 8))
    proposed = be.propose(ctx, n)
    return dict(ctx, candidates={c["candidate_id"]: c for c in proposed})
