"""Build the SIX states per candidate through ONE PyRosetta backend instance.

    D0      no ligand, DNA-compatible backbone
    I0      no ligand, induced backbone
    DL      target ligand, DNA-compatible backbone
    IL      target ligand, induced backbone
    D_DNA   DNA-compatible backbone + operator
    I_DNA   induced backbone + operator

Why six and not four: the original four tested endpoint COMPATIBILITY, not ligand-INDUCED
switching. A constitutive mutant - one that already prefers I with no ligand - passed every gate.
D0/I0 are the constitutive filter, and they are what make the linkage a linkage:

    dG_apo   = E(I0) - E(D0)                     want > 0   no ligand -> still prefers D
    dG_lig   = E(IL) - E(DL)                     want < 0   ligand    -> prefers I
    ddG_coup = dG_lig - dG_apo                   want < 0   the LIGAND is what shifts it
    S_release= E_DNA(I) - E_DNA(D)               want > 0   (topology sign; corepressors invert)

ALL SIX must share, by construction and not by convention:
    same candidate sequence, same protonation scheme, same ligand .params version,
    same scorefunction, same repack/minimise protocol, same movable-residue mask.
A double difference between differently prepared states is uninterpretable. That is why every
state goes through one backend instance and the sameness is ASSERTED, not assumed.

Frozen on purpose: protein backbone, all jumps (ligand rigid body AND DNA). Free: design +
second-shell chi, and ligand internal torsions. A free backbone lets minimisation relax X_D and
X_I toward each other until the state difference evaporates; a free ligand rigid body turns the
double difference into a docking result.

CAVEAT: X_D and X_I are IMPOSED from native structures. These states test whether a sequence is
compatible with each conformation - they do NOT prove it can interconvert between them.
"""
STATES = ["D0", "I0", "DL", "IL", "D_DNA", "I_DNA"]
LIGANDED = {"DL", "IL"}
DNA_STATES = {"D_DNA", "I_DNA"}
# which native backbone each state is built on
BACKBONE = {"D0": "X_D", "DL": "X_D", "D_DNA": "X_D",
            "I0": "X_I", "IL": "X_I", "I_DNA": "X_I"}


def build_six(backend, candidate, templates, design_positions, second_shell=(), chain="A"):
    """backend: one PyRosettaBackend shared by all six states (this is the whole point).
    templates: {'X_D': pdb, 'X_I': pdb, 'X_D_DNA': pdb, 'X_I_DNA': pdb,
                'X_D_lig': pdb, 'X_I_lig': pdb}
    candidate: {pdb_resnum: 'ALA'}
    -> {state: score_terms_dict}. Any state that fails to build is a REJECT (None), never a zero.
    """
    key = {"D0": "X_D", "I0": "X_I", "DL": "X_D_lig", "IL": "X_I_lig",
           "D_DNA": "X_D_DNA", "I_DNA": "X_I_DNA"}
    out = {}
    for st in STATES:
        tpl = templates.get(key[st])
        if not tpl:
            out[st] = None
            continue
        pose = backend.prepare_pose(tpl)
        _assert_state_label(backend, pose, st, tpl)
        pose = backend.mutate_and_repack(pose, candidate, design_positions, second_shell, chain)
        pose = backend.restrained_minimize(pose, design_positions, second_shell,
                                           allow_ligand_torsions=True,
                                           ligand_rigid_body=False, chain=chain)
        out[st] = backend.score_terms(pose)
    assert_consistent(out)
    return out


def _assert_state_label(backend, pose, state, tpl):
    """A state must contain what its label promises. A holo PDB fed as D0, or an apo PDB fed as DL,
    poisons the double difference silently - nothing downstream re-checks (GPT-5.6). Verified here
    against the actual pose composition, before any energy is trusted."""
    if not hasattr(backend, "pose_composition"):
        return
    comp = backend.pose_composition(pose)
    has_lig, has_dna = comp["n_ligand"] > 0, comp["n_dna"] > 0
    if state in LIGANDED and not has_lig:
        raise RuntimeError("state %s (from %s) must contain the ligand but has none: mislabelled "
                           "template" % (state, tpl))
    if state not in LIGANDED and has_lig:
        raise RuntimeError("state %s (from %s) must be ligand-FREE but contains a ligand: the apo "
                           "and DNA states cannot carry the effector or the linkage is wrong"
                           % (state, tpl))
    if state in DNA_STATES and not has_dna:
        raise RuntimeError("state %s (from %s) must contain operator DNA but has none" % (state, tpl))
    if state not in DNA_STATES and has_dna:
        raise RuntimeError("state %s (from %s) must NOT contain DNA but does: only D_DNA/I_DNA are "
                           "operator complexes" % (state, tpl))


def assert_consistent(state_terms):
    """The six states must have been prepared identically. Checked, not trusted."""
    seen_sfxn, seen_params = set(), set()
    for st, t in state_terms.items():
        if not t:
            continue
        seen_sfxn.add(t.get("_sfxn"))
        seen_params.add(t.get("_params_hash"))
    if len(seen_sfxn) > 1:
        raise RuntimeError("states scored with different scorefunctions: %s" % seen_sfxn)
    if len(seen_params) > 1:
        raise RuntimeError("states used different ligand params: %s - the double difference "
                           "would be meaningless" % seen_params)


def totals(state_terms):
    return {st: (t["total_score"] if t else None) for st, t in state_terms.items()}


def linkage(energies):
    """energies: {state: total}. -> dict(dG_apo, dG_lig, ddG_coup) or None (fail closed).

    ref cancels here by construction: one sequence, identical composition in all four states.
    """
    need = ("D0", "I0", "DL", "IL")
    if any(energies.get(s) is None for s in need):
        return None
    dG_apo = energies["I0"] - energies["D0"]
    dG_lig = energies["IL"] - energies["DL"]
    return dict(dG_apo=dG_apo, dG_lig=dG_lig, ddG_coup=dG_lig - dG_apo)


def dna_affinity(energies, state):
    """DNA-binding energy of one backbone: E(X.DNA) - E(X0).

    The free-DNA term is a constant across states and cancels in any comparison, so it is omitted
    rather than fabricated.
    """
    dna, apo = ("I_DNA", "I0") if state == "I" else ("D_DNA", "D0")
    if energies.get(dna) is None or energies.get(apo) is None:
        return None
    return energies[dna] - energies[apo]


def dna_release(energies, topology_sign=+1):
    """S_release = [E(I.DNA) - E(I0)] - [E(D.DNA) - E(D0)], sign from topology (corepressors: -1).

    A DOUBLE difference, for exactly the reason the ligand linkage is one. The old version was
    E_DNA(I) - E_DNA(D), a single difference of total complex energies, and it is wrong:

        E(I.DNA) - E(D.DNA) = [true differential DNA affinity] + [E(I0) - E(D0)]
                            = S_release_true + dG_apo

    so it carried the apo bias as an additive contaminant. Because the apo gate REQUIRES
    dG_apo > 0, every candidate that passed that gate had its DNA-release score inflated by a
    positive amount - the two gates were coupled such that passing one guaranteed cheating the
    other. Found by GPT-5.6 with this counterexample, reproduced against this code:

        D0=0  I0=+4  DL=-8  IL=-9  D_DNA=-20  I_DNA=-17
        dG_apo=+4 PASS   dG_lig=-1 PASS   ddG_coup=-5 PASS   old S_release=+3 PASS
        true S_release = (-17-4) - (-20-0) = -1  ->  DNA binds the INDUCED state better.
        Every gate green, and the sensor never lets go of the operator.

    This is the same error class as the four-state bug: D0/I0 were added to make the ligand
    linkage a true double difference, and the DNA arm was left as a single one.
    """
    a_i = dna_affinity(energies, "I")
    a_d = dna_affinity(energies, "D")
    if a_i is None or a_d is None:
        return None
    return topology_sign * (a_i - a_d)


def run(ctx):
    """requires ctx['candidates'], ctx['states'], ctx['poses'], ctx['cfg']
       produces ctx['candidate_states']"""
    cfg = ctx["cfg"]["design"]
    if cfg.get("repacking_backend") != "pyrosetta":
        raise RuntimeError("production multistate scoring requires repacking_backend: pyrosetta; "
                           "our scoring.py is a geometry prefilter only (its optimum is poly-Gly)")
    from .physallo.rosetta_backend import PyRosettaBackend
    backend = PyRosettaBackend(score_function=cfg.get("score_function", "ref2015"),
                               ligand_params=ctx.get("ligand_params"))

    templates = ctx["states"]["paths"]
    design_positions = ctx["masks"]["recognition_mask"]
    second_shell = ctx["masks"].get("transduction_mask", ())
    chain = ctx["states"].get("chain", "A")
    # S_release is stored RAW here (topology_sign=+1). The topology sign is applied exactly once,
    # in rank.apply_gates via release_sign(). Applying it in both places squared it and a
    # corepressor lost its inversion, (-1)^2 = +1 (GPT-5.6).

    out, failures = {}, {}
    for cid, cand in ctx["candidates"].items():
        # design.propose emits full records; the {resnum: resname} map lives in 'residues'.
        # A bare dict (tests) is used as-is.
        residues = cand["residues"] if isinstance(cand, dict) and "residues" in cand else cand
        try:
            terms = build_six(backend, residues, templates, design_positions, second_shell, chain)
        except Exception as exc:
            # a candidate whose states could not be built is dropped, never zero-filled:
            # a zero here reads as "neutral energy" and sails through every gate.
            failures[cid] = str(exc)[:300]
            continue
        tot = totals(terms)
        link = linkage(tot)
        rel = dna_release(tot, +1)          # raw; topology sign applied once, in rank
        packed = all(v is not None for v in tot.values())
        if not packed:
            # a candidate missing any of the six states is dropped, not carried with None holes.
            # linkage/release would be None and the fail-closed REQUIRED check in rank would reject
            # it anyway, but keeping it here invites a later stage to read a half-built record.
            failures[cid] = "incomplete: states not built = %s" % (
                [s for s, v in tot.items() if v is None])
            continue
        out[cid] = {
            "terms": terms,
            "totals": tot,
            "all_states_packed": packed,
            "dG_apo": link["dG_apo"] if link else None,
            "dG_lig": link["dG_lig"] if link else None,
            "ddG_coupling": link["ddG_coup"] if link else None,
            "S_release": rel,
            # apo DNA competence is an AFFINITY, E(D.DNA)-E(D0), not the raw complex total: the
            # total carries D0's own stability and would gate on the wrong quantity (GPT-5.6).
            "E_DNA_X_D": dna_affinity(tot, "D"),
            "interface": {st: backend.interface_energy(terms[st]["_pose"])
                          if terms.get(st) and "_pose" in terms[st] else None
                          for st in ("DL", "IL")},
        }
    if not out:
        raise RuntimeError("no candidate produced six buildable states (%d failures). "
                           "First error: %s" % (len(failures),
                                                next(iter(failures.values()), "none")))
    return {"candidate_states": out, "state_failures": failures}
