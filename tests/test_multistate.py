"""Multistate contract tests - all runnable BEFORE PyRosetta arrives.

These do not check physics. They check that the software cannot produce a meaningless double
difference: same sequence, same scorefunction, same ligand params, frozen backbone/jumps, no
silent fallback, fail-closed on anything missing.

The mock only validates plumbing. It must never be able to produce a production candidate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.state_builder import (STATES, BACKBONE, build_six, assert_consistent, totals,
                                    linkage, dna_release)


class MockBackend:
    """Mimics PyRosettaBackend's contract and records what it was asked to do."""
    production = False           # a mock must never generate production candidates

    def __init__(self, energies=None, sfxn="ref2015", params_hash="abc123", ref_offset=0.0):
        self.energies = energies or {}
        self.sfxn_name = sfxn
        self.params_hash = params_hash
        self.ref_offset = ref_offset
        self.calls = []
        self._state = None

    def prepare_pose(self, pdb, ligand_params=None):
        self._state = os.path.basename(pdb).replace(".pdb", "")
        return dict(template=self._state, seq=None, movemap=None)

    def mutate_and_repack(self, pose, sequence, design_positions, repack=(), chain="A"):
        self.calls.append(("mutate", pose["template"], tuple(sorted(sequence.items()))))
        return dict(pose, seq=tuple(sorted(sequence.items())))

    def restrained_minimize(self, pose, design_positions, second_shell=(),
                            allow_ligand_torsions=True, ligand_rigid_body=False, chain="A"):
        if ligand_rigid_body:
            raise NotImplementedError("ligand rigid body must stay frozen")
        mm = dict(bb=False, jump=False, chi=True, ligand_torsions=allow_ligand_torsions)
        self.calls.append(("minimize", pose["template"], mm))
        return dict(pose, movemap=mm)

    def score_terms(self, pose):
        e = self.energies.get(pose["template"], -500.0) + self.ref_offset
        return {"total_score": e, "fa_rep": 1.0, "ref": 10.0 + self.ref_offset,
                "_sfxn": self.sfxn_name, "_params_hash": self.params_hash}


TPL = {"X_D": "X_D.pdb", "X_I": "X_I.pdb", "X_D_lig": "X_D_lig.pdb",
       "X_I_lig": "X_I_lig.pdb", "X_D_DNA": "X_D_DNA.pdb", "X_I_DNA": "X_I_DNA.pdb"}
E = {"X_D": -500.0, "X_I": -495.0, "X_D_lig": -510.0, "X_I_lig": -530.0,
     "X_D_DNA": -600.0, "X_I_DNA": -580.0}
CAND = {67: "HIS", 92: "ARG"}


def test_no_silent_fallback():
    """PyRosetta absent must raise, never fall back to our prefilter (whose optimum is poly-Gly)."""
    from pipeline.physallo import rosetta_backend as rb
    if rb.available():
        return
    try:
        rb.PyRosettaBackend()
        raise AssertionError("must not construct without PyRosetta")
    except RuntimeError as e:
        assert "No automatic fallback" in str(e), str(e)


def test_six_states_same_sequence():
    b = MockBackend(E)
    build_six(b, CAND, TPL, [67, 92])
    seqs = {c[2] for c in b.calls if c[0] == "mutate"}
    assert len(seqs) == 1, "states were built from different sequences: %s" % seqs
    assert len([c for c in b.calls if c[0] == "mutate"]) == 6


def test_backbone_and_jumps_are_frozen():
    b = MockBackend(E)
    build_six(b, CAND, TPL, [67, 92])
    for tag, tpl, mm in [c for c in b.calls if c[0] == "minimize"]:
        assert mm["bb"] is False, "backbone free -> X_D and X_I relax together, dG evaporates"
        assert mm["jump"] is False, "free jump -> ligand re-docks, ddG becomes a docking result"
        assert mm["ligand_torsions"] is True, "a rigid ligand is unphysical strain"


def test_same_scorefunction_all_states():
    b = MockBackend(E)
    st = build_six(b, CAND, TPL, [67, 92])
    assert len({t["_sfxn"] for t in st.values() if t}) == 1
    st["IL"]["_sfxn"] = "beta_nov16"
    try:
        assert_consistent(st); raise AssertionError("must reject mixed scorefunctions")
    except RuntimeError as e:
        assert "different scorefunctions" in str(e)


def test_same_ligand_params_all_states():
    b = MockBackend(E)
    st = build_six(b, CAND, TPL, [67, 92])
    st["DL"]["_params_hash"] = "deadbeef"
    try:
        assert_consistent(st); raise AssertionError("must reject mixed ligand params")
    except RuntimeError as e:
        assert "different ligand params" in str(e)


def test_ref_constant_cancels_in_linkage():
    a = linkage(totals(build_six(MockBackend(E), CAND, TPL, [67, 92])))
    b = linkage(totals(build_six(MockBackend(E, ref_offset=137.0), CAND, TPL, [67, 92])))
    assert abs(a["ddG_coup"] - b["ddG_coup"]) < 1e-9, \
        "ref offset changed ddG_coup: %s vs %s" % (a, b)
    assert abs(a["dG_apo"] - b["dG_apo"]) < 1e-9


def test_state_mapping_identical():
    """Every state must be built on the backbone it claims."""
    assert BACKBONE == {"D0": "X_D", "DL": "X_D", "D_DNA": "X_D",
                        "I0": "X_I", "IL": "X_I", "I_DNA": "X_I"}
    b = MockBackend(E)
    build_six(b, CAND, TPL, [67, 92])
    used = [tpl for tag, tpl, _ in b.calls if tag == "minimize"]
    assert sum(1 for u in used if u.startswith("X_D")) == 3
    assert sum(1 for u in used if u.startswith("X_I")) == 3


def test_missing_state_fails_closed():
    """A state that cannot be built must reject the candidate, never default to zero."""
    tpl = dict(TPL); tpl["X_I_lig"] = None
    st = build_six(MockBackend(E), CAND, tpl, [67, 92])
    assert st["IL"] is None
    assert linkage(totals(st)) is None, "missing state must fail closed, not silently score"


def test_end_to_end_plumbing():
    """states -> totals -> linkage -> dna_release -> gate-ready features."""
    st = build_six(MockBackend(E), CAND, TPL, [67, 92])
    t = totals(st)
    lk = linkage(t)
    rel = dna_release(t, topology_sign=+1)
    # E: I0-D0 = +5 (apo prefers D, good);  IL-DL = -20 (ligand prefers I, good)
    assert abs(lk["dG_apo"] - 5.0) < 1e-9, lk
    assert abs(lk["dG_lig"] + 20.0) < 1e-9, lk
    assert abs(lk["ddG_coup"] + 25.0) < 1e-9, lk      # -20 - (+5) = -25, ligand shifts D->I
    # S_release is a DOUBLE difference now (GPT-5.6): [I_DNA-I0] - [D_DNA-D0], NOT I_DNA-D_DNA.
    # (-580-(-495)) - (-600-(-500)) = -85 - (-100) = +15. DNA binds the D backbone more tightly
    # (-100 vs -85), so the induced state releases it: positive, good. The old single difference
    # (+20) silently carried the apo bias dG_apo=+5, which is exactly the contamination removed.
    assert abs(rel - 15.0) < 1e-9, rel
    assert dna_release(t, topology_sign=-1) == -rel   # corepressor inverts, no code edit


def test_mock_cannot_be_production():
    assert MockBackend.production is False


if __name__ == "__main__":
    import traceback
    p = f = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", n); p += 1
            except Exception as e:
                print("FAIL", n, "->", e); traceback.print_exc(); f += 1
    print("\n%d passed, %d failed" % (p, f))
    sys.exit(1 if f else 0)
