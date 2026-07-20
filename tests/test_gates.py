"""Gate logic must reject exactly the failure modes GPT-5.6 found in the first draft:
constitutive mutants, non-binders, and candidates with missing features (fail-closed).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from pipeline.rank import apply_gates, resolve_threshold, release_sign

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "config", "scoring.yaml")))
# This scaffold's own WT controls, PER METRIC. A single scalar per control is wrong:
# wt_native_holo is negative for ddG_coupling (strong shift) but POSITIVE for S_release
# (letting DNA go). Sharing one number across metrics silently inverts gates.
CTRL = {
    "wt_apo": {
        "dG_apo":     {"value":  2.0, "sigma": 0.5},   # apo prefers X_D by +2
        "E_DNA_X_D":  {"value":  1.0, "sigma": 0.5},   # apo holds the operator
        "clash_count":{"value":  1.0, "sigma": 0.5},
    },
    "wt_native_holo": {
        "E_L_I":        {"value": -4.0, "sigma": 0.5},  # native effector binds well
        "dG_lig":       {"value": -2.5, "sigma": 0.5},  # ligand-bound WT prefers X_I (negative)
        "ddG_coupling": {"value": -3.0, "sigma": 0.5},  # strong shift (negative)
        "S_release":    {"value":  2.5, "sigma": 0.5},  # lets DNA go (POSITIVE)
        # WT DEFINES the template ruler (sigma = spread across its own structure ensemble).
        # This used to be anchored on known_dead, which made the threshold uncomputable for any
        # scaffold lacking such a mutant - contradicting negative controls being optional.
        "template_similarity": {"value": 1.0, "sigma": 0.12},
    },
    # WT protein + the NEW TARGET. This, not the native effector, is what a candidate's binding
    # and selectivity are measured against: interface energy moves with molecule size, charge and
    # atom count, so comparing a new molecule's E_L_I to tetracycline's compares two chemistries.
    "wt_target": {
        "E_L_I":         {"value": -3.5, "sigma": 0.5},   # WT already binds the target somewhat
        "S_specificity": {"value":  0.0, "sigma": 0.4},   # undesigned WT is not yet selective
    },
    # Negative controls are optional in availability, mandatory in behaviour when declared.
    # They VERIFY the ruler separates broken variants; they no longer DEFINE any threshold.
    "known_constitutive": {"dG_apo": {"value": -1.0, "sigma": 0.5}},
    "known_nonresponder": {
        "template_similarity": {"value": 0.3, "sigma": 0.1},
        "S_specificity":       {"value": 0.0, "sigma": 0.4},
    },
}

def base():
    """a genuine sensor: binds, apo still prefers X_D, ligand shifts it, induced releases DNA"""
    return dict(E_L_I=-4.0, dG_apo=3.0, dG_lig=-2.0, ddG_coupling=-5.0, E_DNA_X_D=1.0,
                S_release=2.0, S_specificity=1.0, clash_count=2.0, template_similarity=0.8,
                all_states_packed=True, ligand_strain=0.5, pose_confidence=0.9)

def test_real_sensor_passes():
    ok, why = apply_gates(base(), CFG, CTRL, "inducible_repressor")
    assert ok, why

def test_constitutive_mutant_rejected():
    """apo ALREADY prefers the induced state -> always-ON. The first draft let this through."""
    f = base(); f["dG_apo"] = -1.0
    ok, why = apply_gates(f, CFG, CTRL, "inducible_repressor")
    assert not ok and any("constitutive" in w for w in why), why

def test_non_binder_rejected():
    """+100 vs +99 passed 'state preference' in the first draft. Must die on target_binding."""
    f = base(); f["E_L_I"] = 100.0; f["dG_lig"] = -1.0
    ok, why = apply_gates(f, CFG, CTRL, "inducible_repressor")
    assert not ok and any("does not bind" in w for w in why), why

def test_missing_feature_fails_closed():
    """the old code defaulted fold_clash to 0 and PASSED. Missing must now reject."""
    f = base(); del f["clash_count"]
    ok, why = apply_gates(f, CFG, CTRL, "inducible_repressor")
    assert not ok and any("missing" in w for w in why), why

def test_broken_allostery_rejected():
    f = base(); f["template_similarity"] = 0.2       # worse than known_dead
    ok, why = apply_gates(f, CFG, CTRL, "inducible_repressor")
    assert not ok and any("path broken" in w for w in why), why

def test_corepressor_sign_not_hardcoded():
    """PurR-like topology: the ligand STRENGTHENS DNA binding. A hard-coded +1 sign would have
    thrown away every corepressor scaffold."""
    f = base(); f["S_release"] = -2.0               # ligand strengthens the DNA interface
    ok_ind, _ = apply_gates(f, CFG, CTRL, "inducible_repressor")
    ok_cor, why = apply_gates(f, CFG, CTRL, "corepressor")
    assert not ok_ind, "should fail as an inducible repressor"
    assert ok_cor, why

def test_threshold_resolves_against_controls():
    assert resolve_threshold("wt_apo + 1.0*sigma", CTRL, "dG_apo") == 2.5
    assert resolve_threshold("wt_native_holo - 1.0*sigma", CTRL, "ddG_coupling") == -3.5
    assert resolve_threshold("wt_native_holo - 1.0*sigma", CTRL, "S_release") == 2.0
    assert release_sign("corepressor") == -1

if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", n); passed += 1
            except Exception as e:
                print("FAIL", n, "->", e); traceback.print_exc(); failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
