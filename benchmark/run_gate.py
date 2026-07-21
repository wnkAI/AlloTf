"""Score benchmark variants with the FROZEN model and three baselines, for the switch/non-switch gate.

This runs the finished six-state + RSM model on each manifest variant - it does not train or tune
anything. Four scorers are produced per variant so the gate can ask which one separates known
switches from known non-switches:

    binding_only : the induced-state ligand interface energy alone (the strawman - a good binder is
                   called good, with no regard for whether the switch actually flips)
    rsm          : weakest-link z over the seven functional margins (the model)
    g_resolvent  : the parameter-free pocket->DBD transmission margin alone
    rsm_plus_g   : weakest-link z over the seven margins AND the resolvent margin

WT calibration is label-free by construction: margins are oriented so >0 is the switch/no-switch
boundary, so neg_mean = 0 and only the scale (native_std) comes from WT. Nothing here ever sees a
mutant label, which is the whole point of the gate.
"""
import numpy as np

from pipeline import rsm as rsm_mod

# the seven margins RSM defines, plus the resolvent margin used only by rsm_plus_g
MARGIN_KEYS = rsm_mod.MARGIN_KEYS          # apo, lig, switch, dna, release, spec, integrity
RESOLVENT_KEY = "trans"


def margins_from_states(totals, e_dna_xd, s_spec, integrity):
    """Six-state totals -> the seven RSM margins (reuses the frozen definitions in rsm.py)."""
    return rsm_mod.margins(totals, e_dna_xd, s_spec, integrity)


def _z(margins, native_std, extra=None):
    """z_j = m_j / native_std_j (neg_mean = 0, the definitional boundary). extra adds resolvent."""
    m = dict(margins)
    if extra:
        m.update(extra)
    return {k: m[k] / (native_std.get(k, 1.0) + 1e-9) for k in m}


def variant_scores(totals, e_dna_xd, s_spec, integrity, native_std, m_trans, e_l_i):
    """All four scorers + the margin detail for one variant / one microstate.

    totals: {D0,I0,DL,IL,D_DNA,I_DNA}. e_dna_xd: apo D-state DNA affinity (more negative = binds
    better). s_spec: min-decoy minus target. integrity: >0-is-better structural score. m_trans:
    resolvent transmission margin. e_l_i: induced-state ligand interface energy (binding-only).
    native_std: per-margin WT scale (label-free).
    """
    m = margins_from_states(totals, e_dna_xd, s_spec, integrity)
    z = _z(m, native_std)
    m_worst, worst_key = rsm_mod.weakest_link(z)

    z_with_g = _z(m, native_std, extra={RESOLVENT_KEY: m_trans})
    m_worst_g, worst_key_g = rsm_mod.weakest_link(z_with_g)

    return {
        "binding_only": -e_l_i,               # higher = better binder (interface energy is negative)
        "rsm": m_worst,                        # weakest necessary condition (higher = more switch-like)
        "g_resolvent": m_trans / (native_std.get(RESOLVENT_KEY, 1.0) + 1e-9),
        "rsm_plus_g": m_worst_g,
        "margins": m,
        "z": z,
        "weakest_margin": worst_key,           # RSM's failure-mode attribution
        "weakest_margin_with_g": worst_key_g,
        "m_trans": m_trans,
    }


def score_manifest_row(six_totals, resolvent_g, spec, integrity, e_l_i, native_std):
    """Thin adapter: pull the pieces the four scorers need out of a scored six-state record."""
    from pipeline.state_builder import dna_affinity
    e_dna_xd = dna_affinity(six_totals, "D")
    if e_dna_xd is None:
        return None
    return variant_scores(six_totals, e_dna_xd, spec, integrity, native_std, resolvent_g, e_l_i)


# --- self test: the weakest-link must reject a great binder whose DNA arm has collapsed ---
if __name__ == "__main__":
    # a variant that binds the ligand beautifully but whose induced state binds DNA even better
    # (the sensor never lets go of the operator). binding_only should love it; rsm should not.
    native_std = {k: 1.0 for k in MARGIN_KEYS} | {RESOLVENT_KEY: 1.0}

    good = dict(D0=0.0, I0=4.0, DL=-8.0, IL=-9.0, D_DNA=-20.0, I_DNA=-17.0)  # rsm.py counterexample
    s_good = variant_scores(good, e_dna_xd=-20.0, s_spec=2.0, integrity=1.0,
                            native_std=native_std, m_trans=1.5, e_l_i=-9.0)

    healthy = dict(D0=0.0, I0=4.0, DL=-8.0, IL=-9.0, D_DNA=-20.0, I_DNA=-5.0)  # induced RELEASES DNA
    s_healthy = variant_scores(healthy, e_dna_xd=-20.0, s_spec=2.0, integrity=1.0,
                               native_std=native_std, m_trans=1.5, e_l_i=-9.0)

    print("collapsed-DNA:  binding_only=%.2f  rsm=%.2f  weakest=%s"
          % (s_good["binding_only"], s_good["rsm"], s_good["weakest_margin"]))
    print("healthy switch: binding_only=%.2f  rsm=%.2f  weakest=%s"
          % (s_healthy["binding_only"], s_healthy["rsm"], s_healthy["weakest_margin"]))
    assert s_good["binding_only"] == s_healthy["binding_only"], "same binder, binding-only cannot tell them apart"
    assert s_healthy["rsm"] > s_good["rsm"], "rsm must rank the real switch above the DNA-collapsed one"
    assert s_good["weakest_margin"] == "release", "rsm must attribute the failure to DNA release"
    print("OK")
