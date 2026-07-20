"""Robust Switchability Margin (RSM). No free energies, no MD - a functional objective over the
six states that ranks by the WEAKEST necessary condition under structural/pose uncertainty.

A weighted sum lets a great binding score hide a broken DNA arm or a constitutive apo state. A
switch does not average - it fails at its worst necessary condition. RSM encodes exactly that:

  1. Each six-state energy becomes a positive functional margin (>0 = good), WT-calibrated to a
     z-score so margins of different physical meaning are comparable.
  2. One microstate's fitness is the WEAKEST margin: M = min_j z_j. A candidate is only good if
     every condition holds at once.
  3. Across microstates (structure pairs, poses, protonation, repacks) the score is the CVaR at
     alpha - the mean of the worst alpha fraction - so a candidate is rewarded for holding up in
     UNFAVOURABLE microstates, not for one lucky calculation.
  4. The weakest margin names the failure mode automatically.

All REU. P_consistent is a COMPUTATIONAL consistency probability, not a biophysical one.
"""
import numpy as np

MARGIN_KEYS = ("apo", "lig", "switch", "dna", "release", "spec", "integrity")

FAILURE_MODE = {
    "apo": "constitutive_on_risk",
    "lig": "ligand_nonresponsive",
    "switch": "binding_only",
    "dna": "dna_defective",
    "release": "no_dna_release",
    "spec": "decoy_preferring",
    "integrity": "structurally_unstable",
}


def margins(totals, e_dna_xd, s_spec, integrity):
    """One microstate -> raw functional margins, each oriented so >0 is good.

    totals: {D0,I0,DL,IL,D_DNA,I_DNA}. e_dna_xd: apo D-state DNA affinity S(D.DNA)-S(D0) (more
    negative = binds better). s_spec: min decoy energy - target energy. integrity: a >0-is-better
    structural score (e.g. -clash, -strain), already relative to WT.
    """
    m_apo = totals["I0"] - totals["D0"]                 # apo prefers D (not constitutive)
    m_lig = totals["DL"] - totals["IL"]                 # ligand tips it to I
    m_release = (totals["I_DNA"] - totals["I0"]) - (totals["D_DNA"] - totals["D0"])
    return {"apo": m_apo, "lig": m_lig, "switch": m_apo + m_lig,
            "dna": -e_dna_xd,                           # stronger apo-DNA binding -> larger margin
            "release": m_release, "spec": s_spec, "integrity": integrity}


def zscore(m, neg_mean, native_std, eps=1e-6):
    """z_j = (m_j - mu_j,negative) / (sigma_j,native + eps). WT-calibrated, dimensionless."""
    return {k: (m[k] - neg_mean.get(k, 0.0)) / (native_std.get(k, 1.0) + eps) for k in m}


def weakest_link(z):
    """(min z, the margin key that is weakest). The fitness of one microstate."""
    k = min(z, key=z.get)
    return z[k], k


def cvar(values, alpha=0.2):
    """Mean of the worst alpha fraction (the lowest values, since larger M is better).

    CVaR, not the minimum: a single catastrophic microstate should not own the score, but the
    unfavourable tail should. At least one value is always included.
    """
    v = np.sort(np.asarray(values, float))
    n = max(1, int(np.ceil(alpha * len(v))))
    return float(v[:n].mean())


def rsm(microstate_z, alpha=0.2):
    """microstate_z: list of z-score dicts, one per microstate. -> the RSM report.

    Ranking uses rsm_cvar (robust to the unfavourable tail). consistency and failure_mode are for
    reporting and diagnosis, not for the score.
    """
    if not microstate_z:
        return None
    per_M = []
    per_weak = []
    for z in microstate_z:
        m, k = weakest_link(z)
        per_M.append(m)
        per_weak.append(k)

    # failure mode = the margin that is worst in the MEDIAN microstate (stable across the ensemble)
    med = {j: float(np.median([z[j] for z in microstate_z])) for j in microstate_z[0]}
    worst_j = min(med, key=med.get)

    # pose_sensitive overrides: the candidate sometimes works and sometimes does not
    consistent = float(np.mean([m > 0 for m in per_M]))
    mode = FAILURE_MODE.get(worst_j, worst_j)
    if 0.2 < consistent < 0.8:
        mode = "pose_sensitive"

    return {"rsm_cvar": cvar(per_M, alpha),
            "M_median": float(np.median(per_M)),
            "M_worst": float(np.min(per_M)),
            "consistency": consistent,           # computational consistency probability
            "failure_mode": mode,
            "median_margins": med,
            "n_microstates": len(microstate_z)}
