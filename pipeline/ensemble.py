"""Multistate scoring across a discrete ENSEMBLE, and the sign-probabilities that replace point
estimates. No MD, no free energies.

The single-structure linkage in state_builder is a point estimate: one D/I pair, one pose, one
rotamer packing. A reviewer's fair objection is that a fixed-backbone ref2015 difference is not a
free energy and that one crystal pair need not represent the scaffold. This module answers BOTH
without over-claiming:

  * ENSEMBLE FREE-ENERGY-LIKE AGGREGATE (still REU, never kcal/mol). For a state scored across M
    microstates (independent structure pairs, poses, rotamer repacks), the representative score is
    the log-sum-exp aggregate

        S_state = -(1/beta) * log sum_m w_m exp(-beta * S_state,m)

    beta is an EFFECTIVE parameter calibrated on the scaffold's own WT switch, not a physical
    1/kT. This is a soft-min: it lets the accessible low-score microstates dominate instead of
    trusting a single lucky minimum, which is the actual failure mode of point estimates.

  * SIGN PROBABILITY. The decision is not "is S_link below a threshold" but "how reliably is its
    sign correct across the ensemble":

        P_sign(S_link < 0) = fraction of microstate pairings whose S_link < 0

    A candidate whose linkage flips sign when the crystal pair changes is not a switch we believe,
    no matter how good its median. Gates move from value thresholds to P_sign >= 0.8.

Everything here is a PROXY: multistate linkage proxy and DNA-release compatibility score, not a
predicted dose-response.
"""
import numpy as np

from .state_builder import linkage, dna_release, dna_affinity


def logsumexp_aggregate(scores, beta, weights=None):
    """-(1/beta) log sum_m w_m exp(-beta * s_m). A soft-min over microstate REU scores.

    As beta -> inf this returns min(scores) (trust only the best microstate); as beta -> 0 it
    returns the weighted mean. The WT-calibrated beta sits between: accessible low-score microstates
    dominate without a single minimum owning the result.
    """
    s = np.asarray([v for v in scores if v is not None], float)
    if not len(s):
        return None
    w = np.ones(len(s)) if weights is None else np.asarray(weights, float)[:len(s)]
    w = w / w.sum()
    m = s.min()                                  # shift for numerical stability
    return float(m - (1.0 / beta) * np.log(np.sum(w * np.exp(-beta * (s - m)))))


def aggregate_state_totals(microstate_totals, beta):
    """microstate_totals: list of {state: total} (one per microstate).
    -> {state: aggregated total} via logsumexp per state. A state missing from any microstate is
    aggregated over the ones that have it; a state missing from ALL is None (fail closed)."""
    states = set().union(*[set(t) for t in microstate_totals]) if microstate_totals else set()
    out = {}
    for st in states:
        vals = [t.get(st) for t in microstate_totals if t.get(st) is not None]
        out[st] = logsumexp_aggregate(vals, beta) if vals else None
    return out


def sign_probability(values, want="negative"):
    """Fraction of microstate values with the desired sign. None-safe; returns (p, n_used).

    This is the number gates read: P_sign, not a magnitude. A candidate needs its sign to be
    RELIABLE across microstates, which a median cannot express.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, 0
    if want == "negative":
        k = sum(1 for v in vals if v < 0)
    elif want == "positive":
        k = sum(1 for v in vals if v > 0)
    else:
        raise ValueError("want must be 'negative' or 'positive'")
    return k / len(vals), len(vals)


def ensemble_linkage(microstate_totals, beta):
    """Per-microstate linkage AND the ensemble aggregate. -> dict or None.

    Returns median / worst / sign-probability for S_link, exactly the three numbers the layered
    budget requires, plus the logsumexp aggregate for reporting. 'worst' is the least favourable
    (largest, i.e. most positive) S_link - the microstate that most nearly breaks the switch.
    """
    per = [linkage(t) for t in microstate_totals]
    per = [p for p in per if p is not None]
    if not per:
        return None
    s_link = [p["ddG_coup"] for p in per]                 # S_link = ddG_coup (want < 0)
    p_sign, n = sign_probability(s_link, "negative")
    agg = linkage(aggregate_state_totals(microstate_totals, beta))
    return {
        "S_link_median": float(np.median(s_link)),
        "S_link_worst": float(max(s_link)),              # most positive = closest to non-switch
        "S_link_aggregate": agg["ddG_coup"] if agg else None,
        "P_sign_link": p_sign,
        "n_microstates": n,
        "dG_apo_median": float(np.median([p["dG_apo"] for p in per])),
        "dG_lig_median": float(np.median([p["dG_lig"] for p in per])),
        "per_microstate_S_link": s_link,
    }


def ensemble_dna_release(microstate_totals, beta, topology_sign=1):
    """Same treatment for the DNA-release compatibility score. -> dict or None.
    S_release > 0 wanted, so P_sign is over the POSITIVE sign."""
    per = [dna_release(t, topology_sign) for t in microstate_totals]
    per = [v for v in per if v is not None]
    if not per:
        return None
    p_sign, n = sign_probability(per, "positive")
    agg_totals = aggregate_state_totals(microstate_totals, beta)
    agg = dna_release(agg_totals, topology_sign)
    return {
        "S_release_median": float(np.median(per)),
        "S_release_worst": float(min(per)),              # least positive = closest to non-release
        "S_release_aggregate": agg,
        "P_sign_release": p_sign,
        "n_microstates": n,
        "E_DNA_X_D_median": float(np.median(
            [dna_affinity(t, "D") for t in microstate_totals if dna_affinity(t, "D") is not None])),
        "per_microstate_S_release": per,
    }


def calibrate_beta(wt_switch_microstate_totals, grid=None):
    """Choose beta on the scaffold's OWN WT switch: the beta whose aggregate S_link is most
    negative (sharpest recovery of the known native switch), within a bounded grid.

    beta is an EFFECTIVE parameter, not 1/kT. Calibrating it on the WT switch is what keeps the
    aggregate honest across scaffolds whose REU scales differ - the same reason every gate is
    WT-relative. Falls back to 1.0 if the WT linkage cannot be formed.
    """
    grid = grid or [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    best_beta, best_val = 1.0, None
    for b in grid:
        agg = linkage(aggregate_state_totals(wt_switch_microstate_totals, b))
        if agg is None:
            continue
        v = agg["ddG_coup"]
        if best_val is None or v < best_val:      # most negative = best recovery of the WT switch
            best_val, best_beta = v, b
    return best_beta, best_val


def robustness_verdict(link, release, p_min=0.8):
    """Layered-budget decision for a final-plate candidate. -> (ok, reasons).

    A candidate reaches the plate only if BOTH signs are reliable across the structure pairs. With
    exactly two pairs, 0.8 collapses to 'both agree', which is the intended floor.
    """
    reasons = []
    if link is None or link.get("P_sign_link") is None:
        reasons.append("no ensemble linkage")
    elif link["P_sign_link"] < p_min:
        reasons.append("P_sign_link=%.2f < %.2f: linkage sign flips across structure pairs"
                       % (link["P_sign_link"], p_min))
    if release is None or release.get("P_sign_release") is None:
        reasons.append("no ensemble DNA-release")
    elif release["P_sign_release"] < p_min:
        reasons.append("P_sign_release=%.2f < %.2f: DNA-release sign flips across structure pairs"
                       % (release["P_sign_release"], p_min))
    return (len(reasons) == 0), reasons
