"""Clean the raw F(c, t) fluorescence surface and read robust functional readouts off it.

Deliberately does NOT fit rate/lag kinetics (k, tau). At a 6 h read cadence those two are unstable
and would inject noise that the downstream GP then has to unlearn. response_gp learns the whole
surface; this module only (a) cleans plate data and (b) extracts readouts that survive coarse time
sampling. The SAME extractor runs on a measured surface and on a GP-predicted surface, so a
prediction and a measurement are summarised identically.

Readouts (no k, no tau, no fixed-weight reward — multi-objective selection lives in acquisition):
    basal        leak at zero inducer
    max_fold     max_{c,t} F / basal
    EC50         from a 1-D dose-response Hill fit at the endpoint (stable: one curve over 5 doses)
    t_on_bin     first 6 h time bin where the top dose crosses a fraction of its own span
    auc          time-integrated induced response at the top dose (0..T)
"""
import numpy as np

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False

ON_FRACTION = 0.5        # "response has turned on" = half of the trace's own span
MIN_FOLD = 1.5           # below this the well is treated as a non-responder for EC50 / t_on

# numpy 2.x renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def clean(conc, time, F, background=0.0):
    """-> (conc, time, F) as sorted float arrays with background subtracted and clipped at 0.

    F is (len(conc), len(time)). background is the empty-vector / autofluorescence level; it is
    subtracted, not ignored, because basal leak is one of the objectives and raw wells carry a
    constant plate offset.
    """
    conc = np.asarray(conc, float)
    time = np.asarray(time, float)
    F = np.asarray(F, float)
    if F.shape != (len(conc), len(time)):
        raise ValueError("F must be (len(conc), len(time)) = (%d,%d), got %s"
                         % (len(conc), len(time), F.shape))
    ci = np.argsort(conc)
    ti = np.argsort(time)
    return conc[ci], time[ti], np.clip(F[np.ix_(ci, ti)] - background, 0.0, None)


def _hill(c, bottom, top, ec50, n):
    return bottom + (top - bottom) * (c ** n) / (ec50 ** n + c ** n)


def dose_ec50(conc, response):
    """1-D Hill fit over concentration (a fixed timepoint's dose-response). Returns EC50 or None.

    Only the dose dimension is fit — it has a real sigmoid and five points, so it is stable, unlike
    the time dimension at 6 h spacing. Zero-concentration points are kept for the bottom plateau.
    Returns None (not a fabricated number) when the response is too flat to define an EC50.
    """
    conc = np.asarray(conc, float)
    response = np.asarray(response, float)
    pos = conc > 0
    if not _SCIPY or pos.sum() < 3:
        return None
    span = float(response.max() - response.min())
    if span < 1e-9 or (response.max() / max(response.min(), 1e-9)) < MIN_FOLD:
        return None
    cpos = conc[pos]
    lo = [response.min() - span, response.min(), cpos.min() * 1e-2, 0.5]
    hi = [response.max() + span, response.max() + span, cpos.max() * 1e2, 6.0]
    p0 = [response.min(), response.max(), float(np.median(cpos)), 1.5]
    try:
        popt, _ = curve_fit(_hill, conc, response, p0=p0, bounds=(lo, hi), maxfev=10000)
    except Exception:
        return None
    ec50 = float(popt[2])
    if ec50 <= cpos.min() * 1e-2 * 1.001 or ec50 >= cpos.max() * 1e2 * 0.999:
        return None                         # rail-pinned: EC50 is outside the tested range
    return ec50


def phenotypes(conc, time, F, on_fraction=ON_FRACTION):
    """Robust readouts from a cleaned surface. Works on measured OR GP-predicted F.

    -> dict(basal, max_fold, EC50, t_on_bin, auc, responder, note)
    responder=False (with a note) when the surface never rises MIN_FOLD over basal — its EC50 and
    t_on_bin are then None rather than invented.
    """
    conc, time, F = np.asarray(conc, float), np.asarray(time, float), np.asarray(F, float)
    zero = np.isclose(conc, 0.0)
    basal = float(F[zero].mean()) if zero.any() else float(F[conc.argmin()].mean())
    basal_eff = max(basal, 1e-6)
    max_fold = float(F.max() / basal_eff)

    responder = max_fold >= MIN_FOLD
    top = F[conc.argmax()]                  # top-dose time course
    ec50 = dose_ec50(conc, F[:, -1]) if responder else None    # endpoint dose-response

    t_on = None
    if responder:
        span = top.max() - basal
        if span > 0:
            thr = basal + on_fraction * span
            above = np.where(top >= thr)[0]
            if len(above):
                t_on = float(time[above[0]])

    auc = float(_trapz(np.clip(top - basal, 0, None), time)) if len(time) > 1 else 0.0

    return {"basal": basal, "max_fold": max_fold, "EC50": ec50,
            "t_on_bin": t_on, "auc": auc, "responder": responder,
            "note": "" if responder else "non-responder (max_fold %.2f < %.2f)" % (max_fold, MIN_FOLD)}


def objectives(pheno):
    """Map readouts to a common 'higher is better' objective vector for multi-objective selection.

    Returns None on a non-responder (no EC50 / t_on). Selection (acquisition.py) consumes these;
    there is deliberately NO scalar reward here — weighting the objectives is the user's call,
    made per run, not baked into the fitness.
    """
    if not pheno.get("responder") or pheno.get("EC50") is None:
        return None
    return {
        "log_fold": float(np.log2(max(pheno["max_fold"], 1e-6))),
        "auc": pheno["auc"],
        "neg_log_EC50": -float(np.log10(max(pheno["EC50"], 1e-9))),
        "neg_t_on": -(pheno["t_on_bin"] if pheno["t_on_bin"] is not None else 1e3),
        "neg_basal": -pheno["basal"],
    }
