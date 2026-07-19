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
    Fs = F[np.ix_(ci, ti)]
    bg = np.asarray(background, float)
    if bg.ndim == 0:
        Fc = Fs - float(bg)
    else:
        # a per-(c,t) empty-vector surface: inducer autofluorescence varies with dose AND the plate
        # drifts with time, so a single averaged scalar cannot remove either
        if bg.shape != F.shape:
            raise ValueError("background surface %s does not match F %s" % (bg.shape, F.shape))
        Fc = Fs - bg[np.ix_(ci, ti)]
    return conc[ci], time[ti], np.clip(Fc, 0.0, None)


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
    # response here is INDUCTION over the own-time control, so its floor sits near zero; judge it
    # by span and by having a real positive rise, not by a max/min fold ratio.
    span = float(response.max() - response.min())
    if span < 1e-9 or response.max() <= 0:
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
    """Robust readouts, normalised against the SAME-TIMEPOINT zero-dose control.

    Cells keep growing and reporter protein keeps accumulating, so F(0,t) rises on its own.
    Dividing by one time-averaged basal scores that drift as induction. Everything below is
    computed against F(0,t) at the matching t:

        dF(c,t)   = F(c,t) - F(0,t)
        fold(c,t) = (F(c,t)+eps) / (F(0,t)+eps)

    -> dict(basal, basal_mean, max_fold, EC50, t_on_bin, auc, responder, note)
    responder=False (with a note) when nothing ever rises MIN_FOLD over its own-time control; EC50
    and t_on_bin are then None rather than invented.
    """
    conc, time, F = np.asarray(conc, float), np.asarray(time, float), np.asarray(F, float)
    zero_i = int(np.argmin(conc))            # the zero dose (or the lowest one measured)
    baseline = F[zero_i]                     # per-timepoint control trace, NOT a scalar
    eps = 1e-6

    delta = F - baseline[None, :]
    fold_ct = (F + eps) / (baseline[None, :] + eps)
    max_fold = float(fold_ct.max())
    responder = max_fold >= MIN_FOLD

    top_i = int(np.argmax(conc))
    dtop = delta[top_i]                      # top-dose induction over its own-time control
    # dose-response on the endpoint INDUCTION, so growth cancels out of EC50 too
    ec50 = dose_ec50(conc, delta[:, -1]) if responder else None

    t_on = None
    if responder:
        span = float(dtop.max())
        if span > 0:
            above = np.where(dtop >= on_fraction * span)[0]
            if len(above):
                t_on = float(time[above[0]])

    auc = float(_trapz(np.clip(dtop, 0.0, None), time)) if len(time) > 1 else 0.0

    return {"basal": float(baseline[-1]),          # leak at the last read (steady state)
            "basal_mean": float(baseline.mean()),
            "max_fold": max_fold, "EC50": ec50,
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
