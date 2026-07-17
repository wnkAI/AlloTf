"""The only experimental readout: a fluorescence matrix F(c, t), and the reward it becomes.

Wet-lab measures one thing per candidate - fluorescence over a grid of inducer concentrations and
post-induction times. Nothing subjective is recorded (no hand-noted "time fluorescence appeared");
the whole time course is kept and the functional phenotypes are fit from it here.

A dose-time response is one separable surface:

    F(c, t) = B + Amax * hill(c) * rise(t)
    hill(c) = c^n / (EC50^n + c^n)
    rise(t) = (1 - exp(-k * (t - tau)))  clamped to 0 before tau

Six phenotypes come straight out of the fit: B (basal leak), Amax (dynamic amplitude), EC50
(sensitivity), n (Hill cooperativity), k (response speed), tau (lag). fold induction and t50 are
derived. The reward is a weighted combination the user tunes to specify the sensor they want.

The fit is done honestly: bounded parameters, multiple starts, and a REFUSAL (None) when the data
cannot support a fit - a flat non-responder must come back as "no induction", never as a fabricated
EC50 that would then be rewarded. That fail-closed instinct is the same one the physics side uses.
"""
import numpy as np

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False

# default reward weights. Positive term = higher is better; the signs live in reward(), not here.
DEFAULT_WEIGHTS = {"fold": 1.0, "ec50": 0.3, "t50": 0.1, "basal": 0.2}


def response_surface(ct, B, Amax, EC50, n, k, tau):
    """ct: (M,2) array of (concentration, time). Vectorised so curve_fit can call it directly."""
    c = np.asarray(ct)[:, 0]
    t = np.asarray(ct)[:, 1]
    with np.errstate(over="ignore", invalid="ignore"):
        hill = c ** n / (EC50 ** n + c ** n)
        hill = np.where((c == 0) & (EC50 > 0), 0.0, hill)
        rise = np.clip(1.0 - np.exp(-k * (t - tau)), 0.0, None)
    return B + Amax * hill * rise


def _initial_guess(conc, time, F):
    B0 = float(np.min(F))
    Amax0 = max(float(np.max(F) - B0), 1e-3)
    pos = conc[conc > 0]
    EC0 = float(np.median(pos)) if len(pos) else 1.0
    # crude rise-time guess: first time the top-dose trace passes half of its own span
    return [B0, Amax0, EC0, 1.5, 0.5, float(np.min(time))]


def fit(conc, time, F, bounds=None, max_starts=4, r2_min=0.5):
    """conc, time: 1D grids. F: (len(conc), len(time)) fluorescence matrix.

    -> dict(params, phenotypes, rmse, r2, ok, note). ok=False (with a reason) when the surface
    cannot be fit or the Hill structure does not explain the data - not an exception, so a batch of
    candidates where some fail does not abort the whole round.

    The non-responder test is POST-fit on R^2, not a pre-fit span/noise ratio. A span-over-noise
    threshold is fooled by point count: across dozens of pure-noise points, max-minus-min is several
    sigma regardless, so it never fires. R^2 asks the right question - did the dose-time Hill
    structure actually account for the variance, or is this just scatter around a constant?
    """
    if not _SCIPY:
        raise RuntimeError("fluorescence fitting needs scipy (pip install scipy)")
    conc = np.asarray(conc, dtype=float)
    time = np.asarray(time, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.shape != (len(conc), len(time)):
        raise ValueError("F must be (len(conc), len(time)) = (%d,%d), got %s"
                         % (len(conc), len(time), F.shape))

    span = float(np.max(F) - np.min(F))
    if span < 1e-9:
        return {"ok": False, "note": "F is constant: non-responder", "params": None,
                "phenotypes": None, "rmse": None, "r2": None}

    CC, TT = np.meshgrid(conc, time, indexing="ij")
    ct = np.column_stack([CC.ravel(), TT.ravel()])
    y = F.ravel()

    lo = [np.min(F) - span, 1e-3, max(np.min(conc[conc > 0]) * 1e-2, 1e-6) if np.any(conc > 0) else 1e-6,
          0.5, 1e-3, float(np.min(time))]
    hi = [np.max(F) + span, span * 5 + 1e-3, np.max(conc) * 1e2 + 1e-6,
          6.0, 100.0, float(np.max(time)) + 1e-6]
    if bounds is not None:
        lo, hi = bounds

    best = None
    rng = np.random.default_rng(0xF10A)
    p0 = _initial_guess(conc, time, F)
    for s in range(max_starts):
        guess = p0 if s == 0 else [
            np.clip(p0[i] * (1 + 0.5 * rng.standard_normal()), lo[i], hi[i]) for i in range(6)]
        try:
            popt, _ = curve_fit(response_surface, ct, y, p0=guess, bounds=(lo, hi), maxfev=20000)
        except Exception:
            continue
        pred = response_surface(ct, *popt)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rmse < best[1]:
            best = (popt, rmse)
    if best is None:
        return {"ok": False, "note": "curve_fit did not converge from any start",
                "params": None, "phenotypes": None, "rmse": None, "r2": None}

    popt, rmse = best
    pred = response_surface(ct, *popt)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    B, Amax, EC50, n, k, tau = [float(x) for x in popt]

    if r2 < r2_min:
        # the Hill surface explains too little: forcing phenotypes here would invent an EC50 for a
        # non-responder, which the reward would then happily score. Refuse.
        return {"ok": False, "note": "R2=%.3f < %.2f: Hill structure does not explain the data "
                "(non-responder or too noisy)" % (r2, r2_min), "params": None,
                "phenotypes": None, "rmse": rmse, "r2": r2}

    pheno = phenotypes(B, Amax, EC50, n, k, tau)
    return {"ok": True, "note": "", "params": dict(B=B, Amax=Amax, EC50=EC50, n=n, k=k, tau=tau),
            "phenotypes": pheno, "rmse": rmse, "r2": r2}


def phenotypes(B, Amax, EC50, n, k, tau):
    """Derived functional readouts. fold uses (B+Amax)/B with a floor so a zero-leak sensor does
    not divide by zero - a genuinely leak-free sensor has effectively infinite fold, capped here."""
    basal = max(B, 1e-6)
    fold = (B + Amax) / basal
    t50 = tau + np.log(2.0) / k if k > 0 else float("inf")
    return {"basal": B, "amplitude": Amax, "EC50": EC50, "hill_n": n,
            "rate_k": k, "lag_tau": tau,
            "fold_induction": float(fold), "t50": float(t50)}


def reward(pheno, weights=None):
    """R = w_fold*log2(fold) - w_ec50*log10(EC50) - w_t50*t50 - w_basal*basal.

    Higher fold, lower EC50, faster response, lower leak -> higher reward. Weights let the user
    define what "good" means for their application without touching this formula.
    A None phenotype set (fit failed / non-responder) yields None: an unmeasured sensor is not a
    zero-reward sensor, and averaging it in as zero would reward giving up.
    """
    if pheno is None:
        return None
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    fold = max(pheno["fold_induction"], 1e-6)
    ec50 = max(pheno["EC50"], 1e-9)
    t50 = pheno["t50"] if np.isfinite(pheno["t50"]) else 1e3
    return (w["fold"] * np.log2(fold)
            - w["ec50"] * np.log10(ec50)
            - w["t50"] * t50
            - w["basal"] * pheno["basal"])


def specificity_reward(target_pheno, decoy_phenos, weights=None):
    """R_spec = R(target) - max_d R(decoy). Positive = the sensor responds to the target more than
    to any analogue. Decoys that could not be fit are dropped (unmeasured != beats the target)."""
    rt = reward(target_pheno, weights)
    if rt is None:
        return None
    rds = [reward(p, weights) for p in decoy_phenos]
    rds = [r for r in rds if r is not None]
    if not rds:
        return {"specificity": None, "note": "no decoy could be fit: specificity unresolved"}
    return {"specificity": rt - max(rds), "R_target": rt, "R_best_decoy": max(rds)}
