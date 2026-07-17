"""Multi-objective posterior Thompson selection with a diversity penalty.

No fixed weighted score — weighting the objectives is the user's call, not baked into fitness.
Each candidate carries a set of joint-posterior surface samples reduced to objective vectors. Per
Thompson round we take one sample from every candidate, find the Pareto-non-dominated set among
them, and credit its members; a candidate's score is how often it lands on the sampled front — a
Thompson estimate of P(on the true Pareto front). A greedy diversity pass then spreads the chosen
batch so it is not N variants of one mechanism, and basal leak is enforced as a hard constraint,
not traded off.
"""

OBJ_KEYS = ("log_fold", "auc", "neg_log_EC50", "neg_t_on", "neg_basal")   # all maximised


def dominates(a, b):
    """a Pareto-dominates b (all objectives maximised): >= everywhere, > somewhere."""
    ge = all(a[k] >= b[k] for k in OBJ_KEYS)
    gt = any(a[k] > b[k] for k in OBJ_KEYS)
    return ge and gt


def thompson_pareto_prob(cand_samples):
    """cand_samples: {cid: [obj_vector | None] * R} (equal length R, None = non-responder draw).
    -> {cid: fraction of rounds this candidate sits on the sampled Pareto front}."""
    cids = list(cand_samples)
    R = len(next(iter(cand_samples.values()))) if cids else 0
    for c in cids:
        if len(cand_samples[c]) != R:
            raise ValueError("candidate %s has %d samples, expected %d (rounds must align for "
                             "Thompson)" % (c, len(cand_samples[c]), R))
    count = {c: 0 for c in cids}
    for r in range(R):
        live = {c: cand_samples[c][r] for c in cids if cand_samples[c][r] is not None}
        for c, o in live.items():
            if not any(dominates(o2, o) for c2, o2 in live.items() if c2 != c):
                count[c] += 1
    return {c: (count[c] / R if R else 0.0) for c in cids}


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def select(cand_samples, meta, n, basal_max=None, min_seq_dist=2):
    """-> dict(selected, prob, n_eligible, dropped_basal).

    meta: {cid: {'sequence':, 'basal':}}. Order of preference is Thompson front probability; a
    candidate is skipped if it sits within min_seq_dist of one already chosen (diversity), and
    dropped outright if its basal leak exceeds basal_max (hard constraint).
    """
    prob = thompson_pareto_prob(cand_samples)
    dropped = [c for c in cand_samples
               if basal_max is not None and meta.get(c, {}).get("basal", 0.0) > basal_max]
    eligible = [c for c in cand_samples if c not in dropped]
    eligible.sort(key=lambda c: -prob[c])

    chosen = []
    for c in eligible:
        s = meta.get(c, {}).get("sequence", "")
        if all(_hamming(s, meta.get(o, {}).get("sequence", "")) >= min_seq_dist for o in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return {"selected": chosen, "prob": prob, "n_eligible": len(eligible),
            "dropped_basal": dropped,
            "diversity_shortfall": max(0, n - len(chosen))}
