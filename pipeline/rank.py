"""Hard gates -> Pareto -> diversity. REVISED after GPT-5.6 review.

Fixed in this revision:
  * fail-closed: a missing feature is a REJECT. The old code did features.get("fold_clash", 0)
    which silently PASSED any candidate whose clash count was never computed.
  * no raw-zero thresholds: every gate is resolved against the scaffold's own WT controls
    ("wt_apo + 1.0*sigma"), because absolute energies are not comparable across structures,
    operators or preparations.
  * apo-bias and target-binding gates added: without them constitutive mutants and outright
    non-binders (+100 vs +99) passed.
  * the DNA-release sign comes from topology, not hard-coded: corepressors invert it.
  * Pareto front over objectives; weights only break ties inside the front.
"""
import re

# a candidate must carry every one of these, or it is rejected outright (fail closed)
REQUIRED = ["E_L_I", "dG_apo", "dG_lig", "ddG_coupling", "E_DNA_X_D", "S_release",
            "S_specificity", "clash_count", "template_similarity", "all_states_packed"]
# dG_lig is REQUIRED and now also GATED: its presence was checked but its sign never was, which
# let an apo-dominated non-switch through the coupling gate (GPT-5.6).

_EXPR = re.compile(r"^\s*(?P<ctrl>[A-Za-z_]+)\s*(?P<op>[+-])\s*(?P<k>[\d.]+)\s*\*\s*sigma\s*$")


def resolve_threshold(expr, controls, metric):
    """'wt_apo + 1.0*sigma' -> float, using this scaffold's control value FOR THIS METRIC.

    controls is per-metric and must be: {control_name: {metric_name: {'value':..,'sigma':..}}}
    A single scalar per control is WRONG: wt_native_holo is negative for ddG_coupling (a strong
    shift) but positive for S_release (letting DNA go). Sharing one number across metrics silently
    inverts gates - caught by tests/test_gates.py::test_corepressor_sign_not_hardcoded.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    s = str(expr).strip()
    m = _EXPR.match(s)
    if m:
        name = m.group("ctrl")
        try:
            c = controls[name][metric]
        except KeyError:
            raise KeyError("no control '%s' calibrated for metric '%s' - calibration is mandatory,"
                           " never fall back to a raw zero" % (name, metric))
        k = float(m.group("k")) * c["sigma"]
        return c["value"] + (k if m.group("op") == "+" else -k)
    m2 = re.match(r"^\s*([\d.]+)\s*\*\s*sigma\s*$", s)
    if m2:
        sig = max(c[metric]["sigma"] for c in controls.values() if metric in c)
        return float(m2.group(1)) * sig
    return float(s)


def release_sign(topology_mode, route_direction=None):
    """+1 inducible repressor (ligand weakens DNA), -1 corepressor (ligand strengthens it)."""
    if topology_mode == "corepressor":
        return -1
    if topology_mode == "inducible_repressor":
        return +1
    if route_direction in ("corepressor", "co-repressor"):
        return -1
    return +1


def apply_gates(f, cfg, controls, topology_mode="auto", route_direction=None):
    """-> (passed: bool, reasons: list[str]). Fail closed on any missing feature."""
    g = cfg["hard_gates"]
    bad = []

    if cfg.get("fail_closed", True):
        missing = [k for k in REQUIRED if k not in f or f[k] is None]
        if missing:
            return False, ["missing:" + ",".join(missing)]
    if not f.get("all_states_packed", False):
        bad.append("a state failed to pack")

    T = lambda e, metric: resolve_threshold(e, controls, metric)

    if f["E_L_I"] > T(g["target_binding"]["max"], "E_L_I"):
        bad.append("target does not bind (E_L_I above WT native effector)")
    if f["dG_apo"] < T(g["apo_state_bias"]["min"], "dG_apo"):
        bad.append("constitutive: apo already prefers the induced state")
    if f["ddG_coupling"] > T(g["coupling"]["max"], "ddG_coupling"):
        bad.append("ligand does not shift the state population (weak linkage)")
    if "ligand_prefers_induced" in g and \
            f["dG_lig"] > T(g["ligand_prefers_induced"]["max"], "dG_lig"):
        bad.append("holo protein still prefers D: ddG_coupling was bought by apo bias, not a switch")
    if f["E_DNA_X_D"] > T(g["apo_dna_competence"]["max"], "E_DNA_X_D"):
        bad.append("apo can no longer hold the operator")

    s = release_sign(topology_mode, route_direction)
    if s * f["S_release"] < T(g["dna_release"]["min"], "S_release"):
        bad.append("induced state does not release DNA (sign-corrected)")

    if f["S_specificity"] < T(g["specificity"]["min"], "S_specificity"):
        bad.append("not selective against decoys")
    if f["clash_count"] > T(g["state_integrity"]["fold_clash"]["max"], "clash_count"):
        bad.append("fold clashes above WT")
    if f.get("ligand_strain", 1e9) > g["state_integrity"]["ligand_strain"]["max"]:
        bad.append("ligand strained")
    # S_state (interface) and ddG_coup (double difference) are the same claim measured two ways.
    # ligand_score.consistency() flags when they disagree; a disagreement means the pose moved
    # during packing, so the candidate is rejected rather than averaged (GPT-5.6 caught this being
    # computed and then ignored). Only enforced when the field is present.
    if "agree" in f and f["agree"] is False:
        bad.append("S_state and ddG_coupling disagree: unstable pose, not a trustworthy switch")
    if f.get("pose_confidence", -1) < g["state_integrity"]["pose_confidence"]["min"]:
        bad.append("low pose confidence")
    if f["template_similarity"] < T(g["allosteric_template"]["min"], "template_similarity"):
        bad.append("allosteric path broken (worse than known dead mutants)")

    return (len(bad) == 0), bad


# what each control MUST do to the gates, and - for the negatives - WHICH gate must be the one that
# catches it. A constitutive mutant that fails only because its ligand is strained is not evidence
# the apo-bias gate works; it is a coincidence that would collapse the moment a real design fails
# the same way. So the negatives name their diagnostic gate (GPT-5.6).
CONTROL_EXPECT = {
    "wt_native_holo":     {"pass": True},
    "known_constitutive": {"pass": False, "by": "constitutive: apo already prefers the induced state"},
    "known_dead":         {"pass": False, "by": "ligand does not shift the state population (weak linkage)"},
    "known_nonbinder":    {"pass": False, "by": "target does not bind (E_L_I above WT native effector)"},
}


def check_controls(control_scores, cfg, controls, topology_mode="auto", route_direction=None):
    """Before ANY design is generated: the scaffold must reproduce its own controls.
    wt_native_holo must PASS the gates; known_constitutive and known_dead must FAIL.
    -> (ok, report). A scaffold that fails this must not enter design.

    This is the calibration check, and it is the only thing standing between us and a confidently
    ranked list computed on a scaffold whose energies mean nothing. It runs BEFORE design, so the
    cost of a bad scaffold is one control run, not ten thousand designs.

    A control that is absent is not a pass. Every control named in CONTROL_EXPECT must be present
    and behave, or the scaffold is rejected - a scaffold with no known constitutive mutant simply
    cannot be validated this way, and pretending otherwise is how the constitutive-mutant bug got
    through the first time.
    """
    report, ok = {}, True
    missing = [k for k in CONTROL_EXPECT if k not in control_scores]
    if missing:
        return False, {"missing_controls": missing,
                       "note": "cannot calibrate: no control means no evidence the gates work "
                               "on this scaffold"}
    for name, want in CONTROL_EXPECT.items():
        passed, reasons = apply_gates(control_scores[name], cfg, controls,
                                      topology_mode, route_direction)
        good = (passed == want["pass"])
        # a negative control must fail FOR THE RIGHT REASON: the diagnostic gate must fire, not
        # merely some gate. Otherwise a lucky failure masquerades as a working filter.
        right_reason = True
        if not want["pass"] and good:
            right_reason = any(want["by"] in r for r in reasons)
            good = good and right_reason
            ok &= good
        else:
            ok &= good
        report[name] = {"passed": passed, "expected": want["pass"], "as_expected": good,
                        "reasons": reasons,
                        "diagnostic_gate": want.get("by"),
                        "caught_by_right_gate": right_reason if not want["pass"] else None}
    if not ok:
        report["verdict"] = ("this scaffold does not separate its own controls (or a control fails "
                            "for the wrong reason): the gates cannot tell a working sensor from a "
                            "broken one here. Do not design on it.")
    return ok, report


def dominates(a, b, objectives, maximize):
    """a dominates b: at least as good everywhere, strictly better somewhere."""
    better = False
    for k in objectives:
        av, bv = a.get(k), b.get(k)
        if av is None or bv is None:
            return False
        if maximize.get(k, True):
            if av < bv:
                return False
            if av > bv:
                better = True
        else:
            if av > bv:
                return False
            if av < bv:
                better = True
    return better


def pareto_front(cands, objectives, maximize):
    """-> non-dominated subset.

    A weighted sum would let a spectacular binding energy buy its way past a dead linkage. The
    front keeps the trade-off explicit; weights only order candidates INSIDE it.
    """
    front = []
    for c in cands:
        if not any(dominates(o, c, objectives, maximize) for o in cands if o is not c):
            front.append(c)
    return front


def _seq_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def cluster_diverse(cands, n, key="sequence", min_dist=2):
    """Greedy max-min selection: never hand back 96 designs that are the same design.

    Ranked order is respected (best first); a candidate is skipped only if it is within min_dist of
    something already chosen. If the pool is too homogeneous to fill n, that is reported by
    returning fewer - padding with near-duplicates would fake diversity.
    """
    out = []
    for c in cands:
        s = c.get(key, "")
        if all(_seq_distance(s, o.get(key, "")) >= min_dist for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def run(ctx):
    """requires ctx['candidate_features'], ctx['controls'], ctx['cfg_scoring']
       produces ctx['ranked'] -> ranked_candidates.csv + final_N.fasta"""
    import csv
    import os

    cfg = ctx["cfg_scoring"]
    controls = ctx["controls"]
    topo = cfg.get("topology", {}).get("mode", "auto")
    direction = (ctx.get("route") or {}).get("direction")

    cs = ctx.get("control_scores")
    fail_closed = cfg.get("fail_closed", True)
    if not cs:
        # calibration is mandatory: without controls we cannot show the gates work on this
        # scaffold, and an uncalibrated ranking is exactly the thing this module exists to refuse.
        if fail_closed:
            raise RuntimeError("no control_scores: the scaffold's own controls are required before "
                               "ranking (wt must pass, constitutive/dead/nonbinder must fail). "
                               "Set fail_closed:false only to knowingly skip calibration.")
    else:
        ok, report = check_controls(cs, cfg, controls, topo, direction)
        ctx["control_report"] = report
        if not ok and fail_closed:
            raise RuntimeError("scaffold failed its own control separation: %s"
                               % report.get("verdict", report))

    kept, rejected = [], []
    for c in ctx["candidate_features"]:
        passed, reasons = apply_gates(c, cfg, controls, topo, direction)
        (kept if passed else rejected).append(dict(c, _reasons=reasons))

    objectives = cfg["pareto"]["objectives"]
    maximize = cfg["pareto"].get("maximize", {})
    front = pareto_front(kept, objectives, maximize) if kept else []

    w = cfg["pareto"].get("weights", {})
    def rank_key(c):
        s = 0.0
        for k, wt in w.items():
            v = c.get(k)
            if v is None:
                continue
            s += wt * (v if maximize.get(k, True) else -v)
        return -s
    front.sort(key=rank_key)

    n_final = ctx.get("n_final", cfg.get("final_designs", 96))
    final = cluster_diverse(front, n_final,
                            min_dist=cfg.get("diversity", {}).get("min_mutations", 2))

    out_dir = ctx.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ranked_candidates.csv")
    cols = ["id", "sequence"] + REQUIRED + ["_reasons"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for c in front:
            wr.writerow({k: c.get(k) for k in cols})
    fasta_path = os.path.join(out_dir, "final_%d.fasta" % len(final))
    with open(fasta_path, "w") as f:
        for c in final:
            f.write(">%s\n%s\n" % (c.get("id", "cand"), c.get("sequence", "")))

    return {"ranked": {"front": front, "final": final, "rejected": len(rejected),
                       "n_in": len(ctx["candidate_features"]), "n_passed": len(kept),
                       "csv": csv_path, "fasta": fasta_path,
                       "diversity_shortfall": max(0, n_final - len(final))}}
