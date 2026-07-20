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
        if c.get("sigma") is None:
            # calibration reports sigma=None when a control had a single replicate. Treating that
            # as 0 would collapse "control +/- k*sigma" onto the control value and make the gate
            # infinitely sharp; guessing a sigma would be worse. Say which control is under-sampled.
            raise ValueError(
                "control '%s' has no measured sigma for metric '%s' (n=%s): the threshold '%s' "
                "cannot be resolved. Raise calibration.robustness.min_structure_pairs, or supply "
                "replicate structures/poses for this control."
                % (name, metric, c.get("n"), s))
        k = float(m.group("k")) * float(c["sigma"])
        return float(c["value"]) + (k if m.group("op") == "+" else -k)
    m2 = re.match(r"^\s*([\d.]+)\s*\*\s*sigma\s*$", s)
    if m2:
        sigmas = [c[metric]["sigma"] for c in controls.values()
                  if metric in c and c[metric].get("sigma") is not None]
        if not sigmas:
            raise ValueError("no control has a measured sigma for metric '%s': the threshold '%s' "
                             "cannot be resolved (every control was single-replicate)" % (metric, s))
        return float(m2.group(1)) * max(sigmas)
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


# Gates that ask "does this protein switch on ITS OWN native effector". The native positive control
# is judged on these ONLY: it is not being retargeted, so demanding that it bind the new target
# well (target_binding), out-compete the target's decoys (specificity), or inherit the target's
# pose confidence would fail a perfectly good scaffold for the wrong reason.
NATIVE_SWITCH_GATES = ("apo_state_bias", "coupling", "ligand_prefers_induced",
                       "apo_dna_competence", "dna_release", "allosteric_template",
                       "fold_clash", "ligand_strain", "packing", "agreement")
# fields a native-switch row is required to carry (the target-specific ones are not its business)
NATIVE_REQUIRED = ["dG_apo", "dG_lig", "ddG_coupling", "E_DNA_X_D", "S_release",
                   "clash_count", "template_similarity", "all_states_packed"]


# Which gate failure means which FUNCTIONAL CATEGORY. A candidate gets a category, not just a
# pass/fail - the failure-mode decomposition is itself the methodological contribution, and it maps
# a reviewer's question ("why did this one fail") onto a mechanism, not a number.
# Order matters: the first matching gate (structural integrity first, then the mechanism gates)
# assigns the category, because a clashing pose makes every downstream energy meaningless.
FAILURE_CATEGORY = [
    ("structural_failure",   ("packing", "fold_clash", "ligand_strain", "pose_confidence",
                              "agreement")),
    ("constitutive_on_risk", ("apo_state_bias",)),
    ("dna_binding_defective", ("apo_dna_competence",)),
    ("binding_only_nonresponder", ("target_binding", "coupling", "ligand_prefers_induced",
                                   "dna_release", "allosteric_template")),
    ("decoy_preferring",     ("specificity",)),
]


def classify_candidate(passed, reasons):
    """-> functional category. 'functional_switch' iff it passed everything; otherwise the category
    of the FIRST gate it failed, in the priority order above (structure before mechanism)."""
    if passed:
        return "functional_switch"
    fired = {r.split(":", 1)[0] for r in reasons}
    for category, gates in FAILURE_CATEGORY:
        if fired & set(gates):
            return category
    return "rejected_other"


def apply_gates(f, cfg, control_metrics, topology_mode="auto", route_direction=None,
                only_gates=None, required=None):
    """-> (passed: bool, reasons: list[str]). Fail closed on any missing feature.

    Every reason STARTS WITH ITS GATE NAME ("apo_state_bias: ..."). check_controls relies on that
    prefix to verify a declared negative control was rejected BY ITS OWN diagnostic gate - a
    constitutive mutant that merely trips the strain gate is a coincidence, not evidence the
    apo-bias gate works.

    control_metrics is {control: {metric: {'value','sigma'}}} - the NUMBERS thresholds resolve
    against, never the name list from config.
    """
    g = cfg["hard_gates"]
    bad = []
    want = set(only_gates) if only_gates else None

    def fires(gate):
        """Is this gate in scope for the object being judged?"""
        return want is None or gate in want

    if cfg.get("fail_closed", True):
        need = required if required is not None else REQUIRED
        missing = [k for k in need if k not in f or f[k] is None]
        if missing:
            return False, ["missing:" + ",".join(missing)]
    if fires("packing") and not f.get("all_states_packed", False):
        bad.append("packing: a state failed to pack")

    T = lambda e, metric: resolve_threshold(e, control_metrics, metric)

    if fires("target_binding") and f["E_L_I"] > T(g["target_binding"]["max"], "E_L_I"):
        bad.append("target_binding: target does not bind (E_L_I above WT native effector)")
    if fires("apo_state_bias") and f["dG_apo"] < T(g["apo_state_bias"]["min"], "dG_apo"):
        bad.append("apo_state_bias: constitutive - apo already prefers the induced state")
    if fires("coupling") and f["ddG_coupling"] > T(g["coupling"]["max"], "ddG_coupling"):
        bad.append("coupling: ligand does not shift the state population (weak linkage)")
    if fires("ligand_prefers_induced") and "ligand_prefers_induced" in g and \
            f["dG_lig"] > T(g["ligand_prefers_induced"]["max"], "dG_lig"):
        bad.append("ligand_prefers_induced: holo protein still prefers D - ddG_coupling was bought "
                   "by apo bias, not a switch")
    if fires("apo_dna_competence") and f["E_DNA_X_D"] > T(g["apo_dna_competence"]["max"], "E_DNA_X_D"):
        bad.append("apo_dna_competence: apo can no longer hold the operator")

    s = release_sign(topology_mode, route_direction)
    if fires("dna_release") and s * f["S_release"] < T(g["dna_release"]["min"], "S_release"):
        bad.append("dna_release: induced state does not release DNA (sign-corrected)")

    if fires("specificity") and f["S_specificity"] < T(g["specificity"]["min"], "S_specificity"):
        bad.append("specificity: not selective against decoys")
    # No host-metabolite gate on purpose. A charged metabolite's Rosetta energy is not comparable
    # to a hydrophobic effector's, so the difference cannot carry a pass/fail decision, and a
    # docking failure means the pose method does not apply rather than that the molecule is
    # excluded. Cellular mis-activation is measured where it is measurable: basal leak in the
    # fluorescence assay. metabolite_margin is carried in the report for inspection only.
    if fires("fold_clash") and f["clash_count"] > T(g["state_integrity"]["fold_clash"]["max"], "clash_count"):
        bad.append("fold_clash: fold clashes above WT")
    # dict.get(key, default) returns None when the key EXISTS and holds None, so an explicitly
    # unassessed value slips past the default and reaches the comparison. Unassessed is not a pass:
    # it is missing evidence, and it fails closed.
    if fires("ligand_strain"):
        strain = f.get("ligand_strain")
        if strain is None or strain > g["state_integrity"]["ligand_strain"]["max"]:
            bad.append("ligand_strain: ligand strained or strain not assessed")
    # S_state (interface) and ddG_coup (double difference) are the same claim measured two ways.
    # ligand_score.consistency() flags when they disagree; a disagreement means the pose moved
    # during packing, so the candidate is rejected rather than averaged (GPT-5.6 caught this being
    # computed and then ignored). Only enforced when the field is present.
    if fires("agreement") and "agree" in f and f["agree"] is False:
        bad.append("agreement: S_state and ddG_coupling disagree - unstable pose, not a "
                   "trustworthy switch")
    if fires("pose_confidence"):
        pc = f.get("pose_confidence")
        if pc is None or pc < g["state_integrity"]["pose_confidence"]["min"]:
            bad.append("pose_confidence: low or unassessed pose confidence")
    if fires("allosteric_template") and f["template_similarity"] < T(g["allosteric_template"]["min"], "template_similarity"):
        bad.append("allosteric_template: allosteric path broken (below the WT template scale)")

    return (len(bad) == 0), bad


def check_controls(control_scores, cfg, control_metrics, topology_mode="auto",
                   route_direction=None):
    """Calibrate the scaffold on its OWN controls before any design is generated.

    Negative controls are OPTIONAL IN AVAILABILITY, MANDATORY IN BEHAVIOUR WHEN DECLARED:

        mandatory (wt_apo, wt_native_holo) missing        -> STOP
        WT native system does not reproduce a switch      -> STOP
        an optional negative control is absent            -> record it, continue at a lower
                                                             validation level
        a declared negative control is NOT caught by its
        own diagnostic gate                               -> STOP

    Requiring every scaffold to have known constitutive / non-binding / dead mutants would exclude
    most designable TFs; letting a scaffold skip calibration would leave its six-state energies
    with no internal reference. Hence WT is mandatory and negatives are conditional.

    Expectations come from cfg['calibration'], never from a hardcoded table: which mutants exist
    is a property of the scaffold's literature, not of this module.
    -> (ok, report) with report['validation_level'].
    """
    cal = cfg.get("calibration", {})
    mandatory = list(cal.get("mandatory_controls", ["wt_apo", "wt_native_holo"]))
    optional = dict(cal.get("optional_validation_controls", {}))

    missing_mandatory = [n for n in mandatory if n not in control_scores]
    if missing_mandatory:
        return False, {"missing_mandatory_controls": missing_mandatory,
                       "verdict": "scaffold rejected: WT calibration is incomplete - without it "
                                  "the six-state energies have no internal reference"}

    report = {"mandatory": {}, "optional": {}, "optional_missing": []}

    # The scaffold must reproduce its own native switch - and ONLY that. wt_native_holo is scored
    # with the NATIVE effector, so judging it on target_binding, specificity or the target's pose
    # confidence would fail a perfectly good scaffold for not being pre-adapted to a molecule it
    # has never seen. Those questions belong to the candidates.
    passed, reasons = apply_gates(control_scores["wt_native_holo"], cfg, control_metrics,
                                  topology_mode, route_direction,
                                  only_gates=NATIVE_SWITCH_GATES, required=NATIVE_REQUIRED)
    report["mandatory"]["wt_native_holo"] = {"passed": passed, "reasons": reasons,
                                             "gates_applied": list(NATIVE_SWITCH_GATES)}
    if not passed:
        report["verdict"] = ("scaffold rejected: its native ligand + WT system does not reproduce "
                             "a functional switch, so the gates cannot be trusted here")
        return False, report
    # wt_apo is a metric baseline, not a holo sensor - it is not run through the gates
    report["mandatory"]["wt_apo"] = {"available": True,
                                     "metrics": sorted(control_scores["wt_apo"])}

    strict = cal.get("require_all_declared_optional", True)
    passed_optional = []
    for name, spec in optional.items():
        if name not in control_scores:
            report["optional_missing"].append(name)
            continue
        ok_gates, why = apply_gates(control_scores[name], cfg, control_metrics,
                                    topology_mode, route_direction)
        expect_gate = spec.get("expected_failure_gate")
        expect_any = spec.get("expected_failure_any", [])
        # reasons are prefixed with their gate name, so this checks the RIGHT gate fired
        if expect_gate:
            right = any(r.startswith(expect_gate + ":") for r in why)
        else:
            right = any(r.startswith(g + ":") for g in expect_any for r in why)
        valid = (not ok_gates) and right
        report["optional"][name] = {"passed_gates": ok_gates, "rejected_as_expected": valid,
                                    "expected_gate": expect_gate or expect_any, "reasons": why}
        if not valid and strict:
            report["verdict"] = ("scaffold rejected: declared control %s was not rejected by its "
                                 "expected mechanism gate" % name)
            return False, report
        if valid:
            passed_optional.append(name)

    report["validation_level"] = ("native_plus_negative_controls" if passed_optional
                                  else "native_controls_only")
    report["mandatory_controls_passed"] = mandatory
    report["optional_controls_passed"] = passed_optional
    report["optional_controls_missing"] = report["optional_missing"]
    report["verdict"] = "calibration passed"
    return True, report


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


# integrity terms rank gates on that MUST come from upstream, with who owes them. rank defaults a
# missing ligand_strain to +inf and a missing pose_confidence to -1, so an absent value does not
# weaken the ranking - it rejects every candidate while still looking like a strict filter. Better
# to say which stage did not compute it.
REQUIRED_UPSTREAM = {
    "clash_count":         "state_builder: fold clashes in the packed candidate",
    "template_similarity": "allostery/state_builder: candidate vs the scaffold's native template",
    "ligand_strain":       "pose/state_builder: internal strain of the placed ligand",
    "pose_confidence":     "pose: confidence of the target pose used for the liganded states",
}


def assemble_candidate_features(ctx):
    """The ONE place candidate feature rows are built, from what upstream actually produced.

    Keys here are exactly the names the gates and the Pareto objectives use (E_L_I, ddG_coupling,
    S_release, S_specificity, template_similarity) - a concept name would silently drop an
    objective instead of failing.
    """
    need = ("candidates", "candidate_states", "ligand_scores", "dna_scores", "specificity_scores")
    absent_ctx = [k for k in need if not ctx.get(k)]
    if absent_ctx:
        raise RuntimeError("cannot assemble candidate features: upstream produced no %s"
                           % ", ".join(absent_ctx))

    common = (set(ctx["candidates"]) & set(ctx["candidate_states"]) & set(ctx["ligand_scores"])
              & set(ctx["dna_scores"]) & set(ctx["specificity_scores"]))
    if not common:
        raise RuntimeError(
            "no candidate survived every upstream stage (candidates=%d states=%d ligand=%d "
            "dna=%d specificity=%d): nothing to rank"
            % (len(ctx["candidates"]), len(ctx["candidate_states"]), len(ctx["ligand_scores"]),
               len(ctx["dna_scores"]), len(ctx["specificity_scores"])))

    rows = []
    for cid in sorted(common):
        cand, state = ctx["candidates"][cid], ctx["candidate_states"][cid]
        ligand, dna, spec = (ctx["ligand_scores"][cid], ctx["dna_scores"][cid],
                             ctx["specificity_scores"][cid])
        missing = [k for k in REQUIRED_UPSTREAM if state.get(k) is None]
        if missing:
            raise RuntimeError(
                "candidate %s is missing upstream integrity terms %s - these must be COMPUTED, "
                "never defaulted:\n  %s"
                % (cid, missing, "\n  ".join("%s <- %s" % (k, REQUIRED_UPSTREAM[k])
                                             for k in missing)))
        rows.append({
            "id": cid,
            "sequence": cand.get("full_sequence") or cand.get("sequence", ""),
            "E_L_I": ligand["E_L_I"],
            "dG_apo": ligand.get("dG_apo", state.get("dG_apo")),
            "dG_lig": ligand.get("dG_lig", state.get("dG_lig")),
            "ddG_coupling": ligand.get("ddG_coupling", state.get("ddG_coupling")),
            "agree": ligand.get("agree"),
            "E_DNA_X_D": dna["E_DNA_X_D"],
            "S_release": dna["S_release"],
            "S_specificity": spec.get("specificity"),
            # None when no metabolite could be posed: the gate skips it rather than failing the
            # candidate, but a posed metabolite that beats the target is caught
            "metabolite_margin": spec.get("metabolite_margin"),
            "clash_count": state["clash_count"],
            "template_similarity": state["template_similarity"],
            "ligand_strain": state["ligand_strain"],
            "pose_confidence": state["pose_confidence"],
            "all_states_packed": state.get("all_states_packed", False),
        })
    return rows


def run(ctx):
    """requires ctx['candidates','candidate_states','ligand_scores','dna_scores',
       'specificity_scores','cfg_scoring'];  produces ctx['ranked']"""
    import csv
    import os

    cfg = ctx["cfg_scoring"]
    # assembled here rather than by a separate stage: one owner, one field vocabulary
    features = ctx.get("candidate_features") or assemble_candidate_features(ctx)

    # THREE DISTINCT THINGS - never substitute one for another:
    #   cfg['calibration']    the SPEC: which controls are mandatory, which gate each declared
    #                         negative must trip
    #   ctx['control_metrics'] the NUMBERS: {control: {metric: {'value','sigma',...}}}, what
    #                         thresholds like "wt_apo + 1.0*sigma" resolve against
    #   ctx['control_scores'] full feature ROWS per control, to actually run the gates on
    # Falling back to cfg['calibration']['controls'] would hand resolve_threshold a list of NAMES
    # where it expects per-metric values, and every threshold would silently fail to resolve.
    control_metrics = ctx.get("control_metrics") or {}
    if not control_metrics:
        raise RuntimeError(
            "no control_metrics: every gate is expressed relative to this scaffold's own controls "
            "(e.g. 'wt_apo + 1.0*sigma') and cannot resolve without their per-metric values. "
            "config lists control NAMES, not numbers - it is not a substitute.")
    topo = cfg.get("topology", {}).get("mode", "auto")
    direction = (ctx.get("route") or {}).get("direction")

    cs = ctx.get("control_scores")
    validation_level = None
    if not cs:
        # calibration is not optional: WT is what the six-state energies are relative to.
        raise RuntimeError(
            "no control_scores: this scaffold's own WT controls (wt_apo, wt_native_holo) must be "
            "scored before ranking. Negative controls may be absent - WT may not.")
    ok, report = check_controls(cs, cfg, control_metrics, topo, direction)
    ctx["control_report"] = report
    if not ok:
        raise RuntimeError("scaffold failed calibration: %s" % report.get("verdict", report))
    # evidence level, not performance: it is recorded and reported, never fed to the Pareto score
    validation_level = report.get("validation_level")

    kept, rejected = [], []
    category_counts = {}
    for c in features:
        passed, reasons = apply_gates(c, cfg, control_metrics, topo, direction)
        category = classify_candidate(passed, reasons)
        category_counts[category] = category_counts.get(category, 0) + 1
        rec = dict(c, _reasons=reasons, functional_category=category)
        (kept if passed else rejected).append(rec)

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
    # validation_level travels WITH every ranked row: a reader must see on what evidence this
    # scaffold was calibrated. It is not an objective and never enters the Pareto score.
    cols = ["id", "sequence", "functional_category"] + REQUIRED + ["validation_level", "_reasons"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for c in front:
            row = {k: c.get(k) for k in cols}
            row["validation_level"] = validation_level
            wr.writerow(row)
    fasta_path = os.path.join(out_dir, "final_%d.fasta" % len(final))
    with open(fasta_path, "w") as f:
        for c in final:
            f.write(">%s\n%s\n" % (c.get("id", "cand"), c.get("sequence", "")))

    # failure-mode decomposition of the whole pool: every candidate has a category, not just the
    # survivors. A deliverable, not a log line.
    import json as _json
    by_cat = {}
    for r in rejected:
        by_cat.setdefault(r.get("functional_category"), []).append(r.get("id"))
    fm_path = os.path.join(out_dir, "failure_modes.json")
    with open(fm_path, "w") as f:
        _json.dump({"n_total": len(features), "n_functional_switch": len(kept),
                    "category_counts": category_counts, "rejected_by_category": by_cat},
                   f, indent=2)

    return {"ranked": {"front": front, "final": final, "rejected": len(rejected),
                       "n_in": len(features), "n_passed": len(kept),
                       "csv": csv_path, "fasta": fasta_path, "failure_modes": fm_path,
                       "category_counts": category_counts,
                       "diversity_shortfall": max(0, n_final - len(final)),
                       # evidence level of the calibration behind this ranking
                       "validation_level": validation_level,
                       "mandatory_controls_passed": report.get("mandatory_controls_passed", []),
                       "optional_controls_passed": report.get("optional_controls_passed", []),
                       "optional_controls_missing": report.get("optional_controls_missing", [])}}
