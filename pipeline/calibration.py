"""Score this scaffold's OWN controls, so every gate has something to be relative to.

Absolute Rosetta energies are not comparable across structures, operators or preparations, so every
threshold in scoring.yaml is written as "wt_apo + 1.0*sigma" and needs two things this stage
produces:

    control_scores   {control: feature row}   - full rows, to run the gates on
    control_metrics  {control: {metric: {value, sigma, n, ...}}} - the NUMBERS thresholds resolve
                                                                  against

WT is mandatory (wt_apo, wt_native_holo). Declared negative controls are scored when the scaffold
config names their mutations; absent ones are simply not produced, and rank records the resulting
validation level. That policy lives in rank.check_controls - this stage only measures.

sigma is not decoration. It is the spread over independent repeats: structure pairs, poses and
protonation states, exactly the robustness axes scoring.yaml declares. A sigma taken from one
calculation would make every threshold a hair's breadth from its control value.
"""
import os

import numpy as np

from .state_builder import build_six, totals, linkage, dna_release, dna_affinity

# metrics each control contributes, and where they come from in a scored six-state record
METRIC_SOURCES = {
    "dG_apo": lambda r: r.get("dG_apo"),
    "dG_lig": lambda r: r.get("dG_lig"),
    "ddG_coupling": lambda r: r.get("ddG_coupling"),
    "S_release": lambda r: r.get("S_release"),
    "E_DNA_X_D": lambda r: r.get("E_DNA_X_D"),
    "E_L_I": lambda r: (r.get("interface") or {}).get("IL"),
    "clash_count": lambda r: r.get("clash_count"),
    "template_similarity": lambda r: r.get("template_similarity"),
    "ligand_strain": lambda r: r.get("ligand_strain"),
    # the specificity gate is written relative to the controls too, so a control must contribute it
    "S_specificity": lambda r: r.get("S_specificity"),
}


def _replicate_templates(paths, n_structure_pairs):
    """Independent X_D/X_I pairs to repeat the calculation over.

    Only the pairs that actually exist are returned; the caller decides whether that meets the
    configured minimum. Inventing a second pair by perturbing the first would give a sigma that
    reports numerical noise rather than structural uncertainty.
    """
    reps = [paths]
    alts = paths.get("alt_pairs") or []
    for alt in alts[: max(n_structure_pairs - 1, 0)]:
        merged = dict(paths)
        merged.update(alt)
        reps.append(merged)
    return reps


def _score_one(backend, residues, templates, design_positions, second_shell, chain,
               symmetric_chains, template_ctx, specificity_fn=None, pose_conf=None):
    """One control, one replicate -> a feature row shaped EXACTLY like a candidate's.

    "Exactly" matters: check_controls runs these rows through apply_gates, which fail-closes on
    rank.REQUIRED. A row missing S_specificity or pose_confidence is rejected for being incomplete,
    so the scaffold would fail its own calibration for a plumbing reason and never reach design.
    """
    terms = build_six(backend, residues, templates, design_positions, second_shell, chain,
                      symmetric_chains=symmetric_chains)
    tot = totals(terms)
    if any(v is None for v in tot.values()):
        return None
    link = linkage(tot)
    row = {
        "all_states_packed": True,
        "dG_apo": link["dG_apo"] if link else None,
        "dG_lig": link["dG_lig"] if link else None,
        "ddG_coupling": link["ddG_coup"] if link else None,
        "S_release": dna_release(tot, +1),
        "E_DNA_X_D": dna_affinity(tot, "D"),
        "interface": {st: backend.interface_energy(terms[st]["_pose"])
                      if terms.get(st) and terms[st].get("_pose") is not None else None
                      for st in ("DL", "IL")},
    }
    row["E_L_I"] = row["interface"].get("IL")
    clash = {st: backend.clash_count(terms[st]["_pose"]) for st in terms
             if terms.get(st) and terms[st].get("_pose") is not None}
    row["clash_count"] = max(clash.values()) if clash else None
    strain = {}
    for st in ("DL", "IL"):
        p = (terms.get(st) or {}).get("_pose")
        if p is not None:
            v = backend.ligand_strain(p)
            if v is not None:
                strain[st] = v
    row["ligand_strain"] = max(strain.values()) if strain else None
    il = (terms.get("IL") or {}).get("_pose")
    row["template_similarity"] = (
        template_ctx(backend, il) if il is not None and template_ctx else None)
    # the gates fail-close on rank.REQUIRED, so a control row must carry the same fields a
    # candidate does - otherwise the scaffold fails its own calibration for a plumbing reason
    row["S_specificity"] = specificity_fn(residues, templates) if specificity_fn else None
    row["pose_confidence"] = pose_conf
    return row


def run(ctx):
    """requires ctx['states','poses','ligand_params','template','masks','backend']
       produces ctx['control_metrics'], ctx['control_scores']"""
    backend = ctx.get("backend")
    if backend is None:
        raise RuntimeError("calibration needs the same backend the candidates were scored with: "
                           "controls measured with a different scorefunction or ligand params are "
                           "not a reference for anything")

    cfg_scoring = ctx.get("cfg_scoring") or {}
    cal = cfg_scoring.get("calibration", {})
    robust = cal.get("robustness", {})
    states = ctx["states"]
    templates = states["paths"]
    design_positions = ctx["masks"]["recognition_mask"]
    second_shell = ctx["masks"].get("transduction_mask", ())
    chain = states.get("chain", "A")
    protein_chains = states.get("protein_chains")

    # the native transduction network candidates are compared against - same definition
    # state_builder uses, so control and candidate template_similarity are the same quantity
    tpl = ctx.get("template") or {}
    path_resnums = sorted(int(rn) for rn, r in (tpl.get("residues") or {}).items()
                          if r.get("class") in ("transduction", "output"))
    native_contacts = None
    if path_resnums:
        native_contacts = backend.contact_pairs(backend.prepare_pose(templates["X_I"]),
                                                path_resnums, chain=chain)

    def template_ctx(be, pose):
        from .state_builder import _template_similarity
        return _template_similarity(be, pose, native_contacts, path_resnums, chain)

    # WT's own selectivity, computed the SAME way candidates' is: the specificity gate is expressed
    # relative to the controls, so without this the threshold has nothing to resolve against.
    from .specificity import _dock_and_score, default_decoys, decoy_tier
    rt = ctx.get("route") or {}
    lp = ctx.get("ligand_params") or {}
    spec_method = (ctx.get("cfg", {}).get("design") or {}).get("specificity_method", "smina")
    bundle_decoys = lp.get("decoys") or {}
    target_smiles = rt.get("target_smiles") or rt.get("smiles")
    target_resname = (lp.get("target") or {}).get("resname", "TGT")

    def specificity_fn(residues, ctrl_templates):
        x_i = ctrl_templates.get("X_I")
        e_t = _dock_and_score(backend, target_smiles, x_i, design_positions, target_resname,
                              residues, design_positions, second_shell, chain, protein_chains,
                              spec_method, ctx.get("cfg", {}).get("design") or {})
        if e_t is None:
            return None
        best = None
        for did, d in bundle_decoys.items():
            name = d.get("decoy_name") or d.get("role") or did
            if decoy_tier(name) != "mandatory":
                continue
            e = _dock_and_score(backend, d.get("smiles"), x_i, design_positions, d.get("resname"),
                                residues, design_positions, second_shell, chain, protein_chains,
                                spec_method, ctx.get("cfg", {}).get("design") or {})
            if e is not None and (best is None or e < best):
                best = e
        return (best - e_t) if best is not None else None

    pose_conf = (ctx.get("poses") or {}).get("confidence")

    # WT residues at the design positions, read from the structure itself
    wt = {}
    from .design import _wt_chain_residues
    for rn, name in _wt_chain_residues(templates["X_I"], chain):
        if rn in set(design_positions):
            wt[rn] = name
    missing = sorted(set(design_positions) - set(wt))
    if missing:
        raise RuntimeError("WT residues absent at design positions %s: cannot build the WT "
                           "controls" % missing)

    # wt_apo and wt_native_holo are the SAME SEQUENCE but must NOT be the same calculation.
    # wt_native_holo is the scaffold's own working switch and has to be scored with its NATIVE
    # effector (the crystal pose kept as X_I_lig_native), not with the design target. Scoring both
    # on the target would put target-ligand numbers into thresholds labelled "native holo", and
    # every gate anchored on wt_native_holo would then be calibrated against the wrong molecule.
    native_templates = dict(templates)
    native_lig = templates.get("X_I_lig_native")
    if native_lig:
        native_templates["X_I_lig"] = native_lig
        # the native effector's DL state has no crystal either; without one the native-holo control
        # cannot be built as a full six-state record
        native_templates["X_D_lig"] = templates.get("X_D_lig_native") or native_lig
    else:
        raise RuntimeError(
            "no X_I_lig_native: wt_native_holo must be scored with the scaffold's NATIVE effector. "
            "pose.run keeps the native crystal under that key - run it before calibration.")

    controls_spec = {
        "wt_apo": (wt, templates),                    # target-liganded states; apo metrics used
        "wt_native_holo": (wt, native_templates),     # NATIVE effector - the real positive control
    }
    # declared negative controls, if the scaffold config names their mutations
    declared = ctx.get("scaffold_controls") or (ctx.get("states") or {}).get("controls") or {}
    for name, muts in declared.items():
        row = dict(wt)
        row.update({int(k): v for k, v in muts.items()})
        controls_spec[name] = (row, native_templates)

    n_pairs = int(robust.get("min_structure_pairs", 1) or 1)
    replicates = _replicate_templates(templates, n_pairs)

    control_scores, control_metrics, per_control_reps = {}, {}, {}
    for cname, (residues, ctrl_templates) in controls_spec.items():
        rows = []
        for rep_templates in _replicate_templates(ctrl_templates, n_pairs):
            try:
                r = _score_one(backend, residues, rep_templates, design_positions, second_shell,
                               chain, protein_chains, template_ctx,
                               specificity_fn=specificity_fn, pose_conf=pose_conf)
            except Exception as exc:
                raise RuntimeError("control %s could not be scored: %s" % (cname, exc))
            if r is not None:
                rows.append(r)
        if not rows:
            raise RuntimeError("control %s produced no complete six-state record" % cname)
        control_scores[cname] = rows[0]
        per_control_reps[cname] = len(rows)

        metrics = {}
        for metric, get in METRIC_SOURCES.items():
            vals = [get(r) for r in rows]
            vals = [float(v) for v in vals if v is not None]
            if not vals:
                continue
            # with a single replicate sigma is unmeasured, not zero: a zero sigma collapses every
            # "control +/- k*sigma" threshold onto the control value itself
            sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            metrics[metric] = {"value": float(np.mean(vals)), "sigma": sigma, "n": len(vals)}
        control_metrics[cname] = metrics

    achieved = {"structure_pairs": len(replicates),
                "poses": len((ctx.get("poses") or {}).get("X_I") or []),
                "protonation_states": 1}
    required = {"structure_pairs": int(robust.get("min_structure_pairs", 1) or 1),
                "poses": int(robust.get("min_poses", 1) or 1),
                "protonation_states": int(robust.get("min_protonation_states", 1) or 1)}
    shortfalls = {k: (achieved[k], required[k]) for k in required if achieved[k] < required[k]}

    # a sigma from one replicate is not a spread. Say so loudly rather than letting every threshold
    # silently degenerate to its control value.
    unmeasured = sorted({m for c in control_metrics.values() for m, v in c.items()
                         if v.get("sigma") is None})
    report = {"replicates": per_control_reps, "robustness_achieved": achieved,
              "robustness_required": required, "robustness_shortfall": shortfalls,
              "metrics_without_sigma": unmeasured}
    if shortfalls:
        report["warning"] = ("robustness below the configured minimum %s: thresholds derived from "
                             "these controls carry that uncertainty" % shortfalls)

    out_dir = ctx.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"control_metrics": control_metrics, "report": report}, f, indent=2, default=str)

    return {"control_metrics": control_metrics, "control_scores": control_scores,
            "calibration_report": report}
