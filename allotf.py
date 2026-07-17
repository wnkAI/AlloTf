"""AlloTF-RL - target molecule in, functional aTF sensors out via dose-time fluorescence feedback.

Three commands run one project across the design -> assay -> select loop:

  design       target -> PhysAllo six-state physics design -> ranked -> initial-N plasmids,
               a plate layout, and the mechanism/mutation features the GP will condition on
  feedback     import a dose-time fluorescence plate CSV and update the project
  select-next  refit the hierarchical GP and Thompson-select the next batch

No LigandMPNN in production, no general allostery model, no MD. Every physics number is a relative
proxy on one scaffold; ranking a functional sensor is what the fluorescence feedback is for.
"""
import os
import re
import csv
import sys
import json
import time
import hashlib
import argparse
import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))

# name, owner, description, requires, produces
STAGES = [
    ("route",         "A", "target molecule -> native ligand + scaffold + design mode",
     [], ["route", "scaffold"]),
    ("structure",     "B", "prepare & QC native states (X_D, X_I, X_D_DNA) + ligand pose",
     ["scaffold"], ["states", "residue_mapping"]),
    ("allostery",     "B", "system-specific allosteric template + design masks",
     ["states"], ["template", "masks"]),
    ("pose",          "C", "target ligand poses on both backbones",
     ["states", "masks"], ["poses"]),
    ("design",        "C", "PhysAllo physics-grounded pocket design (aa_filter + rotamer search)",
     ["states", "poses", "masks"], ["candidates"]),
    ("state_builder", "D", "six PyRosetta states per candidate (D0/I0/DL/IL/D_DNA/I_DNA)",
     ["candidates", "states", "poses"], ["candidate_states"]),
    ("ligand_score",  "D", "dG_lig, dG_apo, ddG_coupling",
     ["candidate_states"], ["ligand_scores"]),
    ("dna_release",   "D", "S_release as a double difference, sign from topology",
     ["candidate_states"], ["dna_scores"]),
    ("specificity",   "D", "negative design vs native / analogues / metabolites",
     ["candidate_states"], ["specificity_scores"]),
    ("rank",          "E", "control calibration -> hard gates -> Pareto -> diversity",
     ["ligand_scores", "dna_scores", "specificity_scores", "cfg_scoring"], ["ranked"]),
]
MODE_MAP = {"auto": None, "enhance": "ENHANCEMENT", "retarget": "RETARGETING"}

# the frozen plate grid (relative to a per-target reference concentration Cref)
DOSE_FRACTIONS = [0.0, 0.1, 0.3, 1.0, 3.0]
TIME_POINTS = [0, 6, 12, 18, 24, 30, 36]
REPLICATES = 2


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def project_id(target, scaffold):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", target).strip("-")[:24] or "target"
    h = hashlib.sha1(target.encode()).hexdigest()[:6]
    return "%s-%s_%s" % (slug, h, scaffold)


# ---- design stage plumbing ------------------------------------------------------------------

def run_stage(name, ctx, requires, produces):
    missing = [k for k in requires if k not in ctx or ctx[k] is None]
    if missing:
        raise RuntimeError("stage '%s' requires %s which upstream never produced" % (name, missing))
    mod = __import__("pipeline." + name, fromlist=["*"])
    fn = getattr(mod, "run", None)
    if fn is None:
        raise NotImplementedError("pipeline/%s.py has no run(ctx) entry point" % name)
    out = fn(ctx)
    if out is None:
        raise RuntimeError("stage '%s' returned None; it must return the updated ctx" % name)
    absent = [k for k in produces if k not in out or out[k] is None]
    if absent:
        raise RuntimeError("stage '%s' did not produce %s" % (name, absent))
    return out


def mechanism_features(cid, cand, cstate, masks):
    """Six-state physics + mutation summary -> the GP's context vector for this candidate.

    Mechanism terms come straight from the double differences state_builder computed (they are
    already relative quantities on this scaffold). Mutation terms use FUNCTIONAL position counts
    from the masks, not absolute residue numbers, so the same feature means the same thing on a
    different scaffold.
    """
    mech = {
        "ddG_coupling": cstate.get("ddG_coupling"),
        "dna_release": cstate.get("S_release"),
        "d_apo_dna_affinity": cstate.get("E_DNA_X_D"),
        "d_target_binding": (cstate.get("interface") or {}).get("IL"),
    }
    mech = {k: (0.0 if v is None else float(v)) for k, v in mech.items()}
    rec = set(masks.get("recognition_mask", []))
    trans = set(masks.get("transduction_mask", []))
    muts = cand.get("mutations", [])
    mut = {
        "n_mut": len(muts),
        "n_recognition": sum(1 for m in muts if m[0] in rec),
        "n_transduction": sum(1 for m in muts if m[0] in trans),
    }
    return mech, mut


def finalize_design(ctx, project, n_initial):
    """After rank: write initial-N plasmids, the plate layout, the feature table, and open the
    project so feedback / select-next can continue it."""
    from pipeline.ai import closed_loop as cl

    ranked = ctx["ranked"]
    order = ranked.get("final") or ranked.get("front") or []
    picked = order[:n_initial]
    cand_states = ctx.get("candidate_states", {})
    masks = ctx.get("masks", {})
    scaffold = ctx["scaffold"]
    cref = float(ctx.get("cfg", {}).get("design", {}).get("cref", 1.0))
    conc = [f * cref for f in DOSE_FRACTIONS]

    candidates = {}
    feat_rows = []
    for rec in picked:
        cid = rec.get("id") or rec.get("candidate_id")
        cand = ctx["candidates"].get(cid, rec)
        mech, mut = mechanism_features(cid, cand, cand_states.get(cid, rec), masks)
        candidates[cid] = {"sequence": cand.get("sequence", rec.get("sequence", "")),
                           "mech": mech, "mut": mut, "scaffold": scaffold}
        feat_rows.append(dict(candidate_id=cid, sequence=candidates[cid]["sequence"],
                              **{("mech_" + k): v for k, v in mech.items()},
                              **{("mut_" + k): v for k, v in mut.items()}))

    if not candidates:
        raise RuntimeError("rank produced no candidate to seed the project")

    cl.init_project(project, candidates, conc, TIME_POINTS,
                    basal_max=ctx.get("cfg_scoring", {}).get("basal_max"))
    _write_fasta(os.path.join(project, "initial_%d.fasta" % len(candidates)), candidates)
    _write_plate_layout(os.path.join(project, "initial_%d_plate_layout.csv" % len(candidates)),
                        list(candidates), conc)
    _write_features(os.path.join(project, "initial_%d_features.csv" % len(candidates)), feat_rows)
    print("\ndesign done -> %d plasmids, plate layout and features in %s" % (len(candidates), project))
    return 0


def _write_fasta(path, candidates):
    with open(path, "w") as f:
        for cid, c in candidates.items():
            f.write(">%s\n%s\n" % (cid, c["sequence"]))


def _write_plate_layout(path, cids, conc):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "concentration", "time_h", "replicate", "well"])
        well = 0
        for cid in cids:
            for c in conc:
                for t in TIME_POINTS:
                    for rep in range(1, REPLICATES + 1):
                        w.writerow([cid, c, t, rep, "W%04d" % well])
                        well += 1


def _write_features(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_predictions(path, res, state):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "front_probability", "pred_max_fold", "pred_EC50",
                    "pred_t_on_bin", "pred_auc", "pred_basal", "pred_uncertainty", "selected"])
        chosen = set(res.get("selected", []))
        for cid, prob in sorted(res.get("prob", {}).items(), key=lambda kv: -kv[1]):
            ph = res.get("phenotypes", {}).get(cid, {})
            w.writerow([cid, round(prob, 4), _r(ph.get("max_fold")), _r(ph.get("EC50")),
                        ph.get("t_on_bin"), _r(ph.get("auc")), _r(ph.get("basal")),
                        _r(ph.get("pred_uncertainty")), cid in chosen])


def _r(x):
    return round(x, 3) if isinstance(x, (int, float)) else x


# ---- commands -------------------------------------------------------------------------------

def cmd_design(a):
    cfg = load_yaml(a.config)
    cfg_scoring = load_yaml(a.scoring)
    cfg.setdefault("design", {})["initial_designs"] = a.initial_designs

    from pipeline.route import route as run_route
    r = run_route(a.target, max(a.top_scaffolds, 10))
    print("=" * 78)
    print("AlloTF-RL design   target=%s   mode=%s   %s"
          % (a.target, r["mode"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("=" * 78)

    designable = [h for h in r["hits"] if h["designable"]]
    sel = ([h for h in r["hits"] if h["tf"] == a.scaffold] if a.scaffold
           else designable[:1])
    if not sel:
        print("STOP: no designable scaffold for this target.")
        return 1
    hit = sel[0]
    scaffold = hit["tf"]
    project = a.project or os.path.join(ROOT, "projects", project_id(a.target, scaffold))
    os.makedirs(project, exist_ok=True)

    ctx = dict(cfg=cfg, cfg_scoring=cfg_scoring, out_dir=project, scaffold=scaffold, route=r,
               target=a.target, mode=r["mode"], hit=hit, seed=a.seed, started=time.time())
    for name, owner, desc, req, prod in STAGES[1:]:
        print("\n[%s] (%s) %s" % (name, owner, desc))
        try:
            ctx = run_stage(name, ctx, req, prod)
        except NotImplementedError as e:
            print("  NOT IMPLEMENTED -> %s" % e)
            return 2
        except Exception:
            print("  FAILED:\n" + traceback.format_exc())
            return 3
    return finalize_design(ctx, project, a.initial_designs)


def cmd_feedback(a):
    from pipeline.ai import closed_loop as cl
    info = cl.ingest_plate(a.project, a.plate)
    print("feedback: +%d observations (background %.2f), %d total"
          % (info["added"], info["background"], info["total_observations"]))
    if info["skipped"]:
        print("  skipped ids not in project (e.g. WT/controls): %s" % info["skipped"])
    return 0


def cmd_select_next(a):
    from pipeline.ai import closed_loop as cl
    res = cl.select_next(a.project, a.n)
    if not res.get("selected"):
        print(res.get("note", "nothing to select"))
        return 0
    state = cl.load(a.project)
    picked = {cid: state["candidates"][cid] for cid in res["selected"]}
    conc = state["grid"]["conc"]
    tag = os.path.join(a.project, "next_%d" % len(picked))
    _write_fasta(tag + ".fasta", picked)
    _write_plate_layout(tag + "_plate_layout.csv", list(picked), conc)
    _write_predictions(os.path.join(a.project, "posterior_predictions.csv"), res, state)
    print("select-next: chose %s from %d untested" % (res["selected"], res.get("n_untested", 0)))
    print("  -> %s.fasta, %s_plate_layout.csv, posterior_predictions.csv" % (tag, tag))
    return 0


def main():
    ap = argparse.ArgumentParser(description="AlloTF-RL")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("design", help="target -> initial plasmids + plate layout")
    d.add_argument("--target", required=True, help="molecule name or SMILES (or .sdf path)")
    d.add_argument("--initial-designs", type=int, default=8)
    d.add_argument("--top-scaffolds", type=int, default=1)
    d.add_argument("--scaffold", default=None)
    d.add_argument("--project", default=None)
    d.add_argument("--config", default=os.path.join(ROOT, "config", "default.yaml"))
    d.add_argument("--scoring", default=os.path.join(ROOT, "config", "scoring.yaml"))
    d.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("feedback", help="import a dose-time fluorescence plate")
    f.add_argument("--project", required=True)
    f.add_argument("--plate", required=True, help="CSV: candidate_id,concentration,time_h,fluorescence,replicate")

    s = sub.add_parser("select-next", help="Thompson-select the next batch")
    s.add_argument("--project", required=True)
    s.add_argument("--n", type=int, default=4)

    a = ap.parse_args()
    return {"design": cmd_design, "feedback": cmd_feedback,
            "select-next": cmd_select_next}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
