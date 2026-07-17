"""AlloTF V1 - target molecule in, ranked customised aTF designs out.

  python allotf.py --target xylitol --objective retarget --top-scaffolds 3 \
                   --raw-designs 10000 --final-designs 96

V1 scope: enrich designs where the target ligand binds, the apo state STILL prefers the
DNA-compatible conformation (not constitutive), the ligand shifts that population, and the induced
state releases the operator. It does NOT train a general allostery model, does NOT run MD, and does
NOT claim absolute Kd. Every score is a relative proxy calibrated on the scaffold's own controls.

REVISED after GPT-5.6 review:
  * scoring.yaml is actually loaded and handed to rank (it was silently ignored)
  * --top-scaffolds N really runs N scaffolds, each in its own output dir (it used to run 1)
  * an explicit --objective that contradicts Route's automatic mode is an ERROR, not a silent mismatch
  * project ids are sanitised + hashed, and an existing dir is never overwritten without --overwrite
  * every stage declares REQUIRES/PRODUCES and the framework enforces the contract, so a stage
    cannot return a partial ctx and silently discard upstream state
"""
import os, re, sys, json, time, hashlib, argparse, datetime, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))

# name, owner, description, requires, produces
STAGES = [
    ("route",         "A", "target molecule -> native ligand + scaffold + design mode",
     [], ["route", "scaffold"]),
    ("structure",     "B", "prepare & QC native states (X_D, X_I, X_A)",
     ["scaffold"], ["states", "residue_mapping"]),
    ("allostery",     "B", "system-specific allosteric template + masks",
     ["states"], ["template", "masks"]),
    ("pose",          "C", "target ligand poses on BOTH backbones",
     ["states", "masks"], ["poses"]),
    ("design",        "C", "LigandMPNN pocket proposals under masks",
     ["states", "poses", "masks"], ["candidates"]),
    ("state_builder", "D", "six static states per candidate (incl. ligand-free apo bias)",
     ["candidates", "states", "poses"], ["candidate_states"]),
    ("ligand_score",  "D", "dG_lig, dG_apo, ddG_coupling",
     ["candidate_states"], ["ligand_scores"]),
    ("dna_release",   "D", "S_release = E_DNA(X_I) - E_DNA(X_D), sign from topology",
     ["candidate_states"], ["dna_scores"]),
    ("specificity",   "D", "negative design vs native / analogues / metabolites",
     ["candidate_states"], ["specificity_scores"]),
    ("rank",          "E", "control calibration -> hard gates -> Pareto -> diversity",
     ["ligand_scores", "dna_scores", "specificity_scores", "cfg_scoring"], ["ranked"]),
]
MODE_MAP = {"auto": None, "enhance": "ENHANCEMENT", "retarget": "RETARGETING"}


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def project_id(target, scaffold):
    """Sanitised, collision-resistant. Raw SMILES contain / \\ = # which break paths, and
    truncated names collide."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", target).strip("-")[:24] or "target"
    h = hashlib.sha1(target.encode()).hexdigest()[:6]
    return "%s-%s_%s" % (slug, h, scaffold)


def run_stage(name, owner, ctx, requires, produces):
    missing = [k for k in requires if k not in ctx or ctx[k] is None]
    if missing:
        raise RuntimeError("stage '%s' requires %s which upstream never produced" % (name, missing))
    mod = __import__("pipeline." + name, fromlist=["*"])
    fn = getattr(mod, "run", None)
    if fn is None:
        raise NotImplementedError("pipeline/%s.py has no run(ctx) entry point yet" % name)
    out = fn(ctx)
    if out is None:
        raise RuntimeError("stage '%s' returned None; it must return the updated ctx" % name)
    lost = [k for k in ctx if k not in out]
    if lost:
        raise RuntimeError("stage '%s' dropped upstream keys %s from ctx" % (name, lost))
    absent = [k for k in produces if k not in out or out[k] is None]
    if absent:
        raise RuntimeError("stage '%s' did not produce %s" % (name, absent))
    return out


def run_one_scaffold(hit, a, cfg, cfg_scoring, r, root_out):
    scaffold = hit["tf"]
    out = os.path.join(root_out, project_id(a.target, scaffold))
    if os.path.exists(out) and not a.overwrite:
        print("  SKIP %s: %s already exists (use --overwrite)" % (scaffold, out))
        return 4
    os.makedirs(os.path.join(out, "structures"), exist_ok=True)
    json.dump(r, open(os.path.join(out, "route_report.json"), "w"), indent=2, default=str)

    print("\n" + "-" * 78)
    print("scaffold %s  (%s / %s)  tier T%d  S_chem=%.3f S_struct=%.3f"
          % (scaffold, hit["family"], hit["native"], hit["tier"], hit["s_chem"], hit["s_struct"]))
    print("-" * 78)

    ctx = dict(cfg=cfg, cfg_scoring=cfg_scoring, out=out, scaffold=scaffold, route=r,
               target=a.target, target_smiles=r["smiles"], mode=r["mode"], hit=hit,
               seed=a.seed, started=time.time())
    for name, owner, desc, req, prod in STAGES[1:]:
        print("\n[%s] (owner %s) %s" % (name, owner, desc))
        try:
            ctx = run_stage(name, owner, ctx, req, prod)
        except NotImplementedError as e:
            print("  NOT IMPLEMENTED -> %s" % e)
            print("  owner %s implements pipeline/%s.py:run(ctx)" % (owner, name))
            print("    requires: %s\n    produces: %s" % (req or "-", prod))
            _summary(out, name, owner)
            return 2
        except Exception:
            print("  FAILED:\n" + traceback.format_exc())
            return 3
    print("\ndone -> %s/ranked_candidates.csv" % out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="molecule name or SMILES")
    ap.add_argument("--objective", default="auto", choices=["auto", "enhance", "retarget"])
    ap.add_argument("--top-scaffolds", type=int, default=3, help="how many designable scaffolds to actually run")
    ap.add_argument("--raw-designs", type=int, default=10000)
    ap.add_argument("--final-designs", type=int, default=96)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "default.yaml"))
    ap.add_argument("--scoring", default=os.path.join(ROOT, "config", "scoring.yaml"))
    ap.add_argument("--scaffold", default=None, help="force one scaffold, bypassing Route's pick")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--force", action="store_true", help="proceed despite an objective/mode conflict")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    cfg_scoring = load_yaml(a.scoring)          # was never loaded before -> gates never reached rank
    cfg["target"]["name"] = a.target
    cfg["route"]["top_scaffolds"] = a.top_scaffolds
    cfg["design"]["raw_designs"] = a.raw_designs
    cfg["output"]["final_designs"] = a.final_designs

    from pipeline.route import route as run_route
    r = run_route(a.target, max(a.top_scaffolds, 10))

    print("=" * 78)
    print("AlloTF V1   target=%s   mode=%s   %s"
          % (a.target, r["mode"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("=" * 78)
    print("\n[route] (owner A) chemistry and structure scored separately - a chemically perfect")
    print("        hit with no usable structure is NOT designable")
    print("  %-11s %-9s %-22s %6s %8s %5s %s"
          % ("TF", "family", "native ligand", "S_chem", "S_struct", "tier", "designable"))
    for h in r["hits"][:10]:
        print("  %-11s %-9s %-22s %6.3f %8.3f   T%d  %s"
              % (h["tf"], h["family"], h["native"][:22], h["s_chem"], h["s_struct"], h["tier"],
                 "yes" if h["designable"] else "NO (no usable structure)"))

    # explicit objective must not silently disagree with Route
    want = MODE_MAP[a.objective]
    if want and not r["mode"].startswith(want):
        msg = ("objective/mode conflict: you asked for %s but Route resolved the target as %s "
               "(nearest native ligand decides this)." % (want, r["mode"]))
        if not a.force:
            print("\nSTOP: " + msg + "\n      Re-run with --force to override, or pick a different target.")
            return 1
        print("\nWARNING: " + msg + " Proceeding because --force.")
    cfg["objective"]["mode"] = r["mode"]

    designable = [h for h in r["hits"] if h["designable"]]
    if a.scaffold:
        sel = [h for h in r["hits"] if h["tf"] == a.scaffold]
        if not sel:
            print("\nSTOP: --scaffold %s is not in the retrieved set." % a.scaffold)
            return 1
        if not sel[0]["designable"] and not a.force:
            print("\nSTOP: --scaffold %s is Tier %d (no usable structure). Use --force to accept."
                  % (a.scaffold, sel[0]["tier"]))
            return 1
    else:
        sel = designable[:a.top_scaffolds]
    if not sel:
        print("\nSTOP: no designable scaffold. The closest chemistry has no usable structure.")
        print("      Pick a target nearer a structurally characterised effector, or force a")
        print("      Tier-3 scaffold at your own risk (--scaffold NAME --force).")
        return 1

    root_out = os.path.join(ROOT, cfg["output"]["project_dir"])
    print("\n[route] running %d scaffold(s): %s   mode: %s"
          % (len(sel), ", ".join(h["tf"] for h in sel), r["mode"]))
    codes = [run_one_scaffold(h, a, cfg, cfg_scoring, r, root_out) for h in sel]
    return 0 if all(c == 0 for c in codes) else max(codes)


def _summary(out, stopped_at, owner):
    done = [n for n, *_ in STAGES if n != stopped_at]
    done = done[:[n for n, *_ in STAGES].index(stopped_at)]
    print("\n" + "-" * 78)
    print("progress: %d/%d stages wired" % (len(done), len(STAGES)))
    print("  done   : %s" % (", ".join(done) if done else "-"))
    print("  next   : %s  (owner %s)" % (stopped_at, owner))
    print("  outputs: %s" % out)
    print("-" * 78)


if __name__ == "__main__":
    sys.exit(main())
