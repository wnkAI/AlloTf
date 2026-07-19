"""Run manifest: everything needed to reproduce or audit one design run.

A candidate list is only trustworthy if you can answer "which code, which inputs, which parameters
produced it". This writes that record - git commit, environment, seeds, and a hash of every input
file - so a run can be replayed and a reviewer can check it was not quietly changed after the fact.
"""
import os
import sys
import json
import hashlib
import platform
import subprocess


def _sha1(path):
    if not path or not os.path.isfile(path):
        return None
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def _git(*args):
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _pyrosetta_version():
    try:
        import pyrosetta
        return getattr(pyrosetta, "__version__", "unknown")
    except Exception:
        return None


def _env():
    pkgs = {}
    for name in ("rdkit", "scipy", "sklearn", "numpy", "Bio"):
        try:
            pkgs[name] = __import__(name).__version__
        except Exception:
            pkgs[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "pyrosetta": _pyrosetta_version(), "packages": pkgs}


def _ligand_provenance(ctx):
    """Per-ligand chemistry that must match between the pose and the .params, or Rosetta reads the
    wrong molecule. Recorded from the parameter bundle."""
    lp = ctx.get("ligand_params") or {}
    out = {}
    for key, entry in [("target", lp.get("target"))] + list((lp.get("decoys") or {}).items()):
        if not entry:
            continue
        out[key] = {"resname": entry.get("resname"), "formal_charge": entry.get("formal_charge"),
                    "smiles": entry.get("smiles"), "params_sha1": _sha1(entry.get("params"))}
    return {"ligands": out, "bundle_hash": lp.get("bundle_hash")}


def build(ctx, args=None):
    """-> the manifest dict for this run. Pulls from ctx whatever the stages already produced;
    absent fields are recorded as None rather than omitted, so a gap is visible, not hidden."""
    states = ctx.get("states") or {}
    paths = states.get("paths") or {}
    cfg = ctx.get("cfg") or {}

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "environment": _env(),
        "seed": ctx.get("seed"),
        "scorefunction": cfg.get("design", {}).get("score_function", "ref2015"),
        "target": {"query": ctx.get("target"), "smiles": (ctx.get("route") or {}).get("smiles"),
                   "mode": ctx.get("mode")},
        "scaffold": ctx.get("scaffold"),
        "config_sha1": {"default": _sha1(getattr(args, "config", None)),
                        "scoring": _sha1(getattr(args, "scoring", None))},
        "structure_sha1": {k: _sha1(v) for k, v in paths.items()},
        "ligands": _ligand_provenance(ctx),
        "validation_level": (ctx.get("ranked") or {}).get("validation_level"),
        "category_counts": (ctx.get("ranked") or {}).get("category_counts"),
    }


def write(ctx, project, args=None):
    """Write run_manifest.json into the project. -> the path."""
    m = build(ctx, args)
    path = os.path.join(project, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=2, default=str)
    return path
