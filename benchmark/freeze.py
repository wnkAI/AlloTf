"""Freeze the scoring contract BEFORE any mutant is scored, so 'freeze then unblind' is enforced in
code, not just promised in a README.

The gate is a zero-shot test: its entire value is that the RSM formula, the six-state signs, the
thresholds and the resolvent were fixed without ever looking at the mutant labels. This module
hashes every file that defines one of those, plus the scalar knobs, into one contract hash. The
scorer calls verify() and refuses to run if the contract drifted since the dataset was frozen. To
change the formula on purpose you re-freeze deliberately - you cannot do it by accident mid-analysis.
"""
import hashlib
import json
import os
import subprocess

# every file whose content defines a margin, a sign convention, a threshold, or the resolvent
CONTRACT_FILES = [
    "pipeline/rsm.py",            # margins, z-score, weakest-link, CVaR
    "pipeline/resolvent.py",     # H_s, directed gain, m_trans
    "pipeline/state_builder.py", # linkage / dna_release / dna_affinity definitions + signs
    "pipeline/ensemble.py",      # P_sign aggregation
    "config/scoring.yaml",       # every gate threshold
    "benchmark/schema.py",       # label taxonomy + expected-weakest map
]

# scalar knobs that live in argument defaults, not in a file
CONSTANTS = {
    "cvar_alpha": 0.2,
    "resolvent_spring_ordering": "backbone>metal>saltbridge>hbond>contact",
}


def _sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def _git(repo):
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", repo, *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def contract_hash(repo):
    files = {f: _sha1(os.path.join(repo, f)) for f in CONTRACT_FILES}
    blob = json.dumps({"files": files, "constants": CONSTANTS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16], files


def freeze(repo, out_path):
    h, files = contract_hash(repo)
    doc = {"contract_hash": h, "files": files, "constants": CONSTANTS, "git": _git(repo)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def verify(repo, frozen_path):
    """Raise if the current scoring contract differs from the frozen one."""
    frozen = json.load(open(frozen_path))
    h, files = contract_hash(repo)
    if h != frozen["contract_hash"]:
        changed = [f for f in files if files[f] != frozen["files"].get(f)]
        raise RuntimeError(
            "scoring contract changed since freeze (%s != %s); changed: %s. The gate is a zero-shot "
            "test - the formula may not move between freezing the dataset and unblinding. Re-freeze "
            "on purpose if this is intended." % (h, frozen["contract_hash"], changed or "constants"))
    return True


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    tmp = os.path.join(repo, "benchmark", "_freeze_selftest.json")
    doc = freeze(repo, tmp)
    print("contract_hash =", doc["contract_hash"])
    verify(repo, tmp)
    print("verify (unchanged) OK")
    # tamper the recorded hash and confirm verify rejects it
    doc["contract_hash"] = "deadbeef"
    json.dump(doc, open(tmp, "w"))
    try:
        verify(repo, tmp)
        print("BUG: tampered contract accepted")
    except RuntimeError:
        print("verify (tampered) correctly rejected")
    os.remove(tmp)
