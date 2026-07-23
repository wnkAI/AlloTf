"""Download the QC-passed apo/holo BIOLOGICAL ASSEMBLIES (assembly-1 mmCIF, the functional oligomer -
not the asymmetric unit) for the scaffolds we are building teachers for. Best-resolution-first; a
per-state cap keeps the ensemble tractable and is LOGGED (available vs downloaded), never a silent
truncation. Entries without a downloadable assembly fail closed and are skipped with a reason.

Layout: <out>/<scaffold>/<apo|holo>/<pdb>.cif  + download_log.json
"""
import json
import os

from .rcsb import download_assembly_cif

# first batch: the directional inducer-release Tier-A scaffolds (generic - pass any subset)
DIRECTIONAL_7 = ["TetR", "QacR", "TtgR", "RamR", "EthR", "KstR", "LacI"]


def fetch(qc_path, manifest_path, out_dir, scaffolds=None, cap=12):
    scaffolds = scaffolds or DIRECTIONAL_7
    qc = json.load(open(qc_path))["scaffolds"]
    man = json.load(open(manifest_path))
    log = {}
    for name in scaffolds:
        if name not in qc or name not in man:
            print("%-6s NOT in QC/manifest - skip" % name); continue
        res = {e["pdb_id"]: (e.get("resolution") or 99.0) for e in man[name]["entries"]}
        passed = qc[name]["passed"]
        log[name] = {}
        for state in ("apo", "holo"):
            ids = sorted(passed.get(state, []), key=lambda p: res.get(p, 99.0))   # best resolution first
            take = ids[:cap]
            d = os.path.join(out_dir, name, state)
            os.makedirs(d, exist_ok=True)
            got = []
            for pid in take:
                try:
                    download_assembly_cif(pid, os.path.join(d, pid + ".cif"))
                    got.append(pid)
                except RuntimeError as e:
                    print("  skip %s/%s %s: %s" % (name, state, pid, str(e)[:50]))
            log[name][state] = {"available": len(ids), "downloaded": got, "capped": len(ids) > cap,
                                "resolutions": {p: res.get(p) for p in got}}
            print("%-6s %-4s available=%d downloaded=%d%s"
                  % (name, state, len(ids), len(got), "  (CAPPED at %d)" % cap if len(ids) > cap else ""))
        json.dump(log, open(os.path.join(out_dir, "download_log.json"), "w"), indent=2)
    return log


if __name__ == "__main__":
    fetch("results/pretrain_manifest/state_qc.json", "results/pretrain_manifest/manifest.json",
          "results/pretrain_structures")
