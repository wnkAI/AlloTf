"""Audit the structure DB for cross-scaffold pretraining readiness. The phase-1 deliverable is THIS
report, not a running network: how many scaffolds are genuinely ensemble-trainable, how many are
pair-only, what is excluded and why. It reports the DATA FACT and never inflates the count.

Availability from data/atf_structure_db.csv (counts only - apo PDB ids, effector, resolution etc. are
NOT in the CSV and need a PDB re-query + QC before a scaffold is CONFIRMED, so these tiers are
CANDIDATES, not QC-passed sets):
  Tier A candidate : n_apo >= 2 AND n_holo >= 2   (full ensemble self-supervision)
  Tier B candidate : n_apo >= 1 AND n_holo >= 1 but not A   (single apo->holo delta only)
  incomplete       : missing apo or holo
A candidate only joins the DIRECTIONAL pretraining if its topology is a curated inducer_release; a
corepressor (PurR) or an uncurated-topology scaffold is held out of the directional ensemble.
"""
import csv
import json
import os

from .structure_manifest import SCAFFOLD_META, INDUCER_RELEASE, COREPRESSOR_BINDING


def audit(csv_path="data/atf_structure_db.csv"):
    rows = list(csv.DictReader(open(csv_path)))
    tiers = {"A": [], "B": [], "incomplete": []}
    topo = {"inducer_release": [], "corepressor": [], "uncurated": []}
    dbd_capable, families = [], {}
    for r in rows:
        name, na, nh, nd = r["tf_name"], int(r["n_apo"]), int(r["n_holo"]), int(r["n_dna"])
        fam, tp = SCAFFOLD_META.get(name, (None, None))
        families.setdefault(fam or "UNCURATED", []).append(name)
        if na >= 2 and nh >= 2:
            tiers["A"].append(name)
        elif na >= 1 and nh >= 1:
            tiers["B"].append(name)
        else:
            tiers["incomplete"].append(name)
        topo[("inducer_release" if tp == INDUCER_RELEASE else
              "corepressor" if tp == COREPRESSOR_BINDING else "uncurated")].append(name)
        if nd >= 1:
            dbd_capable.append(name)

    directional_A = [n for n in tiers["A"] if n in topo["inducer_release"]]
    return {"n_candidates": len(rows),
            "tierA_candidates": sorted(tiers["A"]), "tierB_candidates": sorted(tiers["B"]),
            "incomplete": sorted(tiers["incomplete"]),
            "topology": {k: sorted(v) for k, v in topo.items()},
            "directional_ensemble_ready_candidates": sorted(directional_A),
            "dbd_supervision_capable": sorted(dbd_capable),
            "families": {k: sorted(v) for k, v in families.items()},
            "caveats": [
                "counts come from the CSV; apo PDB ids / effector / resolution / oligomeric state are "
                "NOT stored and must be re-queried + QC'd before any scaffold is confirmed",
                "'holo' in the CSV = a non-additive small molecule is present; it is NOT proven to be "
                "the native effector until QC",
                "these are CANDIDATE counts, not QC-passed ensemble scaffolds"]}


def report(a):
    L = ["STRUCTURE-DB AUDIT (pretraining readiness)", "=" * 44,
         "candidate scaffolds in DB: %d" % a["n_candidates"],
         "Tier A candidates (>=2 apo & >=2 holo): %d  %s" % (len(a["tierA_candidates"]), a["tierA_candidates"]),
         "Tier B candidates (>=1 apo & >=1 holo, not A): %d  %s" % (len(a["tierB_candidates"]), a["tierB_candidates"]),
         "incomplete (missing apo or holo): %d  %s" % (len(a["incomplete"]), a["incomplete"]),
         "",
         "topology  inducer_release: %d | corepressor: %d | uncurated: %d" %
         (len(a["topology"]["inducer_release"]), len(a["topology"]["corepressor"]), len(a["topology"]["uncurated"])),
         "  corepressor (held out of directional ensemble): %s" % a["topology"]["corepressor"],
         "  uncurated topology (needs curation before use): %s" % a["topology"]["uncurated"],
         "",
         "DIRECTIONAL ensemble-ready candidates (Tier A & inducer_release): %d  %s" %
         (len(a["directional_ensemble_ready_candidates"]), a["directional_ensemble_ready_candidates"]),
         "DBD-supervision capable (>=1 operator/DNA structure): %d  %s" %
         (len(a["dbd_supervision_capable"]), a["dbd_supervision_capable"]),
         "",
         "families: " + ", ".join("%s(%d)" % (k, len(v)) for k, v in a["families"].items()),
         "", "CAVEATS:"]
    L += ["  - " + c for c in a["caveats"]]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    a = audit(sys.argv[1] if len(sys.argv) > 1 else "data/atf_structure_db.csv")
    print(report(a))
    out = "results/pretrain_audit.json"
    os.makedirs("results", exist_ok=True)
    json.dump(a, open(out, "w"), indent=2)
    print("\nwritten -> %s" % out)
