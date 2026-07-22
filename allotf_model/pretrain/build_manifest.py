"""Re-query every candidate scaffold from RCSB and build the real StructureManifest. Fixes the CSV's
gaps: it recovers apo PDB ids (never stored), reads the actual bound ligands (to tell apo/holo/dna/
ternary and to flag the likely native effector), and records resolution and entity counts.

State classification (flagged, not final - effector identity is confirmed later in state_qc):
  ternary  = has operator DNA AND a non-additive organic ligand
  dna      = has operator DNA, no organic ligand
  holo     = no DNA, has a non-additive organic ligand (candidate effector)
  apo      = no DNA, no organic ligand (ions/additives only)
"""
import csv
import json
import os
import time

from .rcsb import search_by_uniprot, fetch_entry

# ions and common crystallisation additives are NOT effectors
IONS = {"MG", "MN", "ZN", "CA", "NA", "K", "CL", "BR", "IOD", "FE", "FE2", "NI", "CO", "CU", "CD",
        "HG", "SR", "CS", "RB", "BA", "PB", "AU", "AG", "PT", "SO4", "PO4", "NO3", "F"}
ADDITIVES = {"GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "ACT", "ACY", "DMS", "MPD",
             "FMT", "TRS", "EPE", "IMD", "BME", "CIT", "TLA", "MES", "MLI", "SCN", "AZI", "DTT",
             "BOG", "LDA", "OLC", "PEO", "12P", "15P", "BU3", "FLC", "UNX", "UNL", "DOD"}


def classify(meta):
    ligs = [c.upper() for c in meta["nonpolymer_comp_ids"]]
    organic = [c for c in ligs if c not in IONS and c not in ADDITIVES]
    has_dna = meta["n_dna_entities"] >= 1
    if has_dna and organic:
        return "ternary", organic
    if has_dna:
        return "dna", []
    if organic:
        return "holo", organic
    return "apo", []


def build(csv_path="data/atf_structure_db.csv", out_dir="results/pretrain_manifest",
          scaffolds=None, sleep=0.1):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(csv_path)))
    manifest = {}
    for r in rows:
        name, uni = r["tf_name"], r["uniprot"]
        if scaffolds and name not in scaffolds:
            continue
        if not uni or uni.lower() in ("", "na", "none"):
            manifest[name] = {"uniprot": uni, "error": "no uniprot", "entries": []}
            continue
        try:
            pdbs = search_by_uniprot(uni)
        except RuntimeError as e:
            manifest[name] = {"uniprot": uni, "error": str(e), "entries": []}
            continue
        entries = []
        for pid in pdbs:
            try:
                m = fetch_entry(pid)
            except RuntimeError as e:
                entries.append({"pdb_id": pid, "error": str(e)}); continue
            state, effs = classify(m)
            entries.append({"pdb_id": pid, "state": state, "effector_candidates": effs,
                            "resolution": m["resolution"], "n_dna_entities": m["n_dna_entities"]})
            time.sleep(sleep)
        counts = {s: sum(1 for e in entries if e.get("state") == s) for s in ("apo", "holo", "dna", "ternary")}
        manifest[name] = {"uniprot": uni, "n_pdbs": len(pdbs), "counts": counts, "entries": entries}
        json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
        print("%-10s uniprot=%s pdbs=%d  apo=%d holo=%d dna=%d ternary=%d"
              % (name, uni, len(pdbs), counts["apo"], counts["holo"], counts["dna"], counts["ternary"]))
    return manifest


def requeried_tiers(manifest):
    """Tiers from the RE-QUERIED counts (apo now real, not missing from the CSV)."""
    A, B, inc = [], [], []
    for name, d in manifest.items():
        c = d.get("counts")
        if not c:
            inc.append(name); continue
        if c["apo"] >= 2 and c["holo"] >= 2:
            A.append(name)
        elif c["apo"] >= 1 and c["holo"] >= 1:
            B.append(name)
        else:
            inc.append(name)
    return {"tierA": sorted(A), "tierB": sorted(B), "incomplete": sorted(inc)}


if __name__ == "__main__":
    import sys
    scaf = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    man = build(scaffolds=scaf)
    t = requeried_tiers(man)
    print("\nRE-QUERIED tiers: A(>=2/>=2)=%d %s\n  B(>=1/>=1)=%d %s\n  incomplete=%d"
          % (len(t["tierA"]), t["tierA"], len(t["tierB"]), t["tierB"], len(t["incomplete"])))
    json.dump(t, open("results/pretrain_manifest/requeried_tiers.json", "w"), indent=2)
