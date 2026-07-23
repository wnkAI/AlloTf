"""Fail-closed QC on the re-queried manifest, generic over EVERY scaffold (no hardcoded names). Two
tiers of check:

METADATA (from the manifest, verifiable now, fail-closed):
  - resolution present and <= RES_MAX (missing resolution is kept but flagged - NMR/cryo has none);
  - an 'apo' entry must carry NO organic ligand (an accidental effector/cofactor means it is not apo);
  - a 'holo' entry must carry an organic effector candidate;
  - a 'dna' entry must actually contain operator DNA.
Entries that fail are EXCLUDED (qc_status='fail' + reasons); the scaffold's tier is recomputed from
the survivors, so "Tier A" after QC means >=2 apo AND >=2 holo actually passed.

STRUCTURE-LEVEL checks the manifest cannot answer (biological-assembly oligomer, native-effector
identity, apo/holo sequence consistency, canonical-mapping and pocket/hinge/DBD coverage) are NOT
faked here - they are listed per scaffold as `needs_structure_qc` and run in the alignment pass on the
downloaded biological assemblies.
"""
RES_MAX = 3.5


def qc_entry(entry, res_max=RES_MAX):
    """entry: one manifest entry dict. -> (passed: bool, reasons: list)."""
    if "error" in entry:
        return False, ["fetch_error"]
    reasons = []
    res = entry.get("resolution")
    if res is not None and res > res_max:
        reasons.append("resolution_%.1f>%.1f" % (res, res_max))
    state, effs = entry.get("state"), entry.get("effector_candidates") or []
    if state == "apo" and effs:
        reasons.append("apo_has_ligand_%s" % ",".join(effs[:3]))
    if state == "holo" and not effs:
        reasons.append("holo_without_effector")
    if state == "dna" and entry.get("n_dna_entities", 0) < 1:
        reasons.append("dna_state_without_dna")
    return (len(reasons) == 0), reasons


def qc_scaffold(scaffold, res_max=RES_MAX):
    """scaffold: manifest[name] dict. -> report with post-QC counts + per-entry verdicts."""
    entries = scaffold.get("entries", [])
    passed = {"apo": [], "holo": [], "dna": [], "ternary": []}
    failed = []
    for e in entries:
        ok, reasons = qc_entry(e, res_max)
        if ok and e.get("state") in passed:
            passed[e["state"]].append(e["pdb_id"])
        elif not ok:
            failed.append({"pdb_id": e.get("pdb_id"), "state": e.get("state"), "reasons": reasons})
    na, nh = len(passed["apo"]), len(passed["holo"])
    tier = "A" if (na >= 2 and nh >= 2) else "B" if (na >= 1 and nh >= 1) else "incomplete"
    return {"uniprot": scaffold.get("uniprot"), "passed": passed, "failed": failed,
            "n_apo_pass": na, "n_holo_pass": nh, "tier_after_qc": tier,
            "needs_structure_qc": ["biological_assembly_oligomer", "native_effector_identity",
                                   "apo_holo_sequence_consistency", "pocket_hinge_dbd_coverage"]}


def qc_manifest(manifest, res_max=RES_MAX):
    """-> {scaffold: qc_scaffold(...)} plus a tier summary."""
    out = {name: qc_scaffold(sc, res_max) for name, sc in manifest.items() if sc.get("entries")}
    tiers = {"A": [], "B": [], "incomplete": []}
    for name, r in out.items():
        tiers[r["tier_after_qc"]].append(name)
    return {"scaffolds": out, "tiers": {k: sorted(v) for k, v in tiers.items()}}
