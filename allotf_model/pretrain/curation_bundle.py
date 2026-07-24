"""Assemble the human-curation bundle for each scaffold - the STOPPING point of the automatic Part-1
pipeline. Nothing here is frozen or fed to training; it is the evidence a human signs off on before the
region YAML is frozen and native teachers are built.

Per scaffold the bundle carries: the PDB assemblies used, the effector clusters, canonical coverage,
pocket / DBD / hinge evidence, the response-consistency verdict per cluster, excluded structures with
reasons, and an AUTO-SUGGESTED region YAML (draft, with evidence + confidence per region). Every region
residue is traceable to its evidence source; confidence is degraded, never invented, when a structure
is missing (e.g. DBD without an operator complex).
"""
import glob
import json
import os

from .structure_qc import qc_scaffold_structures
from .region_evidence import pocket_evidence, dbd_evidence, hinge_evidence
from .effector_clusters import cluster_effectors
from .response_consistency import consistency
from .ensemble_alignment import align_ensemble


def _yaml(bundle):
    """Minimal region-YAML draft emitter (fixed schema, no pyyaml dependency)."""
    L = ["scaffold_id: %s" % bundle["scaffold_id"], "reference_len: %d" % bundle["reference_len"],
         "status: DRAFT_needs_human_curation"]
    for region in ("pocket", "dbd", "dna_contact", "hinge", "gain_tuning"):
        r = bundle["regions"][region]
        L.append("%s:" % region)
        L.append("  canonical_indices: %s" % r["canonical_indices"])
        if r.get("evidence"):
            L.append("  evidence: %s" % r["evidence"])
        if r.get("confidence"):
            L.append("  confidence: %s" % r["confidence"])
        if r.get("exclusions"):
            L.append("  exclusions: %s" % r["exclusions"])
    L.append("teachers:")
    for t in bundle["teacher_candidates"]:
        L.append("  - cluster: %s" % t["cluster"])
        L.append("    ligands: %s" % t["ligands"])
        L.append("    verdict: %s" % t["verdict"])
        L.append("    n_apo: %d   n_holo: %d" % (t["n_apo"], t["n_holo"]))
    return "\n".join(L)


def build_scaffold_bundle(scaffold_dir, sid, role="directional_candidate"):
    q = qc_scaffold_structures(scaffold_dir, sid)
    ref = q["reference_seq"]
    n = len(ref)
    pk = pocket_evidence(scaffold_dir, sid, ref)
    db = dbd_evidence(scaffold_dir, sid, ref)
    ec = cluster_effectors(scaffold_dir, sid, ref)
    apo = sorted(glob.glob(os.path.join(scaffold_dir, "apo", "*.cif")))
    pocket = pk["pocket_high_confidence"]
    dbd = db.get("dbd_contact_residues", [])
    lo = min(dbd) if dbd else (min(pocket) - 30 if pocket else 0)
    hi = min(pocket) if pocket else 40
    distal = sorted(set(dbd) | set(range(lo, hi)))

    teachers, hinge = [], {"hinge_candidates": [], "note": "no ensemble"}
    for cid, v in ec["clusters"].items():
        if v["n_holo"] < 2:
            continue
        holo = [os.path.join(scaffold_dir, "holo", p + ".cif") for p in v["pdbs"]]
        cons = consistency(apo, holo, ref, dbd, distal)
        teachers.append({"cluster": "c%s" % cid, "ligands": v["comp_ids"], "n_apo": cons.get("n_apo", 0),
                         "n_holo": cons.get("n_holo", v["n_holo"]), "verdict": cons["verdict"],
                         "magnitude_significant": cons.get("magnitude_significant"),
                         "direction_consistent": cons.get("direction_consistent"),
                         "real_magnitude": cons.get("real_magnitude"),
                         "mean_holo_direction_cosine": cons.get("mean_holo_direction_cosine")})
        if hinge["hinge_candidates"] == [] and v["n_holo"] >= 2:      # one ensemble for the hinge evidence
            try:
                ens = align_ensemble(apo, holo, ref, dbd, distal)
                hinge = hinge_evidence(ens, pocket, dbd, n)
            except Exception as e:
                hinge = {"hinge_candidates": [], "note": "hinge evidence failed: %s" % str(e)[:60]}

    regions = {
        "pocket": {"canonical_indices": pocket, "evidence": ["holo_ligand_contact", "contact_frequency"],
                   "confidence": "high"},
        "dbd": {"canonical_indices": dbd,
                "evidence": ["operator_contact"] if dbd else ["needs_operator_structure"],
                "confidence": db.get("confidence", "none")},
        "dna_contact": {"canonical_indices": dbd},
        "hinge": {"canonical_indices": hinge["hinge_candidates"],
                  "evidence": ["apo_holo_motion", "contact_churn", "communication_bottleneck"],
                  "confidence": "medium" if hinge.get("dbd_in_ensemble") else "low_dbd_absent"},
        "gain_tuning": {"canonical_indices": hinge["hinge_candidates"],
                        "exclusions": ["dna_contact", "structural_core", "essential_dimer_interface"]}}

    excluded = {"variant_outliers": q["n_variant_outliers"],
                "failed_structures": [{"pdb": k, "reasons": v["reasons"]}
                                      for k, v in q["structures"].items() if not v.get("passed")]}
    return {"scaffold_id": sid, "role": role, "reference_len": n,
            "n_structures": q["n_structures"], "canonical_coverage_note": "ref = dominant sequence cluster",
            "effector_clusters": {c: v["comp_ids"] for c, v in ec["clusters"].items()},
            "regions": regions, "teacher_candidates": teachers,
            "ligand_inventory": q["ligand_inventory"], "excluded": excluded,
            "needs": pk.get("needs", []) + (["operator_structure_for_DBD"] if not dbd else [])}
