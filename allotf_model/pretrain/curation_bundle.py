"""Assemble the human-curation bundle per scaffold - the STOPPING point of the automatic Part-1
pipeline. Nothing is frozen or trained; this is the evidence a human signs off on before the region
YAML is frozen and native teachers are built.

Post-review corrections:
- teacher eligibility (response_consistency) is gated ONLY on the operator-derived DBD residues; when
  there is no operator structure the verdict is no_teacher_missing_DBD (fail closed), never a fabricated
  mask. The DBD..pocket span is used ONLY as an exploratory frame/hinge region, not for pass/fail.
- hinge evidence is computed PER teacher cluster (propagation can differ by ligand); the scaffold-wide
  hinge is only the consensus, with heterogeneity reported.
"""
import glob
import os

from .structure_qc import qc_scaffold_structures
from .region_evidence import pocket_evidence, dbd_evidence, hinge_evidence
from .effector_clusters import cluster_effectors
from .response_consistency import consistency
from .ensemble_alignment import align_ensemble


def _yaml(bundle):
    L = ["scaffold_id: %s" % bundle["scaffold_id"], "reference_len: %d" % bundle["reference_len"],
         "status: DRAFT_needs_human_curation"]
    for region in ("pocket", "dbd", "dna_contact", "hinge", "gain_tuning"):
        r = bundle["regions"][region]
        L.append("%s:" % region)
        L.append("  canonical_indices: %s" % r["canonical_indices"])
        for k in ("evidence", "confidence", "exclusions", "note"):
            if r.get(k):
                L.append("  %s: %s" % (k, r[k]))
    L.append("teachers:")
    for t in bundle["teacher_candidates"]:
        L.append("  - cluster: %s" % t["cluster"])
        L.append("    ligands: %s" % t["ligands"])
        L.append("    verdict: %s" % t["verdict"])
        L.append("    n_apo: %s   n_holo: %s   hinge: %s" % (t.get("n_apo"), t.get("n_holo"), t.get("hinge", [])))
    return "\n".join(L)


def build_scaffold_bundle(scaffold_dir, sid, role="directional_candidate"):
    q = qc_scaffold_structures(scaffold_dir, sid)
    ref = q["reference_seq"]
    n = len(ref)
    pk = pocket_evidence(scaffold_dir, sid, ref)
    db = dbd_evidence(scaffold_dir, sid, ref)
    ec = cluster_effectors(scaffold_dir, sid, ref)
    pocket = pk["pocket_high_confidence"]
    dbd = db.get("dbd_contact_residues", [])
    apo = sorted(glob.glob(os.path.join(scaffold_dir, "apo", "*.cif")))
    # exploratory frame/hinge region (NOT the gate): DBD..pocket span
    lo = min(dbd) if dbd else (min(pocket) - 30 if pocket else 0)
    hi = min(pocket) if pocket else 40
    frame_distal = sorted(set(dbd) | set(range(lo, hi)))

    teachers, cluster_hinges = [], []
    for cid, v in ec["clusters"].items():
        if v["n_holo"] < 2:
            continue
        holo = [os.path.join(scaffold_dir, "holo", p + ".cif") for p in v["pdbs"]]
        cons = consistency(apo, holo, ref, dbd)             # gate on operator DBD only
        hinge = {"hinge_candidates": [], "note": "no ensemble"}
        try:
            ens = align_ensemble(apo, holo, ref, dbd, frame_distal)
            hinge = hinge_evidence(ens, pocket, dbd, n)
            cluster_hinges.append(set(hinge["hinge_candidates"]))
        except Exception as e:
            hinge = {"hinge_candidates": [], "note": "hinge evidence failed: %s" % str(e)[:60]}
        teachers.append({"cluster": "c%s" % cid, "ligands": v["comp_ids"],
                         "n_apo": cons.get("n_apo"), "n_holo": cons.get("n_holo"),
                         "verdict": cons["verdict"], "consistency": cons,
                         "hinge": hinge["hinge_candidates"][:12], "hinge_note": hinge.get("note")})

    consensus_hinge = sorted(set.intersection(*cluster_hinges)) if cluster_hinges else []
    regions = {
        "pocket": {"canonical_indices": pocket, "evidence": ["holo_ligand_contact", "contact_frequency"],
                   "confidence": "high"},
        "dbd": {"canonical_indices": dbd,
                "evidence": ["operator_contact"] if dbd else ["needs_operator_structure"],
                "confidence": db.get("confidence", "none")},
        "dna_contact": {"canonical_indices": dbd},
        "hinge": {"canonical_indices": consensus_hinge,
                  "evidence": ["apo_holo_motion", "contact_churn", "communication_bottleneck"],
                  "confidence": "medium" if consensus_hinge else "per_cluster_no_consensus",
                  "note": "per-cluster hinge sets in teacher_candidates; consensus = intersection"},
        "gain_tuning": {"canonical_indices": consensus_hinge,
                        "exclusions": ["dna_contact", "structural_core", "essential_dimer_interface"]}}

    return {"scaffold_id": sid, "role": role, "reference_len": n, "n_structures": q["n_structures"],
            "effector_clusters": {c: v["comp_ids"] for c, v in ec["clusters"].items()},
            "regions": regions, "teacher_candidates": teachers,
            "ligand_inventory": q["ligand_inventory"],
            "excluded": {"variant_outliers": q["n_variant_outliers"],
                         "failed_structures": [{"pdb": k, "reasons": v["reasons"]}
                                               for k, v in q["structures"].items() if not v.get("passed")]},
            "needs": pk.get("needs", []) + (["operator_structure_for_DBD"] if not dbd else [])}
