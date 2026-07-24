"""Effector clustering - the second half of the teacher primary key (scaffold_id + effector_cluster_id).
A multidrug repressor's holo structures bind chemically different ligands that may induce DIFFERENT
distal responses; averaging them into one teacher is wrong. So the holo ligands are clustered by
chemotype (Morgan-fingerprint Butina) and each cluster's holo PDBs are grouped - a candidate teacher is
built per (scaffold, cluster), never across clusters.

Chemotype is only the INITIAL split; the response-consistency QC later refines/merges clusters by
actual response direction. Ligands with no retrievable SMILES are reported, not silently dropped.
"""
import glob
import os
from collections import Counter

from .structure_parse import parse_protomers
from .rcsb import fetch_comp_smiles
from ..train.splits import ligand_chemotype_clusters


def cluster_effectors(scaffold_dir, sid, reference_seq, cutoff=0.4):
    """-> dict(clusters {cluster_id: {comp_ids, pdbs, n_holo}}, smiles, no_smiles, pdb_effector)."""
    pdb_eff = {}
    for p in sorted(glob.glob(os.path.join(scaffold_dir, "holo", "*.cif"))):
        effs = [pr["effector"][0] for pr in parse_protomers(p, sid, reference_seq) if pr["effector"]]
        if effs:                                          # per-PDB effector = the comp bound on most protomers
            pdb_eff[os.path.splitext(os.path.basename(p))[0]] = Counter(effs).most_common(1)[0][0]

    comp_ids = sorted(set(pdb_eff.values()))
    smiles, no_smiles = {}, []
    for c in comp_ids:
        s = fetch_comp_smiles(c)
        (smiles.__setitem__(c, s) if s else no_smiles.append(c))

    if len(smiles) == 1:
        comp2cluster = {next(iter(smiles)): 0}
    elif smiles:
        comp2cluster = ligand_chemotype_clusters(smiles, cutoff)
    else:
        return {"clusters": {}, "smiles": {}, "no_smiles": no_smiles, "pdb_effector": pdb_eff,
                "error": "no ligand SMILES retrievable"}

    clusters = {}
    for pdb, comp in pdb_eff.items():
        cid = comp2cluster.get(comp)
        if cid is None:                                   # ligand had no SMILES -> its own uncurated bucket
            continue
        clusters.setdefault(cid, {"comp_ids": set(), "pdbs": []})
        clusters[cid]["comp_ids"].add(comp)
        clusters[cid]["pdbs"].append(pdb)
    clusters = {cid: {"comp_ids": sorted(v["comp_ids"]), "pdbs": sorted(v["pdbs"]),
                      "n_holo": len(v["pdbs"])} for cid, v in sorted(clusters.items())}
    return {"clusters": clusters, "smiles": smiles, "no_smiles": no_smiles, "pdb_effector": pdb_eff}
