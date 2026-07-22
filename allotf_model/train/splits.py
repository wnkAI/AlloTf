"""Held-out splits that actually test cross-biosensor generalisation. Random splits leak near-identical
mutants of one TF across train and test, so only whole-group hold-outs mean anything: a whole scaffold,
a whole structural family, a whole ligand, or a whole ligand CHEMOTYPE.
"""
from collections import defaultdict


def _scaf(s):
    return s.scaffold_id


def _family(s):
    return s.provenance.get("family", s.scaffold_id)


def _grouped(samples, key):
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[key(s)].append(i)
    for g, test in groups.items():
        train = [i for i in range(len(samples)) if i not in set(test)]
        yield g, train, test


def leave_one_scaffold_out(samples):
    return list(_grouped(samples, _scaf))


def leave_one_family_out(samples):
    return list(_grouped(samples, _family))


def leave_one_ligand_out(samples):
    """Hold out one ligand ID at a time. NOTE: this is per-ligand, NOT per-chemotype - a close analogue
    of the held-out ligand can still be in training. Use leave_one_ligand_chemotype_out for chemotype
    generalisation."""
    return list(_grouped(samples, lambda s: s.ligand_id))


def ligand_chemotype_clusters(ligand_smiles, cutoff=0.4):
    """{ligand_id: smiles} -> {ligand_id: cluster_id} by Butina clustering on Morgan fingerprints.
    Two ligands share a cluster if their Tanimoto distance is below cutoff."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
    ids = list(ligand_smiles)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ligand_smiles[i]), 2, 2048) for i in ids]
    dists = []
    for a in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[a], fps[:a])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    out = {}
    for cid, members in enumerate(clusters):
        for m in members:
            out[ids[m]] = cid
    return out


def leave_one_ligand_chemotype_out(samples, ligand_smiles, cutoff=0.4):
    """Hold out a whole ligand CHEMOTYPE (Morgan-fingerprint cluster) at a time - the real test that
    the model handles a chemically NEW class of target, not just a new ID of a known chemotype."""
    clusters = ligand_chemotype_clusters(ligand_smiles, cutoff)
    return list(_grouped(samples, lambda s: clusters.get(s.ligand_id, -1)))
