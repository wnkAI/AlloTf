"""Chain-aware parsing of a biological assembly. A homodimer has two protomers, each with its OWN
bound ligand and its OWN operator DNA duplex; the old code chose the longest protein chain and the
largest ligand INDEPENDENTLY, so protomer A could be paired with protomer B's ligand/DNA. Here each
protein chain is aligned to the reference (canonical index by sequence), and every ligand / DNA duplex
is assigned to the protomer it is CLOSEST to, so contacts are always computed within one protomer.

Returned per protomer:
    chain, canonical_ca {ci: CA coord}, heavy_by_ci {ci: [heavy-atom coords]},
    effector (comp_id, coords) or None, dna coords or None.
Consumers average protomers within a PDB so a dimer contributes one observation, not two.
"""
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from .structure_qc import _align_identity, AA3TO1, MIN_IDENTITY, _MIN_CHAIN
from .build_manifest import IONS, ADDITIVES

_CIF, _PDB = MMCIFParser(QUIET=True), PDBParser(QUIET=True)
_NON_EFFECTOR = IONS | ADDITIVES
_DNA_RES = {"DA", "DC", "DG", "DT", "DU"}


def _nearest_chain(centroid, chain_centroids):
    return min(chain_centroids, key=lambda c: np.linalg.norm(centroid - chain_centroids[c]))


def parse_protomers(path, sid, reference_seq):
    parser = _CIF if path.lower().endswith(".cif") else _PDB
    model = next(iter(parser.get_structure(sid, path)))
    chains, ligands, dnas = {}, [], []
    for ch in model:
        res, dna_atoms = [], []
        for r in ch:
            nm = r.get_resname().strip().upper()
            if r.id[0] == " " and nm in AA3TO1 and r.has_id("CA"):
                heavy = np.array([a.coord for a in r if a.element != "H"])
                res.append((AA3TO1[nm], r["CA"].coord.astype(float), heavy))
            elif nm in _DNA_RES:
                dna_atoms.extend(a.coord for a in r if a.element != "H")
            elif r.id[0].startswith("H_") and nm not in _NON_EFFECTOR:
                c = np.array([a.coord for a in r if a.element != "H"])
                if len(c):
                    ligands.append((nm, c, c.mean(0)))
        if len(res) >= _MIN_CHAIN:
            chains[ch.id] = res
        if dna_atoms:
            a = np.array(dna_atoms); dnas.append((a, a.mean(0)))

    aligned = {}                                          # chain_id -> (canonical_ca, heavy_by_ci, centroid)
    for cid, res in chains.items():
        ident, _, r2s = _align_identity("".join(a for a, _, _ in res), reference_seq)
        if ident < MIN_IDENTITY:
            continue
        ca = {ci: res[pos][1] for ci, pos in r2s.items()}
        heavy = {ci: res[pos][2] for ci, pos in r2s.items()}
        aligned[cid] = (ca, heavy, np.mean(list(ca.values()), 0))
    if not aligned:
        return []
    chain_centroids = {cid: v[2] for cid, v in aligned.items()}

    lig_by_chain, dna_by_chain = {}, {}
    for comp, coords, cen in ligands:                     # each ligand -> nearest protomer
        cid = _nearest_chain(cen, chain_centroids)
        lig_by_chain.setdefault(cid, []).append((comp, coords))
    for coords, cen in dnas:                              # each DNA duplex -> nearest protomer
        cid = _nearest_chain(cen, chain_centroids)
        dna_by_chain.setdefault(cid, []).append(coords)

    protomers = []
    for cid, (ca, heavy, _) in aligned.items():
        ligs = lig_by_chain.get(cid, [])
        eff = max(ligs, key=lambda x: len(x[1])) if ligs else None      # largest ligand ON this protomer
        dna = np.concatenate(dna_by_chain[cid]) if cid in dna_by_chain else None
        protomers.append({"chain": cid, "canonical_ca": ca, "heavy_by_ci": heavy,
                          "effector": eff, "dna": dna})
    return protomers
