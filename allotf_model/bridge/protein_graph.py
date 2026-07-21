"""Protein graph extractor: the FIXED scaffold with the variant's mutation ENCODED on the nodes -
never a per-variant regenerated structure. Aligns everything to the canonical residue index.

Produces residue features (mutant one-hot, is-mutated flag, region flags, exposure, confidence), the
WT/mutant residue ids for the mutation encoding, CA positions, and a CA contact graph. It reads no
ligand and no labels - only the scaffold, the mutation and the region definitions.
"""
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

from .transfer_sample import TransferSample

warnings.simplefilter("ignore", PDBConstructionWarning)
_PARSER = PDBParser(QUIET=True)

AA3 = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
AA_IDX = {a: i for i, a in enumerate(AA3)}
UNK = len(AA3)
REGIONS = ("pocket", "pocket_exit", "hinge", "dimer_interface", "dbd", "dna_contact", "distal")
CONTACT = 8.0


def _one_hot(i, n):
    v = torch.zeros(n); v[i] = 1.0; return v


def build_protein_graph(scaffold_pdb, mapping, mutations, regions, confidence=None):
    """scaffold_pdb: WT scaffold. mapping: ResidueMapping. mutations: {canonical_index: AA3}.
    regions: {region_name: iterable of canonical indices}. confidence: {canonical_index: [0,1]} or None.
    -> dict of the protein-graph tensors keyed by TransferSample field names."""
    model = next(iter(_PARSER.get_structure(mapping.scaffold_id, scaffold_pdb)))
    n = mapping.n_res
    ca = {}
    wt = torch.full((n,), UNK, dtype=torch.long)
    for ch in model:
        for r in ch:
            if r.id[0] != " " or not r.has_id("CA"):
                continue
            ci = mapping.canonical(ch.id, r.id[1], r.id[2] or " ")
            if ci is None:
                continue
            ca[ci] = r["CA"].coord.astype(float)
            wt[ci] = AA_IDX.get(r.get_resname().strip(), UNK)
    missing = [i for i in range(n) if i not in ca]
    if missing:
        raise ValueError("scaffold PDB missing %d mapped residues (e.g. %s); mapping and structure "
                         "must match" % (len(missing), missing[:5]))

    mutant = wt.clone()
    mut_mask = torch.zeros(n, dtype=torch.bool)
    for ci, aa in (mutations or {}).items():
        mutant[ci] = AA_IDX.get(aa.strip().upper(), UNK)
        mut_mask[ci] = True

    region_masks = {}
    for name in REGIONS:
        m = torch.zeros(n, dtype=torch.bool)
        for ci in regions.get(name, ()):
            m[ci] = True
        region_masks[name] = m

    pos = torch.tensor(np.stack([ca[i] for i in range(n)]), dtype=torch.float32)
    d = torch.cdist(pos, pos)
    neigh = (d < CONTACT).sum(1).float() - 1
    exposure = 1.0 - (neigh / neigh.clamp_min(1).max()).clamp(0, 1)      # fewer neighbours = exposed
    conf = torch.ones(n) if confidence is None else torch.tensor(
        [float(confidence.get(i, 1.0)) for i in range(n)])

    region_stack = torch.stack([region_masks[r].float() for r in REGIONS], dim=1)   # [n, 7]
    feats = torch.cat([torch.stack([_one_hot(int(mutant[i]), UNK + 1) for i in range(n)]),
                       mut_mask.float().unsqueeze(1), region_stack,
                       exposure.unsqueeze(1), conf.unsqueeze(1)], dim=1)

    ei = (d < CONTACT).nonzero(as_tuple=False)
    ei = ei[ei[:, 0] != ei[:, 1]].t().contiguous()                       # [2, E], drop self-loops
    edge_feat = torch.stack([d[ei[0], ei[1]], (conf[ei[0]] * conf[ei[1]])], dim=1)

    return dict(residue_features=feats, residue_vectors=torch.zeros(n, 0, 3), residue_positions=pos,
                protein_edge_index=ei, protein_edge_features=edge_feat,
                wt_residue_ids=wt, mutant_residue_ids=mutant, mutation_mask=mut_mask,
                pocket_mask=region_masks["pocket"], pocket_exit_mask=region_masks["pocket_exit"],
                hinge_mask=region_masks["hinge"], dimer_interface_mask=region_masks["dimer_interface"],
                dbd_mask=region_masks["dbd"], dna_contact_mask=region_masks["dna_contact"],
                distal_mask=region_masks["distal"])
