"""Residue numbering map across PDB entries/constructs of the same TF.

Different entries of one TF use different author numbering. Every cross-structure comparison
(torsion, contacts, masks) must go through here.
"""
from Bio import pairwise2
from Bio.PDB.Polypeptide import three_to_index, index_to_one


def chain_sequence(chain):
    seq, nums = "", []
    for r in chain:
        if r.id[0] != " ":
            continue
        try:
            seq += index_to_one(three_to_index(r.get_resname()))
        except Exception:
            continue
        nums.append(r.id[1])
    return seq, nums


def map_chains(ref_chain, mob_chain):
    """-> {mobile_resnum: ref_resnum} by sequence alignment."""
    sr, nr = chain_sequence(ref_chain)
    sm, nm = chain_sequence(mob_chain)
    aln = pairwise2.align.globalms(sr, sm, 2, -1, -10, -0.5, one_alignment_only=True)
    if not aln:
        return {}
    a, b = aln[0].seqA, aln[0].seqB
    out, i, j = {}, 0, 0
    for x, y in zip(a, b):
        if x != "-" and y != "-":
            out[nm[j]] = nr[i]
        if x != "-":
            i += 1
        if y != "-":
            j += 1
    return out
