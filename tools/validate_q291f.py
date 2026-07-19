"""Blind check against a KNOWN wet-lab answer: LacI Q291F responds to GlcNAc.

A design platform earns trust by ranking the real hit above random. This does NOT call PyRosetta -
it uses the fast geometry prefilter, so it can run without the WSL backend. It asks one question:
does placing Phe at 291 (filling toward the GlcNAc pocket) score better than WT and better than
random single mutants at the same position set?

Run: python tools/validate_q291f.py     (needs 1LBH downloadable; a few minutes)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pipeline.structure import (fetch_assembly, load_assembly, classify_chains, write_state,
                                resolve_design_positions)
from pipeline import pose as pose_mod
from pipeline.physallo import backend as pb

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "_q291f")
GLCNAC = "CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@@H]1O)CO)O)O"
IPTG = "CC(C)S[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)CO)O)O)O"


def build_scene():
    os.makedirs(OUT, exist_ok=True)
    holo = fetch_assembly("1LBH", OUT)
    ch = [c.id for c in classify_chains(load_assembly(holo))[0]][:2]
    x_i = write_state(holo, os.path.join(OUT, "X_I.pdb"), ch)
    x_i_ipt = write_state(holo, os.path.join(OUT, "X_I_lig_native.pdb"), ch,
                          drop_het=True, keep_ligand="IPT")
    pocket, _ = resolve_design_positions(holo, "IPT", dbd_range=[1, 61], cutoff=5.0)
    poses = pose_mod.generate_poses(GLCNAC, x_i, pocket, n_poses=8,
                                    native_pdb=x_i_ipt, native_resname="IPT", native_smiles=IPTG)
    x_i_lig = pose_mod.write_liganded_state(x_i, poses[0]["mol"],
                                            os.path.join(OUT, "X_I_lig.pdb"),
                                            resname="TGT", conf_id=poses[0].get("conf_id", -1))
    return x_i_lig, pocket


def score_variant(x_i_lig, pocket, mutations):
    """mutations: {resnum: 'PHE'}. -> fast in-house energy of that pocket, GlcNAc present."""
    ctx = pb.prepare(x_i_lig, list(pocket), ligand_resname="TGT", chain_id="A")
    efn = pb.make_energy_fn(ctx)
    from pipeline.physallo import rotamers as rot
    state = {}
    for p in ctx.positions:
        aa = mutations.get(p, ctx.wt[p])
        rots = rot.rotamers(aa, 20) or [()]
        state[p] = (aa, rots[0])
    return efn(state)


def main():
    x_i_lig, pocket = build_scene()
    print("pocket:", pocket)
    if 291 not in pocket:
        print("WARNING: 291 not in the auto-resolved pocket; scoring it anyway")

    wt = score_variant(x_i_lig, pocket, {})
    q291f = score_variant(x_i_lig, pocket, {291: "PHE"})
    print("\nWT pocket      : %.2f" % wt)
    print("Q291F (the hit): %.2f  (delta vs WT = %+.2f)" % (q291f, q291f - wt))

    # random single mutants at 291 as a null: is Phe special, or does anything help?
    rng = np.random.default_rng(0)
    aas = ["ALA", "SER", "THR", "VAL", "LEU", "ILE", "ASN", "ASP", "TYR", "TRP", "HIS", "GLN"]
    nulls = {a: score_variant(x_i_lig, pocket, {291: a}) for a in aas}
    ranked = sorted([("PHE", q291f)] + list(nulls.items()), key=lambda kv: kv[1])
    print("\n291 single-mutant ranking (lower = better pocket fit):")
    for i, (a, e) in enumerate(ranked, 1):
        mark = "  <- WET-LAB HIT" if a == "PHE" else ""
        print("  %2d. %s %8.2f%s" % (i, a, e, mark))
    rank_of_phe = [i for i, (a, _) in enumerate(ranked, 1) if a == "PHE"][0]
    print("\nQ291F rank: %d / %d   (a good platform puts the real hit near the top)"
          % (rank_of_phe, len(ranked)))


if __name__ == "__main__":
    main()
