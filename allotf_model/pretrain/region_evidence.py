"""Evidence-driven CANDIDATE regions from the QC-passed structures - NOT final regions. The pocket is
the objective anchor: residues within POCKET_CUTOFF of the bound effector across the holo ENSEMBLE,
scored by contact FREQUENCY (so the pocket is not defined by a single crystal form). Everything is
indexed by the canonical reference position (alignment, not resseq), so it is comparable across PDBs.

Output per scaffold is a table {canonical_index: {frequency, n_holo, evidence}}; the pocket, hinge and
DBD are finalised only after human curation against the original papers / SI, and DBD needs operator
structures (dna/ternary states), which are flagged when absent. Nothing here is written as a final
region.
"""
import glob
import os

import numpy as np
from Bio.PDB import MMCIFParser

from .structure_qc import _align_identity, AA3TO1, _IONS_ADD, MIN_IDENTITY, _MIN_CHAIN

_PARSER = MMCIFParser(QUIET=True)
POCKET_CUTOFF = 5.0


def _protein_and_effector(cif, sid):
    """-> (list of (aa1, [heavy-atom coords]) for the longest protein chain, effector-ligand heavy
    coords [K,3] or None, effector comp id)."""
    model = next(iter(_PARSER.get_structure(sid, cif)))
    chains, hets = {}, []
    for ch in model:
        res = []
        for r in ch:
            if r.id[0] == " " and r.get_resname() in AA3TO1 and r.has_id("CA"):
                coords = [a.coord for a in r if a.element != "H"]
                res.append((AA3TO1[r.get_resname()], np.array(coords)))
            elif r.id[0].startswith("H_"):
                nm = r.get_resname().strip().upper()
                if nm not in _IONS_ADD:
                    hets.append((nm, np.array([a.coord for a in r if a.element != "H"])))
        if len(res) >= _MIN_CHAIN:
            chains[ch.id] = res
    if not chains:
        return None, None, None
    main = max(chains.values(), key=len)
    eff = max(hets, key=lambda h: len(h[1])) if hets else (None, None)
    return main, eff[1], eff[0]


def pocket_evidence(scaffold_dir, sid, reference_seq, cutoff=POCKET_CUTOFF):
    """-> dict(pocket_frequency {canonical_index: freq}, n_holo, effectors, needs)."""
    holo = sorted(glob.glob(os.path.join(scaffold_dir, "holo", "*.cif")))
    counts = {}
    effectors, used = {}, 0
    for p in holo:
        main, eff_coords, eff_name = _protein_and_effector(p, sid)
        if main is None or eff_coords is None or len(eff_coords) == 0:
            continue
        seq = "".join(a for a, _ in main)
        ident, _, ref_to_seq = _align_identity(seq, reference_seq)
        if ident < MIN_IDENTITY:
            continue                                     # variant outlier - not this scaffold's cluster
        used += 1
        effectors[eff_name] = effectors.get(eff_name, 0) + 1
        for ci, pos in ref_to_seq.items():
            res_atoms = main[pos][1]
            if res_atoms.size == 0:
                continue
            dmin = np.linalg.norm(res_atoms[:, None, :] - eff_coords[None, :, :], axis=2).min()
            if dmin <= cutoff:
                counts[ci] = counts.get(ci, 0) + 1
    freq = {ci: round(c / used, 3) for ci, c in counts.items()} if used else {}
    dna = bool(glob.glob(os.path.join(scaffold_dir, "dna", "*.cif")) +
               glob.glob(os.path.join(scaffold_dir, "ternary", "*.cif")))
    return {"pocket_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
            "n_holo_used": used, "effectors": effectors,
            "pocket_high_confidence": sorted(ci for ci, f in freq.items() if f >= 0.5),
            "needs": ([] if dna else ["operator_structure_for_DBD"])
                     + (["single_effector_confirmation"] if len(effectors) > 3 else [])}
