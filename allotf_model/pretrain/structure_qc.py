"""Structure-level QC on the downloaded biological assemblies - the checks the manifest cannot answer.
Fail-closed, generic over any scaffold directory produced by fetch_structures.

Per scaffold:
  - reference = the longest protein chain observed across the ensemble (the WT scaffold);
  - every structure's main chain is ALIGNED to that reference (PDB numbering differs between entries,
    so canonical index = position in the reference, assigned by sequence alignment, not resseq);
  - sequence identity to the reference and engineered mutations (mismatch positions) are recorded;
  - functional oligomer = number of protein chains in the assembly;
  - coverage = fraction of the reference actually resolved;
  - ligand inventory per holo (comp ids) is reported for effector curation - the native-effector
    IDENTITY is confirmed later against papers/SI, not guessed here.
A structure fails (excluded) if it is not the same protein (identity < MIN_IDENTITY) or too incomplete
(coverage < MIN_COVERAGE).
"""
import glob
import os

from Bio.PDB import MMCIFParser
from Bio.Align import PairwiseAligner

_PARSER = MMCIFParser(QUIET=True)
MIN_IDENTITY = 0.90
MIN_COVERAGE = 0.50
_MIN_CHAIN = 30

AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
_IONS_ADD = {"HOH", "MG", "ZN", "NA", "K", "CL", "CA", "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "MN"}

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.open_gap_score = -10
_ALIGNER.extend_gap_score = -0.5
_ALIGNER.match_score = 2
_ALIGNER.mismatch_score = -1


def _chains(cif, sid):
    """-> list of (chain_id, seq_str, [(resseq, icode, aa1)]) for protein chains; + het comp ids."""
    model = next(iter(_PARSER.get_structure(sid, cif)))
    chains, hets = [], set()
    for ch in model:
        res = []
        for r in ch:
            if r.id[0] == " " and r.get_resname() in AA3TO1 and r.has_id("CA"):
                res.append((r.id[1], r.id[2] or " ", AA3TO1[r.get_resname()]))
            elif r.id[0].startswith("H_"):
                nm = r.get_resname().strip().upper()
                if nm not in _IONS_ADD:
                    hets.add(nm)
        if len(res) >= _MIN_CHAIN:
            chains.append((ch.id, "".join(a for _, _, a in res), res))
    return chains, hets


def _align_identity(seq, ref):
    """-> (identity over ALIGNED columns, coverage of the reference, {ref_index: seq_position}).
    Identity and coverage are separate: a fully-identical but partial structure has identity 1.0 and
    coverage < 1, not a low 'identity'."""
    aln = _ALIGNER.align(ref, seq)[0]
    match = aligned = 0
    ref_to_seq = {}
    for (r0, r1), (s0, s1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(r1 - r0):
            aligned += 1
            if ref[r0 + k] == seq[s0 + k]:
                match += 1
            ref_to_seq[int(r0 + k)] = int(s0 + k)         # aln indices are numpy ints; JSON-safe
    identity = match / aligned if aligned else 0.0
    return identity, aligned / len(ref), ref_to_seq


def qc_scaffold_structures(scaffold_dir, sid):
    apo = sorted(glob.glob(os.path.join(scaffold_dir, "apo", "*.cif")))
    holo = sorted(glob.glob(os.path.join(scaffold_dir, "holo", "*.cif")))
    parsed = {}
    for p in apo + holo:
        try:
            parsed[p] = _chains(p, sid)
        except Exception as e:
            parsed[p] = ("error", str(e))
    # reference = the DOMINANT sequence cluster, not the longest chain: a single UniProt can carry
    # several classes/engineered variants (e.g. TetR B vs D). Pick the sequence whose >=90%-identity
    # cluster covers the most structures, so outlier variants get flagged instead of hijacking the ref.
    from collections import Counter
    mains = [max(v[0], key=lambda c: len(c[1]))[1] for v in parsed.values() if v[0] != "error" and v[0]]
    if not mains:
        return {"error": "no protein chains parsed", "reference_len": 0}
    counts = Counter(mains)
    uniq = list(counts)
    ref = max(uniq, key=lambda cand: sum(counts[s] for s in uniq if _align_identity(s, cand)[0] >= MIN_IDENTITY))
    in_cluster = sum(counts[s] for s in uniq if _align_identity(s, ref)[0] >= MIN_IDENTITY)

    report = {"reference_len": len(ref), "n_structures": len(mains),
              "n_in_dominant_cluster": in_cluster, "n_variant_outliers": len(mains) - in_cluster,
              "structures": {}, "ligand_inventory": {}}
    for state, paths in (("apo", apo), ("holo", holo)):
        for p in paths:
            pid = os.path.splitext(os.path.basename(p))[0]
            v = parsed[p]
            if v[0] == "error":
                report["structures"][pid] = {"state": state, "passed": False, "reasons": ["parse_error"]}
                continue
            chains, hets = v
            if not chains:
                report["structures"][pid] = {"state": state, "passed": False,
                                             "reasons": ["no_protein_chain"]}
                continue
            main = max(chains, key=lambda c: len(c[1]))
            ident, coverage, r2s = _align_identity(main[1], ref)
            muts = [(ri, ref[ri], main[1][r2s[ri]]) for ri in sorted(r2s) if ref[ri] != main[1][r2s[ri]]]
            reasons = []
            if ident < MIN_IDENTITY:
                reasons.append("identity_%.2f<%.2f" % (ident, MIN_IDENTITY))
            if coverage < MIN_COVERAGE:
                reasons.append("coverage_%.2f<%.2f" % (coverage, MIN_COVERAGE))
            report["structures"][pid] = {
                "state": state, "passed": len(reasons) == 0, "reasons": reasons,
                "identity": round(ident, 3), "coverage": round(coverage, 3),
                "n_protein_chains": len(chains), "n_mutations": len(muts),
                "mutations": muts[:10], "ligands": sorted(hets)}
            if state == "holo":
                for h in hets:
                    report["ligand_inventory"][h] = report["ligand_inventory"].get(h, 0) + 1
    passed = [k for k, v in report["structures"].items() if v.get("passed")]
    report["n_passed"] = len(passed)
    return report
