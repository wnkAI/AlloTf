"""LigandMPNN Autopsy - six probes that ask whether the model actually uses ligand chemistry,
and whether its scores track function at all.

Premise: the model's own ablation reports that removing ligand ELEMENT TYPE barely changes
near-ligand sequence recovery. If erasing chemical identity does not move the metric, the metric
is not measuring chemistry. Everything below is designed to find out what it IS measuring.

Each probe returns dict(value=..., verdict=..., detail=...). Thresholds are declared up front so
the conclusion cannot be rationalised after seeing the number.
"""
import numpy as np

# ---- decision thresholds, fixed BEFORE running -----------------------------------------
THR = dict(
    # probe 1: KL between P(S|true ligand) and P(S|ablated ligand), averaged over pocket positions
    deletion_kl_low=0.10,      # < this  -> ligand barely matters (model runs on backbone alone)
    deletion_kl_high=0.50,     # > this  -> ligand strongly drives the design
    # probe 2: fraction of pocket positions whose argmax changes when only CHEMISTRY is edited
    chem_change_low=0.05,      # < 5% of positions react -> blind to chemistry, sees only shape
    chem_change_high=0.25,
    # probe 3: confidence must fall as the pose is corrupted; a flat curve is dangerous
    pose_slope_min=0.05,       # d(mean logP)/d(Angstrom) - flatter than this = pose-insensitive
    # probe 4: enantiomer / epimer discrimination
    stereo_kl_min=0.05,
    # probe 5: agreement with ProteinMPNN (no ligand at all) on pocket positions
    pmpnn_agree_high=0.80,     # > 80% identical argmax -> ligand channel is near-redundant
    # probe 6: does the score separate real functional labels?
    func_auc_min=0.60,         # <= 0.6 -> effectively uninformative about function
)


def _kl(p, q, eps=1e-9):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    p = p / p.sum(-1, keepdims=True); q = q / q.sum(-1, keepdims=True)
    return float((p * np.log2(p / q)).sum(-1).mean())


def probe1_ligand_deletion(backend, state, poses, pocket, cfg):
    """Same backbone, four ligand conditions: true / deleted / random ligand / shuffled coords.
    If P(S|X,L) hardly moves, the model is designing from the backbone, not the ligand.

    TODO: needs backend.residue_probs(state, ligand, pocket) -> (n_pos, 20)
    """
    raise NotImplementedError("needs LigandMPNN installed; see autopsy/README.md")


def probe2_atom_identity(backend, state, pose, pocket, cfg):
    """Freeze every coordinate; change ONLY the chemistry:
        C -> N/O, neutral -> charged, donor -> acceptor, single -> aromatic.
    A model that understands chemistry must react. One that sees only shape cannot.
    This is the direct test of the paper's own element-ablation result.
    """
    raise NotImplementedError


def probe3_pose_perturbation(backend, state, pose, pocket, cfg,
                             shifts=(0.25, 0.5, 1.0, 2.0)):
    """Translate/rotate the ligand by increasing amounts.
    Wanted: robust to small shifts, clearly less confident on large ones.
    Dangerous: high confidence on a wrong pose - it means the pipeline can be confidently wrong.
    """
    raise NotImplementedError


def probe4_stereochemistry(backend, state, pocket, cfg, stereo_set):
    """Correct isomer vs enantiomer vs epimer vs a near-isosteric analogue.
    Sugars/polyols are the acid test: identical formula, different stereocentres.
    Reminder: our own retrieval collapsed all pentoses to Tanimoto 1.0 until chirality was
    switched on - the same blindness would be fatal here.
    """
    raise NotImplementedError


def probe5_vs_proteinmpnn(lig_backend, prot_backend, state, pose, pocket, cfg):
    """Compare pocket-position outputs of LigandMPNN vs ProteinMPNN (which never sees the ligand).
    High agreement => the ligand channel adds little; the 'ligand-aware' model is largely a
    protein-context model wearing a ligand hat.
    """
    raise NotImplementedError


DESIGN_POS = [67, 70, 74, 78, 89, 92, 93, 96, 110, 113, 114]   # TtgR, verified via L92/L113


def probe6_function_vs_recovery(lig_backend, prot_backend, ttgr_state, ligand_poses, tables, cfg):
    """Does the score track FUNCTION? (recovery-style metrics have never been shown to.)

    CORRECTION - this probe is NOT structure-free. LigandMPNN consumes backbone AND ligand
    coordinates, so it needs:
        * one canonical TtgR backbone (same construct, same preparation for every variant), and
        * a UNIFORM, comparable pose for each of the 9 ligands.
    No new experimental structures are required, but without uniform poses this measures POSE
    QUALITY DIFFERENCES BETWEEN LIGANDS, not model ability. Build all 9 poses the same way from
    the same template and record how they were made.

    Score ONLY the 11 design positions:
        LL_pocket(S, L) = sum_{i in DESIGN_POS} log P(a_i | X, L)
    A whole-sequence log-likelihood is swamped by the ~200 residues that are identical across all
    16,191 variants.

    Compare WT-relative, so the question is "does the model prefer this variant over WT for THIS
    ligand":
        dLL(S, L) = LL_pocket(S, L) - LL_pocket(WT, L)

    THE KEY METRIC is not the raw score but the ligand-conditioning increment over a model that
    never sees the ligand at all:
        I(S, L) = LL_LigandMPNN(S | X, L) - LL_ProteinMPNN(S | X)
    If I(S, L) carries no functional signal, the ligand channel contributes nothing and the
    'ligand-aware' model is a protein-context model wearing a ligand hat.

    Labels (from our QC tables; do NOT call these binder-only - F-score cannot prove binding):
        responsive sensor / repression-competent non-responder / repression-defective
    Report: AUC(sensor vs rest) and Spearman(dLL, F-score) for BOTH dLL and I(S,L).

    Reference point already in hand: our own per-ligand sequence model reaches rho ~ 0.495 on the
    same data using nothing but 11-position one-hot. A PDB-scale 'ligand-aware' network that
    cannot beat that has no claim on the ranking.
    """
    raise NotImplementedError


def verdict(results):
    """Fold the six probes into the three possible conclusions.

      A  ligand chemistry genuinely drives it  -> keep encoder+decoder, bolt on multistate physics
      B  it runs on geometry/backbone          -> keep protein encoder, REWRITE ligand encoder
                                                  and the protein-ligand interaction module
      C  scores unrelated to function          -> demote to a plausibility filter, keep it out of
                                                  the final ranking entirely
    """
    kl = results.get("probe1", {}).get("value")
    chem = results.get("probe2", {}).get("value")
    agree = results.get("probe5", {}).get("value")
    auc = results.get("probe6", {}).get("value")

    if auc is not None and auc <= THR["func_auc_min"]:
        return "C", ("score does not track function (AUC=%.2f <= %.2f): plausibility filter only, "
                     "must not influence final ranking" % (auc, THR["func_auc_min"]))
    weak_ligand = ((kl is not None and kl < THR["deletion_kl_low"]) or
                   (chem is not None and chem < THR["chem_change_low"]) or
                   (agree is not None and agree > THR["pmpnn_agree_high"]))
    if weak_ligand:
        return "B", ("ligand channel contributes little (KL=%s, chem-reactivity=%s, "
                     "ProteinMPNN agreement=%s): keep the protein encoder, rewrite the ligand "
                     "encoder and the protein-ligand interaction module" % (kl, chem, agree))
    return "A", "ligand chemistry genuinely drives the design: keep it as the sequence prior"
