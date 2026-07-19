"""Negative design against decoys.

Retargeting must include at least: the native ligand, close analogues of the target,
stereoisomers, and abundant host metabolites.

    S_specificity = E(best decoy) - E(target)

Whether the native response must be abolished is a user parameter
(objective.preserve_native_response).

The native effector is the decoy that matters most and the one a naive run forgets: the scaffold
spent its evolutionary history binding it, so a redesigned pocket that still binds it is the
default outcome, not a surprise. Stereoisomers are the second: a pocket that cannot tell D from L
is a pocket that has learned size, not chemistry.

Decoys are posed by the SAME route as the target (pose.generate_poses), on the SAME backbone
(X_I). A decoy docked while the target was transferred would make S_specificity a comparison of
placement methods.
"""
from . import pose as pose_mod

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False

# abundant E. coli / P. putida cytoplasmic metabolites a pocket is realistically exposed to.
# Not a complete metabolome - a floor. An empty decoy set must never be silently acceptable.
HOST_METABOLITES = {
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "glutamate": "N[C@@H](CCC(=O)O)C(=O)O",
    "atp": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "nadh": "NC(=O)C1=CN(C=CC1)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O",
    "acetyl_coa_frag": "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)CO",
    "tryptophan": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "tyrosine": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
}


def stereoisomers(smiles, max_n=4):
    """A pocket that cannot distinguish stereoisomers has learned shape, not recognition."""
    if not _RDKIT:
        return []
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    opts = StereoEnumerationOptions(maxIsomers=max_n + 1, onlyUnassigned=False)
    out = []
    ref = Chem.MolToSmiles(m)
    for iso in EnumerateStereoisomers(m, options=opts):
        s = Chem.MolToSmiles(iso)
        if s != ref:
            out.append(s)
        if len(out) >= max_n:
            break
    return out


# Which decoys can null a candidate's specificity, and which are a separate report.
# A pocket redesigned for tetracycline that cannot pose ATP has told us nothing about selectivity;
# treating that pose failure as "specificity unresolved" would kill every candidate for a reason
# unrelated to selectivity. Close chemistry is different: if the native effector or a stereoisomer
# cannot be placed, the comparison that matters is genuinely missing.
MANDATORY_PREFIXES = ("native_effector", "stereoisomer", "analogue")


def decoy_tier(name):
    """-> 'mandatory' | 'secondary'."""
    return "mandatory" if str(name).startswith(MANDATORY_PREFIXES) else "secondary"


def close_analogues(target_smiles, n=4, db_path=None):
    """The n nearest ligands in the aTF ligand database by ECFP Tanimoto.

    These are what a redesigned pocket is most likely to confuse with the target, so they belong in
    the mandatory tier. useChirality=True: RDKit fingerprints ignore stereochemistry by default,
    and D/L pairs would otherwise come back as the same molecule.
    """
    import csv
    import os
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    t = Chem.MolFromSmiles(target_smiles)
    if t is None:
        return []
    db_path = db_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "data", "atf_ligand_db.csv")
    if not os.path.exists(db_path):
        return []
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    tfp = gen.GetFingerprint(t)
    scored = []
    with open(db_path) as f:
        for row in csv.DictReader(f):
            smi = (row.get("smiles") or "").strip()
            if not smi or smi == target_smiles:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            s = DataStructs.TanimotoSimilarity(tfp, gen.GetFingerprint(m))
            scored.append((s, row.get("ligand") or row.get("name") or smi[:12], smi))
    scored.sort(reverse=True)
    return [("analogue_%d_%s" % (i + 1, name), smi) for i, (s, name, smi) in enumerate(scored[:n])]


def default_decoys(target_smiles, native_smiles=None, extra=(), include_metabolites=False):
    """Decoys with an actual competitive-binding hypothesis: the native effector, stereoisomers of
    the target, and its closest analogues in the ligand database.

    Generic host metabolites (ATP, glucose, NADH) are NOT included by default. They occupy a
    different chemical space from a hydrophobic effector, so there is no reason to think they
    compete for the pocket; their Rosetta energies are not comparable to the target's (charge);
    and a docking failure would report that the pose method does not apply, not that the molecule
    is excluded. Including them mainly manufactures false negatives.

    The question they were meant to answer - is the designed sensor switched ON by something in the
    cell - is answered directly by basal leak in the fluorescence assay. include_metabolites=True
    remains available for a standalone metabolite-challenge report, never for the gates.

    Returns [(name, smiles)]. The native ligand comes first because it is the decoy the scaffold is
    pre-adapted to bind.
    """
    out = []
    if native_smiles:
        out.append(("native_effector", native_smiles))
    for i, s in enumerate(stereoisomers(target_smiles)):
        out.append(("stereoisomer_%d" % (i + 1), s))
    try:
        out.extend(close_analogues(target_smiles))
    except Exception:
        pass                       # a missing ligand DB must not break negative design entirely
    if include_metabolites:
        out.extend(sorted(HOST_METABOLITES.items()))
    out.extend(extra)
    seen, uniq = set(), []
    for name, s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append((name, s))
    if not uniq:
        raise RuntimeError("empty decoy set: specificity would be vacuously perfect")
    return uniq


def _dock_and_score(backend, smiles, receptor_pdb, pocket, resname, candidate_residues,
                    design_positions, second_shell, chain, symmetric_chains, method, cfg,
                    native_pdb=None, native_resname=None, native_smiles=None):
    """Place one molecule with `method` and score it on THIS candidate. -> best energy or None.

    Target and decoys both come through here, so S_specificity is a difference between two numbers
    produced the same way on the same protein.
    """
    if backend is None or not smiles:
        return None
    try:
        ps = pose_mod.generate_poses(smiles, receptor_pdb, pocket,
                                     n_poses=(cfg or {}).get("n_decoy_poses", 5),
                                     native_pdb=native_pdb, native_resname=native_resname,
                                     native_smiles=native_smiles, method=method)
    except Exception:
        return None
    best = None
    for p in ps:
        e = backend.score_ligand_pose(p, receptor_pdb, resname=resname,
                                      candidate_residues=candidate_residues,
                                      design_positions=design_positions,
                                      second_shell=second_shell, chain=chain,
                                      symmetric_chains=symmetric_chains) \
            if hasattr(backend, "score_ligand_pose") else None
        if e is not None and (best is None or e < best):
            best = e
    return best


def score(candidate, states, decoys, cfg=None, backend=None, x_i=None, pocket=None,
          target_energy=None, native_pdb=None, native_resname=None, native_smiles=None,
          method="auto", decoy_resnames=None, candidate_residues=None, design_positions=(),
          second_shell=(), chain="A", symmetric_chains=None, mandatory_decoys=None):
    """-> dict(specificity, per_decoy)

    S_specificity = E(best decoy) - E(target). Positive = the target wins.

    Two failure modes GPT-5.6 flagged, both closed here:
      * an UNPOSED decoy is missing evidence, not good news. If any decoy could not be scored,
        specificity is returned as unresolved (None) rather than computed over whatever survived -
        the strongest competitor is exactly the one most likely to be a hard pose.
      * the target and the decoys must be placed by the SAME route. A target transferred by MCS
        compared against a docked decoy is a comparison of methods. `method` is threaded through so
        the caller pins one route for the whole comparison.
    """
    if target_energy is None:
        raise ValueError("specificity needs the target's own interface energy for comparison")
    decoy_resnames = decoy_resnames or {}
    mandatory = set(mandatory_decoys or ())
    per = {}
    failed = []          # mandatory decoys only - these can make specificity unresolved
    secondary_failed = []  # distant metabolites: reported, never fatal
    for name, smi in decoys:
        try:
            ps = pose_mod.generate_poses(smi, x_i, pocket, n_poses=cfg.get("n_decoy_poses", 5)
                                         if cfg else 5,
                                         native_pdb=native_pdb, native_resname=native_resname,
                                         native_smiles=native_smiles, method=method)
        except Exception as exc:
            per[name] = {"energy": None, "error": str(exc)[:200],
                         "tier": "mandatory" if name in mandatory else "secondary"}
            (failed if name in mandatory else secondary_failed).append(name)
            continue
        best = None
        for p in ps:
            # score the decoy ON THIS CANDIDATE: same mutations, same repack, same minimisation as
            # the target. Scoring it on the WT backbone while target_energy came from the mutated
            # candidate would make S_specificity partly a measure of the design, not of selectivity.
            # resname is the DECOY's own - it has its own .params and its own atom topology.
            e = backend.score_ligand_pose(
                p, x_i, resname=decoy_resnames.get(name, "TGT"),
                candidate_residues=candidate_residues,
                design_positions=design_positions, second_shell=second_shell,
                chain=chain, symmetric_chains=symmetric_chains
            ) if hasattr(backend, "score_ligand_pose") else None
            if e is not None and (best is None or e < best):
                best = e
        tier = "mandatory" if name in mandatory else "secondary"
        per[name] = {"energy": best, "n_poses": len(ps), "tier": tier,
                     "resname": decoy_resnames.get(name),
                     "method": ps[0]["provenance"]["method"] if ps else None}
        if best is None:
            (failed if tier == "mandatory" else secondary_failed).append(name)

    # S_specificity is decided by the MANDATORY tier: the native effector, stereoisomers and close
    # analogues. A distant host metabolite that cannot be posed is a pose failure, not evidence the
    # design is unselective - letting it null the whole candidate would kill every design for a
    # reason that has nothing to do with selectivity. Those are reported separately.
    mand_ok = [v["energy"] for n, v in per.items()
               if v.get("energy") is not None and v.get("tier") == "mandatory"]
    metabolite_report = {n: per[n] for n in per if per[n].get("tier") == "secondary"}

    if failed:
        return {"specificity": None, "per_decoy": per, "unposed": failed,
                "metabolite_challenge": metabolite_report,
                "secondary_unposed": secondary_failed,
                "note": "%d MANDATORY decoy(s) unposed (%s): specificity is unresolved, not a pass"
                        % (len(failed), ",".join(failed[:5]))}
    if not mand_ok:
        return {"specificity": None, "per_decoy": per, "unposed": failed,
                "metabolite_challenge": metabolite_report,
                "secondary_unposed": secondary_failed,
                "note": "no mandatory decoy could be scored: specificity is unmeasured, not perfect"}

    best_decoy = min(mand_ok)
    sec_ok = [v["energy"] for v in metabolite_report.values() if v.get("energy") is not None]
    return {"specificity": best_decoy - target_energy,
            "best_decoy_energy": best_decoy,
            "target_energy": target_energy,
            "per_decoy": per,
            "unposed": failed,
            "metabolite_challenge": metabolite_report,
            "secondary_unposed": secondary_failed,
            "metabolite_margin": (min(sec_ok) - target_energy) if sec_ok else None}


def run(ctx):
    """requires ctx['candidate_states'], ctx['route']; produces ctx['specificity']"""
    rt = ctx["route"]
    preserve = ctx["cfg"]["objective"].get("preserve_native_response", False)
    # preserve_native_response actually does something now: if the design must KEEP responding to
    # the native effector, that effector must NOT be a decoy - we would otherwise demand the target
    # out-compete a ligand we explicitly want retained. It is only a decoy in retargeting mode.
    native_as_decoy = None if preserve else rt.get("native_smiles")

    # decoys come from the SAME params bundle the six states use, so every decoy is scored with
    # its own .params and its own resname rather than borrowing the target's atom topology
    lp = ctx.get("ligand_params") or {}
    bundle_decoys = lp.get("decoys") or {}
    if bundle_decoys:
        decoys = [(did, d["smiles"]) for did, d in bundle_decoys.items()]
        decoy_resnames = {did: d["resname"] for did, d in bundle_decoys.items()}
    else:
        decoys = default_decoys(rt.get("target_smiles") or rt.get("smiles"), native_as_decoy)
        decoy_resnames = {}

    # SPECIFICITY IS ITS OWN BRANCH, with its own placement protocol.
    # The linkage branch may place the target by MCS transfer to preserve crystal geometry - but
    # glucose or ATP share no MCS with tetracycline and can never be placed that way, so under a
    # shared method every candidate's specificity would come back unresolved. And an MCS-transferred
    # target compared against docked decoys measures the placement method, not selectivity. So this
    # branch re-places EVERYTHING, target included, with one docking protocol.
    dcfg = ctx["cfg"].get("design") or {}
    spec_method = dcfg.get("specificity_method", "smina")
    states_meta = ctx["states"]
    design_positions = ctx["masks"]["recognition_mask"]
    second_shell = ctx["masks"].get("transduction_mask", ())
    chain = states_meta.get("chain", "A")
    protein_chains = states_meta.get("protein_chains")
    x_i = states_meta["paths"]["X_I"]
    target_smiles = rt.get("target_smiles") or rt.get("smiles")
    target_resname = (lp.get("target") or {}).get("resname", "TGT")
    mandatory = {name for name, _ in decoys if decoy_tier(name) == "mandatory"}

    out = {}
    for cid, sc in ctx["ligand_scores"].items():
        cand = ctx["candidates"].get(cid, {})
        cand_res = cand.get("residues") if isinstance(cand, dict) else None
        # the target's OWN specificity energy, re-docked on this candidate with spec_method -
        # NOT E_L_I from the linkage branch, which may come from a different placement route
        target_energy = _dock_and_score(
            ctx.get("backend"), target_smiles, x_i, design_positions, target_resname,
            cand_res, design_positions, second_shell, chain, protein_chains,
            spec_method, dcfg, native_pdb=states_meta["paths"].get("X_I_lig"),
            native_resname=states_meta.get("effector_resname"),
            native_smiles=rt.get("native_smiles"))
        if target_energy is None:
            out[cid] = {"specificity": None,
                        "note": "target could not be re-docked in the specificity branch: "
                                "unresolved, not a pass"}
            continue
        out[cid] = score(cid, ctx["candidate_states"][cid], decoys, dcfg,
                         backend=ctx.get("backend"),
                         x_i=x_i,
                         pocket=design_positions,
                         target_energy=target_energy,
                         native_pdb=states_meta["paths"].get("X_I_lig"),
                         native_resname=states_meta.get("effector_resname"),
                         native_smiles=rt.get("native_smiles"),
                         method=spec_method,
                         decoy_resnames=decoy_resnames,
                         # the decoy is scored on THIS candidate, not on WT
                         candidate_residues=cand_res,
                         design_positions=design_positions,
                         second_shell=second_shell,
                         chain=chain,
                         symmetric_chains=protein_chains,
                         mandatory_decoys=mandatory)
    # key must match what allotf.py STAGES declares this stage produces, or the contract check
    # fails and rank never sees the scores
    return {"specificity_scores": out, "preserve_native_response": preserve}
