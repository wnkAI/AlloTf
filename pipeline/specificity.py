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


# Weights for the combined graph similarity used to CHOOSE decoys. Graph similarity is not binding
# selectivity - it only answers "which molecules are the dangerous chemical neighbours worth
# comparing against". Selectivity itself is decided by structure-based scoring on the same
# candidate protein.
W_ECFP, W_MCS, W_FEAT = 0.60, 0.25, 0.15
MIN_ECFP, MIN_MCS = 0.35, 0.50          # pre-filter: either route into "close enough to matter"
# ...but neither metric alone may carry a molecule through. Measured on tetracycline, a
# cholesterol-CoA derivative passed on MCS=0.50 with ECFP=0.09: a large molecule can accumulate
# ring-system overlap while sharing no local chemistry at all. The combined score has to agree.
MIN_COMBINED = 0.30


def _ecfp_similarity(a, b):
    """ECFP4 Tanimoto. useChirality=True is not optional: RDKit fingerprints ignore stereochemistry
    by default, so enantiomers would come back identical - and a stereoisomer is precisely the decoy
    a redesigned pocket is most likely to confuse with the target."""
    from rdkit.Chem import AllChem, DataStructs
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(a), gen.GetFingerprint(b)))


def _mcs_similarity(a, b, timeout=5):
    """Shared-core coverage: 2*N_MCS / (N_a + N_b) over heavy atoms.

    Catches the case ECFP under-rates - two molecules built on the same scaffold that differ only
    in substituents (tetracycline vs doxycycline).
    """
    from rdkit.Chem import rdFMCS
    na, nb = a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms()
    if not na or not nb:
        return 0.0
    try:
        res = rdFMCS.FindMCS([a, b], timeout=timeout, ringMatchesRingOnly=True,
                             completeRingsOnly=True)
    except Exception:
        return 0.0
    if res.canceled or not res.numAtoms:
        return 0.0
    return float(2 * res.numAtoms) / float(na + nb)


def _functional_feature_count_similarity(a, b):
    """COUNT overlap of functional features: donors, acceptors, aromatic ATOMS, +/- centres,
    hydrophobic carbons.

    Named for what it does. This is NOT a pharmacophore: it compares how MANY of each feature the
    two molecules have, not their 3D distances, spatial arrangement, ring positions or topological
    relationships - and "aromatic" here counts atoms, not rings. It is a 15% auxiliary term that
    guards one specific failure of ECFP and MCS (a similar carbon skeleton whose interacting groups
    differ); ECFP and MCS carry the actual selection.
    """
    from rdkit import Chem
    feats = {
        "donor": Chem.MolFromSmarts("[$([N;!H0]),$([O,S;H1,H2])]"),
        "acceptor": Chem.MolFromSmarts("[$([O,S;v2]),$([N;v3;!$([N+]);!$(n)]),$([n;+0])]"),
        "aromatic": Chem.MolFromSmarts("a"),
        "positive": Chem.MolFromSmarts("[+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))])]"),
        "negative": Chem.MolFromSmarts("[-,$(C(=O)[O;H1,-]),$(S(=O)(=O)[O;H1,-]),$(P(=O)[O;H1,-])]"),
        "hydrophobe": Chem.MolFromSmarts("[C;!$(C=O);!$(C#N)]"),
    }
    inter = union = 0
    for patt in feats.values():
        if patt is None:
            continue
        ca = len(a.GetSubstructMatches(patt))
        cb = len(b.GetSubstructMatches(patt))
        inter += min(ca, cb)
        union += max(ca, cb)
    return float(inter) / float(union) if union else 0.0


def graph_similarity(a, b):
    """-> (combined, {'ecfp','mcs','pharm'}). Combined = 0.60*ECFP + 0.25*MCS + 0.15*feature-count.

    Three complementary views, because each alone misleads: ECFP is size-sensitive and over-
    penalises substitutions on a shared core; MCS can score two molecules with very different
    charge states as near-identical; pharmacophore alone ignores topology.
    """
    e = _ecfp_similarity(a, b)
    m = _mcs_similarity(a, b)
    p = _functional_feature_count_similarity(a, b)
    return W_ECFP * e + W_MCS * m + W_FEAT * p, {"ecfp": e, "mcs": m, "feat": p}


def _load_analogue_library(db_path):
    """-> [(label, smiles)] from an .sdf or a .csv with a smiles column.

    Priority is decided by the caller: a TARGET-SPECIFIC library the user supplied beats a
    project-maintained one, which beats atf_ligand_db.csv. That last one is a fallback, not a
    source of analogues - it is a catalogue of NATIVE aTF EFFECTORS, so for tetracycline it holds
    exactly one tetracycline and no doxycycline or minocycline. Each target deserves an explicit,
    auditable challenge set rather than whatever the effector catalogue happens to contain.
    """
    import csv
    import os
    from rdkit import Chem

    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    out = []
    if db_path.lower().endswith((".sdf", ".mol")):
        for i, m in enumerate(Chem.SDMolSupplier(db_path)):
            if m is None:
                continue
            label = (m.GetProp("_Name") if m.HasProp("_Name") else "") or "lib_%d" % (i + 1)
            out.append((label.strip().replace(" ", "_"), Chem.MolToSmiles(m)))
        return out
    with open(db_path) as f:
        for row in csv.DictReader(f):
            smi = (row.get("smiles") or row.get("SMILES") or "").strip()
            if not smi:
                continue
            label = (row.get("name") or row.get("ligand") or row.get("native_ligand")
                     or smi[:12]).strip().replace(" ", "_")
            out.append((label, smi))
    return out


def analogue_library_path(cfg=None, root=None):
    """User-supplied target library > project library > native-effector catalogue (fallback)."""
    import os
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user = (cfg or {}).get("analogue_library")
    if user:
        if not os.path.exists(user):
            raise FileNotFoundError(
                "specificity.analogue_library points at %s, which does not exist. A declared "
                "challenge set that silently falls back would produce a specificity number "
                "computed against the wrong molecules." % user)
        return user, "user_target_library"
    project = os.path.join(root, "data", "target_analogues.sdf")
    if os.path.exists(project):
        return project, "project_library"
    return os.path.join(root, "data", "atf_ligand_db.csv"), "native_effector_catalogue_fallback"


def close_analogues(target_smiles, n=4, db_path=None, cfg=None):
    """The n most dangerous chemical neighbours of the target in the aTF ligand database.

    Pre-filtered on ECFP >= 0.35 OR MCS >= 0.50 (either route into "close enough to matter"), then
    ranked by the combined graph score. Exact duplicates - salts, solvates, plain protonation
    variants - are dropped: they are the same molecule and would only inflate the decoy set.

    This selects WHAT to compare. It never decides selectivity, which is measured by docking and
    scoring each of these on the same candidate protein as the target.
    """
    import os
    from rdkit import Chem

    t = Chem.MolFromSmiles(target_smiles)
    if t is None:
        raise ValueError("target SMILES could not be parsed for analogue selection: %s"
                         % target_smiles)
    if db_path is None:
        db_path, source = analogue_library_path(cfg)
    else:
        source = "explicit"
    library = _load_analogue_library(db_path)

    from utils.standardize_ligand import same_molecule
    scored = []
    for label, smi in library:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            if same_molecule(smi, target_smiles):          # salt / solvate / protonation variant
                continue
        except Exception:
            if smi == target_smiles:
                continue
        combined, parts = graph_similarity(t, m)
        if (parts["ecfp"] < MIN_ECFP and parts["mcs"] < MIN_MCS) or combined < MIN_COMBINED:
            continue
        scored.append((combined, label, smi, parts))
    scored.sort(key=lambda x: -x[0])
    picked = [("analogue_%d_%s" % (i + 1, name), smi)
              for i, (c, name, smi, parts) in enumerate(scored[:n])]
    if not picked and source == "native_effector_catalogue_fallback":
        # say it out loud: the fallback is a catalogue of native effectors, not an analogue set
        print("  [specificity] no close analogue found in %s (%s). The mandatory tier is then "
              "native effector + stereoisomers only. Supply specificity.analogue_library with a "
              "target-specific challenge set to strengthen it." % (os.path.basename(db_path), source))
    return picked


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
    # NOT wrapped in a bare except: close analogues are a MANDATORY tier, and swallowing an RDKit
    # or database error here would return a confident numeric specificity computed without the very
    # decoys that matter most - a silent downgrade dressed as a result.
    out.extend(close_analogues(target_smiles))
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
        # keep the LOGICAL name (native_effector / stereoisomer_2 / analogue_1_...), not the bundle
        # key (D01). decoy_tier reads the name, so keying on D01 would classify every decoy as
        # secondary, leave the mandatory tier empty and return specificity=None for every candidate.
        decoys = [(d.get("decoy_name") or d.get("role") or did, d["smiles"])
                  for did, d in bundle_decoys.items()]
        decoy_resnames = {(d.get("decoy_name") or d.get("role") or did): d["resname"]
                          for did, d in bundle_decoys.items()}
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
