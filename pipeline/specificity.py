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


def default_decoys(target_smiles, native_smiles=None, extra=()):
    """native ligand + stereoisomers of the target + abundant host metabolites.

    Returns [(name, smiles)]. The native ligand comes first because it is the decoy the scaffold
    is pre-adapted to bind.
    """
    out = []
    if native_smiles:
        out.append(("native_effector", native_smiles))
    for i, s in enumerate(stereoisomers(target_smiles)):
        out.append(("stereoisomer_%d" % (i + 1), s))
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


def score(candidate, states, decoys, cfg=None, backend=None, x_i=None, pocket=None,
          target_energy=None, native_pdb=None, native_resname=None, native_smiles=None,
          method="auto"):
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
    per = {}
    failed = []
    for name, smi in decoys:
        try:
            ps = pose_mod.generate_poses(smi, x_i, pocket, n_poses=cfg.get("n_decoy_poses", 5)
                                         if cfg else 5,
                                         native_pdb=native_pdb, native_resname=native_resname,
                                         native_smiles=native_smiles, method=method)
        except Exception as exc:
            per[name] = {"energy": None, "error": str(exc)[:200]}
            failed.append(name)
            continue
        best = None
        for p in ps:
            e = backend.score_ligand_pose(p, x_i) if hasattr(backend, "score_ligand_pose") else None
            if e is not None and (best is None or e < best):
                best = e
        per[name] = {"energy": best, "n_poses": len(ps),
                     "method": ps[0]["provenance"]["method"] if ps else None}
        if best is None:
            failed.append(name)

    ok = [v["energy"] for v in per.values() if v.get("energy") is not None]
    if failed:
        # unresolved, not perfect: the unscored decoys could be the ones that beat the target.
        return {"specificity": None, "per_decoy": per, "unposed": failed,
                "note": "%d decoy(s) unposed (%s): specificity is unresolved, not a pass"
                        % (len(failed), ",".join(failed[:5]))}
    if not ok:
        return {"specificity": None, "per_decoy": per, "unposed": failed,
                "note": "no decoy could be scored: specificity is unmeasured, not perfect"}
    best_decoy = min(ok)
    return {"specificity": best_decoy - target_energy,
            "best_decoy_energy": best_decoy,
            "target_energy": target_energy,
            "per_decoy": per,
            "unposed": failed}


def run(ctx):
    """requires ctx['candidate_states'], ctx['route']; produces ctx['specificity']"""
    rt = ctx["route"]
    preserve = ctx["cfg"]["objective"].get("preserve_native_response", False)
    # preserve_native_response actually does something now: if the design must KEEP responding to
    # the native effector, that effector must NOT be a decoy - we would otherwise demand the target
    # out-compete a ligand we explicitly want retained. It is only a decoy in retargeting mode.
    native_as_decoy = None if preserve else rt.get("native_smiles")
    decoys = default_decoys(rt["target_smiles"], native_as_decoy)
    method = ctx["poses"]["method"] if ctx.get("poses") else "auto"
    out = {}
    for cid, sc in ctx["ligand_scores"].items():
        out[cid] = score(cid, ctx["candidate_states"][cid], decoys, ctx["cfg"].get("design"),
                         backend=ctx.get("backend"),
                         x_i=ctx["states"]["paths"]["X_I"],
                         pocket=ctx["masks"]["recognition_mask"],
                         target_energy=sc.get("E_L_I"),
                         native_pdb=ctx["states"]["paths"].get("X_I_lig"),
                         native_resname=ctx["states"].get("effector_resname"),
                         native_smiles=rt.get("native_smiles"),
                         method=method)
    return {"specificity": out, "preserve_native_response": preserve}
