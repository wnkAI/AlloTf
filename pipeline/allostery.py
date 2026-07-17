"""Extract the SYSTEM-SPECIFIC native allosteric template. No training, no MD.

Compares the scaffold's own experimental states X_D -> X_I and emits:
  * per-residue torsion redistribution   (utils/torsion.py, circular statistics)
  * contact / hbond / salt-bridge change (utils/contacts.py)
  * DBD output geometry change
  * allosteric path: pocket -> second shell -> hinge/dimer -> DBD (graph search)

and the three masks that constrain Design:
  recognition_mask   ligand first shell, free to redesign
  transduction_mask  second shell / hinge, limited compensatory mutations
  protected_mask     DNA recognition helix, dimer core, known allosteric nodes, fold core

Verified on TetR: torsion redistribution alone recovers alpha6 (103-109) and alpha4 (49, 62),
but also flags terminal residues (5, 205) that merely wobble. Terminal/surface residues that are
NOT on the pocket->DBD path must be discarded as noise. That noise filter is the reason this
module does a path search at all: torsion signal answers "did it move", the path answers "can it
possibly transmit". A residue needs both.

ENSEMBLE CAVEAT: the JS divergence term needs >=2 structures per state. With a single X_D and a
single X_I we can only report the mean shift d_circ, and js_* is emitted as None rather than a
fabricated 0.0 - a lone pair of crystal structures cannot tell a real redistribution from
crystal-packing noise. n_apo/n_holo are recorded so downstream never forgets which regime it is in.
"""
import os
import json
import warnings
from collections import deque

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from utils.torsion import torsions, circ_mean, circ_diff, circ_js, ANGLES
from utils.contacts import contact_map, heavy, CONTACT
from .structure import load_assembly

warnings.simplefilter("ignore", PDBConstructionWarning)
PARSER = PDBParser(QUIET=True)

POCKET_CUT = 4.5        # ligand first shell
SECOND_SHELL_CUT = 8.0  # around the pocket
PATH_SLACK = 1          # residues this far above the shortest pocket->DBD path still count
TERMINAL_MARGIN = 3     # residues this close to a chain end are wobble suspects
BACKBONE_W = 1.0        # phi/psi weight in the torsion signal
CHI_W = 0.7


def _model(pdb_path):
    """ALWAYS via load_assembly, never next(iter(structure)).

    Biological assemblies store symmetry copies as separate MODEL blocks that reuse chain IDs
    (1QPI: 2 MODELs, both chains named 'A'). Taking the first model silently collapses the
    functional dimer to a monomer - the contact graph then misses the entire dimer interface and
    DBD-DBD geometry, the actual readout, becomes undefined. Nothing crashes; you just compute
    confident nonsense from half a protein. This cost us a full debugging cycle once already.
    """
    return load_assembly(pdb_path)


def _protein_chains(model):
    out = []
    for ch in model:
        n = sum(1 for r in ch if r.id[0] == " " and r.has_id("CA"))
        if n > 20:
            out.append((ch, n))
    out.sort(key=lambda t: -t[1])
    return [c for c, _ in out]


def _main_chain(pdb_path):
    ch = _protein_chains(_model(pdb_path))
    if not ch:
        raise ValueError("no protein chain in %s" % pdb_path)
    return ch[0]


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def ligand_atoms(model, ligand_resname):
    at = []
    for ch in model:
        for r in ch:
            if r.id[0] != " " and r.get_resname().strip() == ligand_resname:
                at.extend(heavy(r))
    return at


def pocket_residues(pdb_path, ligand_resname, cutoff=POCKET_CUT):
    """Residues whose heavy atoms come within `cutoff` of the effector. This is the recognition
    shell - the only set Design is free to rewrite."""
    model = _model(pdb_path)
    lig = ligand_atoms(model, ligand_resname)
    if not lig:
        raise ValueError("effector %s not found in %s - cannot define a pocket"
                         % (ligand_resname, pdb_path))
    L = np.array([a.coord for a in lig])
    out = set()
    for ch in _protein_chains(model):
        for r in ch:
            if r.id[0] != " ":
                continue
            A = np.array([a.coord for a in heavy(r)])
            if len(A) and np.linalg.norm(A[:, None, :] - L[None, :, :], axis=2).min() < cutoff:
                out.add(r.id[1])
    return out


def _ensemble_torsions(paths):
    """-> {resnum: {angle: [values across the ensemble]}}, plus n parsed."""
    acc, n = {}, 0
    for p in paths:
        t = torsions(p)
        if not t:
            continue
        n += 1
        for rn, ang in t.items():
            for a, v in ang.items():
                acc.setdefault(rn, {}).setdefault(a, []).append(v)
    return acc, n


def torsion_signal(apo_paths, holo_paths):
    """Per-residue apo->holo dihedral redistribution.

    -> {resnum: {'torsion_signal': float, 'n_apo': int, 'n_holo': int, 'd_<ang>':, 'js_<ang>':}}
    js_* is None when either ensemble has <2 members: with one structure per state a histogram
    divergence is not estimable, and reporting 0.0 would read as "no change".
    """
    A, na = _ensemble_torsions(_as_list(apo_paths))
    H, nh = _ensemble_torsions(_as_list(holo_paths))
    if not A or not H:
        return {}
    have_ensemble = na >= 2 and nh >= 2
    out = {}
    for rn in sorted(set(A) & set(H)):
        row = {"n_apo": na, "n_holo": nh, "ensemble": have_ensemble}
        sig = 0.0
        for a in ANGLES:
            va = A[rn].get(a) or []
            vh = H[rn].get(a) or []
            if not va or not vh:
                continue
            row["d_" + a] = round(circ_diff(circ_mean(vh), circ_mean(va)), 1)
            w = BACKBONE_W if a in ("phi", "psi") else CHI_W
            if len(va) >= 2 and len(vh) >= 2:
                js = circ_js(va, vh)
                row["js_" + a] = round(js, 3)
                sig += w * js
            else:
                row["js_" + a] = None
                # single pair of structures: fall back to the normalised mean shift so the
                # signal is still defined, but flag the regime so it is never mistaken for JS.
                sig += w * (row["d_" + a] / 180.0)
        row["torsion_signal"] = round(sig, 3)
        out[rn] = row
    return out


def _adjacency(chain):
    idx, C = contact_map(chain)
    pos = {rn: i for i, rn in enumerate(idx)}
    adj = {rn: set() for rn in idx}
    n = len(idx)
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                adj[idx[i]].add(idx[j])
                adj[idx[j]].add(idx[i])
    # sequence neighbours are always connected: contact_map skips |i-j|<2 on purpose
    for a, b in zip(idx, idx[1:]):
        if b - a == 1:
            adj[a].add(b)
            adj[b].add(a)
    return idx, adj, pos


def _bfs(adj, sources):
    d = {s: 0 for s in sources if s in adj}
    q = deque(d)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def _path_from_adj(adj, idx, pk, db, slack):
    """Core BFS geometry: residues on a near-shortest pocket->DBD path. Reused by the null model."""
    pk = {r for r in pk if r in adj}
    db = {r for r in db if r in adj}
    if not pk or not db:
        return set(), {}, {}, None
    dp = _bfs(adj, pk)
    dd = _bfs(adj, db)
    reach = [dp[r] for r in db if r in dp]
    if not reach:
        return set(), dp, dd, None
    shortest = min(reach)
    on = {r for r in idx if r in dp and r in dd and dp[r] + dd[r] <= shortest + slack}
    return on, dp, dd, shortest


def allosteric_path(chain, pocket, dbd, slack=PATH_SLACK):
    """Which residues lie on a near-shortest pocket -> DBD contact path.

    RAW geometry only. On its own this is nearly vacuous: in a 4.5 A contact graph of a compact
    ~200-residue protein, half the fold sits on some shortest-plus-one path, so being "on-path"
    barely distinguishes a signalling residue from background. Independently flagged by three
    reviewers. Use calibrated_path() for the filter that actually means something; this stays as
    the geometric primitive it computes.
    -> (on_path: set, d_pocket: dict, d_dbd: dict, shortest: int|None)
    """
    idx, adj, _ = _adjacency(chain)
    return _path_from_adj(adj, idx, {r for r in pocket}, {r for r in dbd}, slack)


def calibrated_path(chain, pocket, dbd, n_null=300, slack=PATH_SLACK, seed=0xA11051, alpha=0.05):
    """On-path, but only where being on-path is NOT what a random pocket/DBD pair would give.

    The raw path is vacuous because almost every residue is on SOME near-shortest path. The fix a
    reviewer panel converged on: a scaffold-specific null. Draw many random label sets of the same
    sizes as the real pocket and DBD, recompute the path each time, and keep a residue only if it
    sits on the real path far more often than random labels put it there (empirical p < alpha).

    This turns "geometrically reachable" into "specifically coupled to THIS pocket and THIS DBD".

    -> dict(
        significant   : set of residues that survive the null,
        on_path_raw   : the uncalibrated set (for comparison / logging),
        pval          : {resnum: fraction of null runs that put it on-path},
        real_size, null_size_mean, null_size_sd,
        enriched      : real path smaller than the null mean? (a specific path is TIGHTER),
        slack_sensitivity : {slack: |on_path|} for slack in 0,1,2  -> is the filter parameter-fragile?
    )
    Deterministic: the RNG is seeded, so the same structure gives the same calibration.
    """
    idx, adj, _ = _adjacency(chain)
    nodes = [r for r in idx if r in adj]
    pk = [r for r in pocket if r in adj]
    db = [r for r in dbd if r in adj]
    on_real, _, _, real_shortest = _path_from_adj(adj, idx, pk, db, slack)
    if not on_real or not pk or not db:
        return {"significant": set(), "on_path_raw": set(on_real), "pval": {},
                "real_size": len(on_real), "null_size_mean": None, "null_size_sd": None,
                "enriched": None, "slack_sensitivity": {},
                "note": "no pocket->DBD path: cannot calibrate"}

    rng = np.random.default_rng(seed)
    npk, ndb = len(pk), len(db)
    node_arr = np.array(nodes)
    count = {r: 0 for r in nodes}
    sizes = []
    for _ in range(n_null):
        perm = rng.permutation(node_arr)
        rpk = set(perm[:npk].tolist())
        rdb = set(perm[npk:npk + ndb].tolist())   # disjoint from rpk by construction
        on_n, _, _, _ = _path_from_adj(adj, idx, rpk, rdb, slack)
        sizes.append(len(on_n))
        for r in on_n:
            count[r] += 1
    sizes = np.array(sizes, dtype=float)
    pval = {r: count[r] / n_null for r in nodes}
    # keep real on-path residues that random labels rarely put on-path
    significant = {r for r in on_real if pval.get(r, 1.0) < alpha}
    slack_sens = {s: len(_path_from_adj(adj, idx, pk, db, s)[0]) for s in (0, 1, 2)}
    return {"significant": significant,
            "on_path_raw": set(on_real),
            "pval": pval,
            "real_size": len(on_real),
            "null_size_mean": float(sizes.mean()),
            "null_size_sd": float(sizes.std()),
            "enriched": bool(len(on_real) < sizes.mean()),
            "shortest": real_shortest,
            "slack_sensitivity": slack_sens}


DNA_BASES = {"DA", "DT", "DG", "DC", "DU", "A", "T", "G", "C", "U"}


def dna_contact_residues(pdb_path, cutoff=4.5):
    """Which protein residues actually touch the operator. Measured, not declared.

    This is the recognition set BY DEFINITION, and it is scaffold-agnostic: no hardcoded helix
    range that happens to be right for TetR and wrong for everything else.
    """
    model = _model(pdb_path)
    dna = [a for ch in model for r in ch
           if r.get_resname().strip() in DNA_BASES for a in heavy(r)]
    if not dna:
        return set()
    D = np.array([a.coord for a in dna])
    out = set()
    for ch in _protein_chains(model):
        for r in ch:
            if r.id[0] != " ":
                continue
            A = np.array([a.coord for a in heavy(r)])
            if len(A) and np.linalg.norm(A[:, None, :] - D[None, :, :], axis=2).min() < cutoff:
                out.add(r.id[1])
    return out


def _centroid_sep(model, resnums):
    cen = []
    for ch in _protein_chains(model):
        A = [a.coord for r in ch if r.id[0] == " " and r.id[1] in resnums for a in heavy(r)]
        if A:
            cen.append(np.array(A).mean(axis=0))
    if len(cen) < 2:
        return None
    return float(np.linalg.norm(cen[0] - cen[1]))


def dbd_geometry(pdb_path, dbd_range, recognition_resnums=None):
    """The readout: how the two DNA-binding domains sit relative to each other.

    MEASURED ON THE RECOGNITION RESIDUES, not the DBD centroid. Verified on TetR 1QPI -> 2TCT:

        whole-DBD centroid (1-45)     33.51 -> 34.04   +0.53   <- says "nothing happened"
        alpha2            (27-34)     32.20 -> 29.25   -2.95
        alpha3 recognition(37-44)     36.62 -> 40.17   +3.55   <- the actual pendulum

    The DBD PIVOTS: alpha2 closes while alpha3 splays. A centroid sits between them and averages
    the motion away - it cancels the very signal it is meant to detect, and would have reported
    that TetR barely moves on induction, contradicting thirty years of TetR literature.

    The opposite signs are themselves diagnostic: a rigid-body slide moves both the same way, a
    pivot moves them apart. Both are reported so downstream can tell those apart.

    A single protein chain makes all of this undefined - the 1QPI asymmetric-unit trap
    (see structure.fetch_assembly). Returns None rather than a number computed from one DBD.
    """
    model = _model(pdb_path)
    lo, hi = dbd_range
    dbd_res = {r.id[1] for ch in _protein_chains(model) for r in ch
               if r.id[0] == " " and lo <= r.id[1] <= hi}
    if not dbd_res:
        return None
    rec = set(recognition_resnums or ()) & dbd_res
    out = {"centroid_dist": _centroid_sep(model, dbd_res)}
    if out["centroid_dist"] is None:
        return None
    if rec:
        out["recognition_sep"] = _centroid_sep(model, rec)
        out["n_recognition"] = len(rec)
    else:
        out["recognition_sep"] = None
        out["note"] = ("no recognition residues supplied: only the centroid is available, and the "
                       "centroid averages a DBD pivot away. Treat as uninformative.")
    return out


def second_shell(chain, pocket, cutoff=SECOND_SHELL_CUT):
    """Residues packed against the pocket but not touching the ligand."""
    res = [r for r in chain if r.id[0] == " "]
    P = [r for r in res if r.id[1] in pocket]
    if not P:
        return set()
    PA = np.array([a.coord for r in P for a in heavy(r)])
    out = set()
    for r in res:
        if r.id[1] in pocket:
            continue
        A = np.array([a.coord for a in heavy(r)])
        if len(A) and np.linalg.norm(A[:, None, :] - PA[None, :, :], axis=2).min() < cutoff:
            out.add(r.id[1])
    return out


def classify(tor, reachable, churn, pocket, shell, dbd, resnums, sig_cut=None, churn_cut=None):
    """recognition / transduction / output / noise / scaffold.

    A transduction residue must show THREE independent pieces of real structural evidence, none of
    them a graph artefact:
      * a real apo->holo torsion change (it moved), AND
      * a real change in its contact network between the two states (churn: the packing around it
        actually rearranged, not just a dihedral flip on a static neighbourhood), AND
      * physical reachability on a near-shortest pocket->DBD path (it CAN be in the chain of cause).

    Graph membership alone is deliberately NOT sufficient. The null calibration proved why: in a
    4.5 A contact graph, random pocket/DBD labels put 136/207 residues "on-path", so no residue is
    graph-significant and the path cannot resolve transmission at residue level. The graph is kept
    only as a reachability veto (excludes residues off any sensible path); the positive evidence is
    torsion + churn, both measured from the crystals.

    Torsion alone is noise (terminal wobble, TetR 5/205). sig_cut guard: with a single crystal pair
    nothing is estimable, so an all-zero / too-sparse torsion field yields sig_cut=None and NO
    transduction calls - reported honestly rather than fabricated by a percentile of zeros.
    """
    tvals = [v.get("torsion_signal") or 0.0 for v in tor.values()]
    if sig_cut is None:
        pos = [v for v in tvals if v > 0]
        sig_cut = float(np.percentile(pos, 75)) if len(pos) >= 4 else None
    if churn_cut is None:
        cpos = [c for c in churn.values() if c and c > 0]
        churn_cut = float(np.percentile(cpos, 75)) if len(cpos) >= 4 else None
    lo, hi = (min(resnums), max(resnums)) if resnums else (0, 0)
    cls = {}
    for rn in resnums:
        s = tor.get(rn, {}).get("torsion_signal") or 0.0
        moved = sig_cut is not None and s >= sig_cut
        churned = churn_cut is not None and (churn.get(rn) or 0.0) >= churn_cut
        terminal = (rn - lo) < TERMINAL_MARGIN or (hi - rn) < TERMINAL_MARGIN
        if rn in dbd:
            cls[rn] = "output"
        elif rn in pocket:
            cls[rn] = "recognition"
        elif moved and churned and rn in reachable and not terminal:
            cls[rn] = "transduction"
        elif moved or churned:
            cls[rn] = "noise"            # some evidence, but not the full three: not a relay
        else:
            cls[rn] = "scaffold"
    return cls, sig_cut


def native_template(state_paths, ligand_resname, out_json,
                    dbd_range=None, apo_ensemble=None, holo_ensemble=None):
    """-> native_allosteric_template.json

    state_paths: {'X_D': pdb, 'X_I': pdb, 'X_I_lig': pdb, ...}. X_I_lig (or X_I) must contain the
    effector, otherwise the pocket is undefined and we refuse rather than guess.
    apo_ensemble / holo_ensemble: optional extra structures; without them js_* stays None.
    """
    xd = state_paths.get("X_D")
    xi = state_paths.get("X_I")
    if not xd or not xi:
        raise ValueError("native_template needs both X_D and X_I")
    holo_for_pocket = state_paths.get("X_I_lig") or state_paths.get("X_D_lig") or xi

    pocket = pocket_residues(holo_for_pocket, ligand_resname)
    chain_d = _main_chain(xd)
    resnums = sorted(r.id[1] for r in chain_d if r.id[0] == " ")
    shell = second_shell(chain_d, pocket)

    apo_paths = _as_list(apo_ensemble) or [xd]
    holo_paths = _as_list(holo_ensemble) or [holo_for_pocket]
    tor = torsion_signal(apo_paths, holo_paths)

    dbd = set()
    if dbd_range:
        dbd = {r for r in resnums if dbd_range[0] <= r <= dbd_range[1]}
    on_path, dp, dd, shortest = allosteric_path(chain_d, pocket, dbd)
    cal = calibrated_path(chain_d, pocket, dbd)
    significant = cal["significant"]

    idx_d, Cd = contact_map(chain_d)
    idx_i, Ci = contact_map(_main_chain(xi))
    common = [r for r in idx_d if r in set(idx_i)]
    ia = [idx_d.index(r) for r in common]
    ib = [idx_i.index(r) for r in common]
    dC = Ci[np.ix_(ib, ib)] - Cd[np.ix_(ia, ia)]
    contact_churn = {rn: float(np.abs(dC[k]).sum()) for k, rn in enumerate(common)}

    # transduction = torsion moved AND contact network churned AND reachable. The null-calibrated
    # `significant` set is kept in the template for the scaffold-level enrichment signal, but is
    # NOT used as the per-residue mask - the null proved the graph has no residue-level resolution.
    cls, sig_cut = classify(tor, on_path, contact_churn, pocket, shell, dbd, resnums)

    # the recognition set is read off the operator complex when we have one: the residues that
    # actually touch DNA are the readout, and they are what must move.
    x_d_dna = state_paths.get("X_D_DNA")
    rec = dna_contact_residues(x_d_dna) if x_d_dna else set()
    geo_d = dbd_geometry(xd, dbd_range, rec) if dbd_range else None
    geo_i = dbd_geometry(xi, dbd_range, rec) if dbd_range else None
    dbd_delta = None
    if geo_d and geo_i:
        dbd_delta = {"centroid_dist_X_D": geo_d["centroid_dist"],
                     "centroid_dist_X_I": geo_i["centroid_dist"],
                     "centroid_delta": geo_i["centroid_dist"] - geo_d["centroid_dist"],
                     "recognition_resnums": sorted(rec) or None,
                     "recognition_sep_X_D": geo_d.get("recognition_sep"),
                     "recognition_sep_X_I": geo_i.get("recognition_sep")}
        rd, ri = geo_d.get("recognition_sep"), geo_i.get("recognition_sep")
        if rd is not None and ri is not None:
            dbd_delta["recognition_delta"] = ri - rd
            # this, not the centroid, is the number the DNA sees
            dbd_delta["delta"] = ri - rd
        else:
            dbd_delta["delta"] = None
            dbd_delta["warning"] = ("no operator complex: DBD motion can only be reported as a "
                                    "centroid shift, which cancels a pivot. Do not gate on it.")

    residues = {}
    for rn in resnums:
        t = tor.get(rn, {})
        residues[str(rn)] = {
            "class": cls[rn],
            "torsion_signal": t.get("torsion_signal"),
            "on_path_raw": rn in on_path,
            "on_path_significant": rn in significant,
            "path_pval": cal["pval"].get(rn),
            "d_pocket": dp.get(rn),
            "d_dbd": dd.get(rn),
            "contact_churn": contact_churn.get(rn),
            "js_estimable": bool(t.get("ensemble", False)),
        }

    tpl = {
        "ligand_resname": ligand_resname,
        "pocket": sorted(pocket),
        "second_shell": sorted(shell),
        "dbd": sorted(dbd),
        "dbd_range": list(dbd_range) if dbd_range else None,
        "on_path_raw": sorted(on_path),
        "on_path_significant": sorted(significant),
        "path_calibration": {"real_size": cal["real_size"],
                             "null_size_mean": cal["null_size_mean"],
                             "null_size_sd": cal["null_size_sd"],
                             "enriched": cal["enriched"],
                             "slack_sensitivity": cal["slack_sensitivity"]},
        "shortest_pocket_to_dbd": shortest,
        "torsion_sig_cut": sig_cut,
        "n_apo": (list(tor.values())[0]["n_apo"] if tor else 0),
        "n_holo": (list(tor.values())[0]["n_holo"] if tor else 0),
        "js_estimable": bool(tor and list(tor.values())[0].get("ensemble")),
        "dbd_geometry": dbd_delta,
        "residues": residues,
        "counts": {c: sum(1 for v in cls.values() if v == c)
                   for c in ("recognition", "transduction", "output", "noise", "scaffold")},
    }
    if shortest is None and dbd:
        tpl["warning"] = ("no contact path pocket->DBD: either the DBD range is wrong or this "
                          "structure is a monomer. Transduction cannot be assigned.")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(tpl, f, indent=2)
    return tpl


def build_masks(template_json, out_json, extra_protected=()):
    """-> masks.json. DBD is always protected in V1: it is the readout, not a design target.

    recognition  free
    transduction same size class AND same charge only (design.py enforces; see the ASP/GLU-for-LEU
                 bug this rule exists to stop)
    protected    everything else that matters: DBD, noise-class movers, explicit extras
    """
    tpl = template_json if isinstance(template_json, dict) else json.load(open(template_json))
    res = tpl["residues"]
    recognition, transduction, protected = [], [], []
    for k, v in res.items():
        rn = int(k)
        c = v["class"]
        if c == "recognition":
            recognition.append(rn)
        elif c == "transduction":
            transduction.append(rn)
        elif c in ("output", "noise"):
            protected.append(rn)
    protected.extend(int(x) for x in extra_protected)
    # a residue must never appear in two masks: protected wins, then transduction.
    protected = sorted(set(protected))
    transduction = sorted(set(transduction) - set(protected))
    recognition = sorted(set(recognition) - set(protected) - set(transduction))
    if not recognition:
        raise RuntimeError("empty recognition mask: nothing left to design. Check the ligand "
                           "resname and the pocket cutoff.")
    masks = {"recognition_mask": recognition,
             "transduction_mask": transduction,
             "protected_mask": protected,
             "policy": {"recognition": "free",
                        "transduction": "same_size_class_and_charge",
                        "protected": "fixed"}}
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(masks, f, indent=2)
    return masks


def run(ctx):
    """requires ctx['states']; produces ctx['template'], ctx['masks']"""
    st = ctx["states"]
    out = ctx["out_dir"]
    tpl = native_template(st["paths"], st["effector_resname"],
                          os.path.join(out, "native_allosteric_template.json"),
                          dbd_range=st.get("dbd_range"),
                          apo_ensemble=st.get("apo_ensemble"),
                          holo_ensemble=st.get("holo_ensemble"))
    masks = build_masks(tpl, os.path.join(out, "masks.json"),
                        extra_protected=st.get("extra_protected", ()))
    return {"template": tpl, "masks": masks}
