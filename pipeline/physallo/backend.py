"""PhysAlloDesign assembly: real structure + ligand pose -> designed pocket sequences.

Ties the five cores together:
    aa_filter   which residues may sit at each position (chemistry + geometry, 3-8 of 20)
    sidechain   build every rotamer of every candidate residue from the backbone
    scoring     score it with explicit terms (vdW, Coulomb, directional H-bond, desolv, unsat)
    rotamers    strain penalty
    search      joint sequence+rotamer simulated annealing

This is the production designer. It never asks a native-sequence prior what belongs in a pocket.
All energies are RELATIVE proxies on one scaffold - never a Kd.
"""
import os
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter("ignore", PDBConstructionWarning)

from . import scoring, rotamers, sidechain
from .aa_filter import allowed as allowed_aa
from .search import SearchSpace, anneal, multi_start, seq_of, mutations, diversify

JUNK = {"HOH", "SO4", "GOL", "EDO", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "ACT",
        "PEG", "DMS", "TRS", "IOD", "BR", "NO3", "FMT", "EPE", "MPD", "BME", "1PE", "PGE"}
BB = ("N", "CA", "C", "O")
NEAR = 10.0     # only environment atoms within this of a design position matter


class Context:
    """Everything the energy function needs, extracted once."""

    def __init__(self, backbone, env_xyz, env_el, env_q, lig, positions, allowed, wt, masks=None):
        self.backbone = backbone            # {resnum: {'N','CA','C'}}
        self.env_xyz = np.asarray(env_xyz) if len(env_xyz) else np.zeros((0, 3))
        self.env_el = env_el
        self.env_q = np.asarray(env_q) if len(env_q) else np.zeros(0)
        self.lig = lig                      # list of dict(xyz, element, charge, ...)
        self.positions = positions
        self.allowed = allowed
        self.wt = wt
        self.masks = masks or {}
        self.lig_xyz = np.array([a["xyz"] for a in lig]) if lig else np.zeros((0, 3))
        self.lig_el = [a.get("element", "C") for a in lig]
        self.lig_q = np.array([a.get("charge", 0.0) for a in lig]) if lig else np.zeros(0)
        self.lig_polar = np.array([a["xyz"] for a in lig
                                   if a.get("element", "C").upper() in ("N", "O")]) \
            if lig else np.zeros((0, 3))
        # per-position local environment (cached: this is the inner loop of the search)
        self._env_near = {}
        for p in positions:
            ca = np.asarray(backbone[p]["CA"])
            if len(self.env_xyz):
                m = np.linalg.norm(self.env_xyz - ca, axis=1) < NEAR
                self._env_near[p] = (self.env_xyz[m], [e for e, k in zip(env_el, m) if k],
                                     self.env_q[m])
            else:
                self._env_near[p] = (np.zeros((0, 3)), [], np.zeros(0))

    def env_near(self, p):
        return self._env_near[p]


def _charge_of(resname, atom_name):
    return scoring.PROT_Q.get((resname.upper(), atom_name), 0.0)


def prepare(pdb_path, design_positions, ligand_resname=None, chain_id=None, masks=None,
            conservation=None):
    """Extract backbone, fixed environment and ligand from a real structure."""
    st = PDBParser(QUIET=True).get_structure("x", pdb_path)
    model = next(iter(st))
    chain = model[chain_id] if chain_id else max(
        (c for c in model), key=lambda c: sum(1 for r in c if r.id[0] == " "))
    dset = set(design_positions)

    backbone, wt = {}, {}
    env_xyz, env_el, env_q = [], [], []
    for r in chain:
        if r.id[0] != " ":
            continue
        num = r.id[1]
        if num in dset:
            if not all(a in r for a in ("N", "CA", "C")):
                raise ValueError("design position %d lacks backbone atoms" % num)
            backbone[num] = {a: np.array(r[a].coord, float) for a in ("N", "CA", "C")}
            wt[num] = r.get_resname().upper()
            for a in r:                      # keep only its backbone in the environment
                if a.get_name() in BB and a.element != "H":
                    env_xyz.append(a.coord); env_el.append(a.element); env_q.append(0.0)
        else:
            for a in r:
                if a.element == "H":
                    continue
                env_xyz.append(a.coord); env_el.append(a.element)
                env_q.append(_charge_of(r.get_resname(), a.get_name()))
    missing = dset - set(backbone)
    if missing:
        raise ValueError("design positions absent from chain %s: %s" % (chain.id, sorted(missing)))

    lig = []
    if ligand_resname:
        for r in chain.get_parent().get_residues():
            if r.get_resname().strip().upper() != ligand_resname.upper():
                continue
            for a in r:
                if a.element == "H":
                    continue
                lig.append(dict(xyz=np.array(a.coord, float), element=a.element, charge=0.0,
                                is_donor=a.element in ("N", "O"),
                                is_acceptor=a.element in ("O", "N", "S")))
        if not lig:
            raise ValueError("ligand %s not found in %s" % (ligand_resname, pdb_path))

    masks = masks or {p: "recognition" for p in design_positions}
    allowed = {}
    for p in design_positions:
        allowed[p] = allowed_aa(backbone[p]["CA"], env_xyz, lig, wt[p], masks.get(p, "recognition"),
                                conservation=(conservation or {}).get(p))
    return Context(backbone, env_xyz, env_el, env_q, lig, list(design_positions), allowed, wt, masks)


def build_state_atoms(ctx, p, aa, chis):
    """-> (xyz[n,3], elements[n], charges[n]) for one designed side chain."""
    at = sidechain.build(aa, ctx.backbone[p], chis)
    if not at:
        return np.zeros((0, 3)), [], np.zeros(0)
    xyz = np.array([x for _, x, _ in at])
    els = [e for _, _, e in at]
    q = np.array([_charge_of(aa, n) for n, _, _ in at])
    return xyz, els, q


def make_energy_fn(ctx, w=None):
    """-> energy(state) with state = {pos: (aa, chis)}. Terms are explicit and inspectable."""
    w = w or dict(vdw=1.0, coulomb=1.0, hbond=1.0, desolv=0.5, unsat=1.0, strain=1.0)
    cache = {}

    def one_body(p, aa, chis):
        key = (p, aa, chis)
        if key in cache:
            return cache[key]
        xyz, els, q = build_state_atoms(ctx, p, aa, chis)
        if not len(xyz):
            cache[key] = (0.0, np.zeros((0, 3)), [], np.zeros(0))
            return cache[key]
        exyz, eel, eq = ctx.env_near(p)
        e = 0.0
        ev, _ = scoring.vdw(xyz, els, exyz, eel);           e += w["vdw"] * ev
        e += w["coulomb"] * scoring.coulomb(xyz, q, exyz, eq)
        if len(ctx.lig_xyz):
            lv, _ = scoring.vdw(xyz, els, ctx.lig_xyz, ctx.lig_el)
            e += w["vdw"] * lv
            e += w["coulomb"] * scoring.coulomb(xyz, q, ctx.lig_xyz, ctx.lig_q)
            polar = np.array([x for x, el in zip(xyz, els) if el in ("N", "O")])
            if len(polar) and len(ctx.lig_polar):
                roots = np.repeat(ctx.backbone[p]["CA"][None, :], len(polar), axis=0)
                hb, _ = scoring.hbonds(polar, roots, ctx.lig_polar, None)
                e += w["hbond"] * hb
                u, _ = scoring.unsat_polar(polar, ctx.lig_polar, exyz)
                e += w["unsat"] * u
        e += w["strain"] * rotamers.strain(aa, chis)
        cache[key] = (e, xyz, els, q)
        return cache[key]

    def energy(state):
        tot = 0.0
        built = {}
        for p, (aa, chis) in state.items():
            e, xyz, els, q = one_body(p, aa, chis)
            tot += e
            built[p] = (xyz, els, q)
        ps = list(state)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = built[ps[i]], built[ps[j]]
                if not len(a[0]) or not len(b[0]):
                    continue
                if np.linalg.norm(ctx.backbone[ps[i]]["CA"] - ctx.backbone[ps[j]]["CA"]) > 14:
                    continue
                ev, _ = scoring.vdw(a[0], a[1], b[0], b[1])
                tot += w["vdw"] * ev + w["coulomb"] * scoring.coulomb(a[0], a[2], b[0], b[2])
        return float(tot)

    return energy


def design(ctx, n_candidates=100, n_steps=6000, n_restarts=12, seed=0, max_rot=20):
    """-> list of dict(seq, mutations, energy, state) sorted by energy."""
    space = SearchSpace(ctx.positions, ctx.allowed,
                        lambda a: rotamers.rotamers(a, max_rot), wt=ctx.wt)
    efn = make_energy_fn(ctx)
    res = multi_start(space, efn, n_restarts=n_restarts, n_steps=n_steps, seed=seed)
    res = diversify(space, res)[:n_candidates]
    return [dict(seq=seq_of(space, st), mutations=mutations(space, st), energy=en, state=st)
            for st, en in res], space, efn
