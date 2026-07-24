"""Mechanism-family-specific DBD readout. The old dbd_geometry measures inter-DBD recognition-residue
separation - a TetR pendulum. LacI does not move that way (its DBD 1-61 is disordered without operator;
the hinge helix 50-58 is only a helix when bound to DNA), so the TetR metric on LacI returns None or a
meaningless number. This dispatches the readout by mechanism family and, critically, tracks COVERAGE:

    dbd_readout(pose, family) -> ReadoutVector          # pure, per-pose, every component carries coverage
    gate_direction(readout_apo, readout_holo, family)   # frozen sign convention + noise-band test ONLY here

Discipline (each of these was a real trap):
  * a component is (value, coverage), never a bare float. helicity is UNDEFINED in the IPTG-holo state
    (the hinge is unresolved) - returning 0 there would read as a huge induction signal and FALSELY
    pass LacI. The gate RAISES on a missing-coverage main signal; it never fills 0 or skips.
  * the hinge angle is a DynDom-style inter-subdomain ROTATION (superpose the N-subdomain, measure the
    residual C-subdomain rotation), not a 3-point centroid - so it is robust to the exact, sequence-
    discontinuous subdomain boundary, and the boundary can be swept to check stability.
  * the sign convention (IPTG -> induction -> DNA release) is frozen from independent knowledge and
    lives only in the gate; the readout never reads direction out of the data it is meant to test.
  * an unknown family RAISES - it never falls back to the TetR readout.
"""
from dataclasses import dataclass, field

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer

_PDB, _CIF = PDBParser(QUIET=True), MMCIFParser(QUIET=True)

# LacI subdomain / hinge / dimer-interface boundaries (defaults; sweepable to check boundary sensitivity)
LACI = {
    "n_subdomain": list(range(62, 161)) + list(range(290, 330)),   # core N-subdomain (discontinuous)
    "c_subdomain": list(range(161, 290)),                          # core C-subdomain + dimer face
    "hinge": list(range(50, 59)),                                  # hinge helix - ORDERED ONLY ON DNA
    "dimer_interface": list(range(220, 290)),                      # functional-dimer N/C interface
    "helix_phi": (-100.0, -30.0), "helix_psi": (-80.0, 5.0),
}


@dataclass
class Component:
    value: float = float("nan")
    coverage: bool = False
    data: object = None                                           # coords etc. for cross-state compare


@dataclass
class ReadoutVector:
    family: str
    components: dict = field(default_factory=dict)
    pose: str = ""


def _model(pdb):
    parser = _CIF if pdb.lower().endswith(".cif") else _PDB
    return next(iter(parser.get_structure("S", pdb)))


def _ca_by_resnum(model):
    out = {}
    for ch in model:
        for r in ch:
            if r.id[0] == " " and r.has_id("CA"):
                out.setdefault(ch.id, {})[r.id[1]] = r["CA"].coord.astype(float)
    return out


def _subdomain_coords(ca_chains, resnums):
    """{resnum: CA coord} for a residue set from the largest protein chain, so cross-state comparison
    can intersect on residue NUMBER (apo and holo resolve different subsets). -> (map, n_resolved)."""
    if not ca_chains:
        return {}, 0
    ch = max(ca_chains, key=lambda c: len(ca_chains[c]))
    m = {rn: ca_chains[ch][rn] for rn in resnums if rn in ca_chains[ch]}
    return m, len(m)


def _hinge_helicity(model, cfg):
    """Fraction of hinge residues in helical phi/psi. coverage = enough hinge residues resolved.
    In an IPTG-bound LacI the hinge is NOT built -> coverage False (not a value of 0)."""
    from Bio.PDB.internal_coords import IC_Chain  # phi/psi
    chain = max((c for c in model), key=lambda c: sum(1 for r in c if r.id[0] == " "), default=None)
    if chain is None:
        return Component(coverage=False)
    try:
        chain.atom_to_internal_coordinates()
    except Exception:
        return Component(coverage=False)
    hinge = set(cfg["hinge"])
    resolved, helical = 0, 0
    lo_phi, hi_phi = cfg["helix_phi"]; lo_psi, hi_psi = cfg["helix_psi"]
    for r in chain:
        if r.id[1] not in hinge or r.id[0] != " " or not getattr(r, "internal_coord", None):
            continue
        phi = r.internal_coord.get_angle("phi"); psi = r.internal_coord.get_angle("psi")
        if phi is None or psi is None:
            continue
        resolved += 1
        if lo_phi <= phi <= hi_phi and lo_psi <= psi <= hi_psi:
            helical += 1
    cover = resolved >= max(3, len(hinge) // 2)
    return Component(value=(helical / resolved if resolved else float("nan")), coverage=cover,
                     data={"resolved": resolved, "n_hinge": len(hinge)})


def _laci_readout(pdb, cfg):
    model = _model(pdb)
    ca = _ca_by_resnum(model)
    n_map, n_n = _subdomain_coords(ca, cfg["n_subdomain"])
    c_map, n_c = _subdomain_coords(ca, cfg["c_subdomain"])
    # subdomain frames (resnum -> coord) carried for the DynDom rotation computed in gate_direction
    sub = Component(value=float(n_n + n_c), coverage=(n_n >= 6 and n_c >= 6),
                    data={"n_map": n_map, "c_map": c_map})
    iface_map, n_if = _subdomain_coords(ca, cfg["dimer_interface"])
    n_chains = sum(1 for _ in model)
    dimer = Component(value=float(n_if), coverage=(n_if >= 6 and n_chains >= 2),
                      data={"iface_map": iface_map, "n_chains": n_chains})
    return ReadoutVector("LacI_family", {"subdomain": sub, "dimer_interface": dimer,
                                         "hinge_helicity": _hinge_helicity(model, cfg)}, pose=pdb)


def _tetr_readout(pdb, cfg):
    """Wrap the existing TetR metric unchanged - the refactor must leave TetR bit-for-bit identical."""
    from .allostery import dbd_geometry
    geo = dbd_geometry(pdb, cfg["dbd_range"], cfg.get("recognition_resnums"))
    if geo is None:
        return ReadoutVector("TetR_family", {"recognition_sep": Component(coverage=False)}, pose=pdb)
    rec = geo.get("recognition_sep")
    return ReadoutVector("TetR_family", {
        "recognition_sep": Component(value=rec, coverage=rec is not None, data=geo),
        "centroid_dist": Component(value=geo.get("centroid_dist"), coverage=True)}, pose=pdb)


_REGISTRY = {"TetR_family": _tetr_readout, "LacI_family": _laci_readout}


def dbd_readout(pose_pdb, family, cfg=None):
    if family not in _REGISTRY:
        raise ValueError("no DBD readout for mechanism family %r - refusing to fall back to the TetR "
                         "metric (that would silently mis-measure the scaffold)" % family)
    cfg = dict(LACI, **(cfg or {})) if family == "LacI_family" else (cfg or {})
    return _REGISTRY[family](pose_pdb, cfg)


def _rotation_angle(rot):
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))


def _paired(map_a, map_b):
    """Two {resnum: coord} maps -> (array_a, array_b) over the SHARED residues, same order."""
    keys = sorted(set(map_a) & set(map_b))
    return np.array([map_a[k] for k in keys]), np.array([map_b[k] for k in keys]), len(keys)


def _hinge_rotation(apo_sub, holo_sub):
    """DynDom-style: superpose N-subdomain apo->holo (on shared residues), then the residual rotation
    carrying the apo C-subdomain onto the holo C-subdomain is the inter-subdomain hinge rotation."""
    aN, hN, kN = _paired(apo_sub.data["n_map"], holo_sub.data["n_map"])
    aC, hC, kC = _paired(apo_sub.data["c_map"], holo_sub.data["c_map"])
    if kN < 6 or kC < 6:
        return None
    svd = SVDSuperimposer(); svd.set(hN, aN); svd.run(); rotN, tranN = svd.get_rotran()
    aC_onN = aC @ rotN + tranN
    svd2 = SVDSuperimposer(); svd2.set(hC, aC_onN); svd2.run()
    return _rotation_angle(svd2.get_rotran()[0])


def noise_band(paths, family, cfg=None, pct=95):
    """The within-state dispersion of the gate signal - the band the induced (apo->holo) change must
    exceed. Computed from apo-apo AND holo-holo pairs POOLED (pass one state's paths, or both). If the
    apo and holo sets sit in different crystal forms this band is measuring packing, not motion - so the
    space groups are reported for the human to check (stratification is a judgement, not silently done).
    -> {component: threshold, 'n_pairs': int, 'space_groups': {...}}."""
    from itertools import combinations
    reads = [dbd_readout(p, family, cfg) for p in paths]
    covered = [r for r in reads if r.components.get("subdomain") and r.components["subdomain"].coverage]
    hinge, iface = [], []
    for a, b in combinations(covered, 2):
        ang = _hinge_rotation(a.components["subdomain"], b.components["subdomain"])
        if ang is not None:
            hinge.append(ang)
        iface.append(abs(a.components["dimer_interface"].value - b.components["dimer_interface"].value))
    sg = {}
    for p in paths:
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
            for line in txt.splitlines():
                if "space_group_name_H-M" in line:
                    sg[p.split("/")[-1]] = line.split(None, 1)[1].strip().strip("'\""); break
        except Exception:
            pass
    return {"hinge_rotation_deg": float(np.percentile(hinge, pct)) if hinge else 0.0,
            "dimer_interface_delta": float(np.percentile(iface, pct)) if iface else 0.0,
            "n_pairs": len(hinge), "space_groups": sg}


def gate_direction(readout_apo, readout_holo, family, noise_band=None):
    """Frozen-sign, coverage-gated direction test. RAISES if a main signal lacks coverage in either
    state. noise_band: {component: threshold} from apo-apo / holo-holo dispersion (see noise_band())."""
    noise = noise_band or {}
    out = {"family": family, "components": {}, "passed": None}
    if family == "LacI_family":
        # MAIN signals: subdomain hinge rotation + dimer-interface change. hinge_helicity is D-DNA-only
        # sanity, never a directional signal here.
        for main in ("subdomain", "dimer_interface"):
            if not (readout_apo.components[main].coverage and readout_holo.components[main].coverage):
                raise ValueError("LacI gate: '%s' lacks coverage in apo or holo - cannot judge "
                                 "direction; fill-0 here would be a false pass" % main)
        angle = _hinge_rotation(readout_apo.components["subdomain"], readout_holo.components["subdomain"])
        if angle is None:
            raise ValueError("LacI gate: subdomain CA sets do not match between apo and holo (construct/"
                             "oligomer mismatch?) - hinge rotation undefined")
        d_iface = abs(readout_holo.components["dimer_interface"].value
                      - readout_apo.components["dimer_interface"].value)
        out["components"] = {"hinge_rotation_deg": angle, "dimer_interface_delta": d_iface}
        # frozen sign is about DIRECTION (release), not encoded as +/- of these magnitudes; the gate only
        # asks whether the induced change exceeds the noise band. Sign/topology is applied downstream.
        thr_a = noise.get("hinge_rotation_deg", 0.0)
        thr_i = noise.get("dimer_interface_delta", 0.0)
        out["passed"] = bool(angle > thr_a and d_iface > thr_i)
        out["noise_band"] = {"hinge_rotation_deg": thr_a, "dimer_interface_delta": thr_i}
    elif family == "TetR_family":
        a, h = readout_apo.components.get("recognition_sep"), readout_holo.components.get("recognition_sep")
        if not (a and h and a.coverage and h.coverage):
            raise ValueError("TetR gate: recognition_sep lacks coverage in apo or holo")
        d = abs(h.value - a.value)
        out["components"] = {"recognition_sep_delta": d}
        out["passed"] = bool(d > noise.get("recognition_sep_delta", 0.0))
    else:
        raise ValueError("no gate for family %r" % family)
    return out
