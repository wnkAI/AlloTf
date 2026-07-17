"""One functional state as a 3D graph, ready for the equivariant surrogate.

A state is a protein backbone in a specific conformation, optionally with the target ligand and/or
operator DNA. AlloSurrogate sees SIX of these per candidate and reads the differences between them;
this module builds one. It does NOT compute energies - it turns a prepared PDB (the same PDB the
PyRosetta oracle scores) plus the ligand graph into node/edge tensors.

Design that keeps the surrogate honest:
  * Residue nodes carry the same masks Design uses (recognition / transduction / protected) and the
    same allosteric annotations (on-path, torsion class) the physical template produced. The
    surrogate is told WHERE the chemistry can change and WHICH residues the physics thinks matter,
    rather than rediscovering it from coordinates.
  * Distance-to-ligand and distance-to-DNA are explicit node features. In the apo states they are
    None, and that Noneness is a real signal (this is the state with no ligand), not zero-filled.
  * Edges are radius-graph contacts with the actual distance on each edge, so an equivariant net
    has geometry to work with without us hand-building directions.

The label side (physics oracle outputs) lives in dataset.py, not here: a graph is an input, and
mixing the target into it is how leakage starts.
"""
import numpy as np

from utils.contacts import heavy
from ..structure import load_assembly
from .ligand_encoder import encode as encode_ligand

DNA_RESNAMES = {"DA", "DT", "DG", "DC", "DU"}
CONTACT_RADIUS = 8.0        # residue Cbeta radius graph
AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
MASK_TYPES = ["recognition", "transduction", "protected", "scaffold"]


def _one_hot(v, vocab):
    x = [0.0] * (len(vocab) + 1)
    x[vocab.index(v) if v in vocab else len(vocab)] = 1.0
    return x


def _repr_atom(res):
    """CB for the sidechain direction, CA as fallback (GLY). The node sits where packing happens."""
    if res.has_id("CB"):
        return res["CB"].coord
    if res.has_id("CA"):
        return res["CA"].coord
    hs = heavy(res)
    return np.mean([a.coord for a in hs], axis=0) if hs else None


def _protein_residues(model):
    out = []
    for ch in model:
        for r in ch:
            if r.id[0] == " " and (r.has_id("CA") or r.has_id("CB")):
                out.append((ch.id, r))
    return out


def _ligand_heavy(model, resname):
    at = []
    for ch in model:
        for r in ch:
            if r.id[0] != " " and r.get_resname().strip() == resname:
                at.extend(heavy(r))
    return at


def _dna_heavy(model):
    at = []
    for ch in model:
        for r in ch:
            if r.get_resname().strip() in DNA_RESNAMES:
                at.extend(heavy(r))
    return at


def residue_nodes(pdb_path, masks=None, template=None, ligand_resname=None):
    """-> (node_features, coords, meta) for every protein residue in the (assembly-merged) pose.

    node features: AA one-hot, mask one-hot, torsion class flags, SASA-free geometry proxies,
    distance-to-ligand, distance-to-DNA. Distances are None when that partner is absent - the apo
    and DNA-free states are supposed to look different, and zero-filling would erase that.
    """
    model = load_assembly(pdb_path)
    residues = _protein_residues(model)
    lig = _ligand_heavy(model, ligand_resname) if ligand_resname else []
    dna = _dna_heavy(model)
    LIG = np.array([a.coord for a in lig]) if lig else None
    DNA = np.array([a.coord for a in dna]) if dna else None

    masks = masks or {}
    rec = set(masks.get("recognition_mask", []))
    trans = set(masks.get("transduction_mask", []))
    prot = set(masks.get("protected_mask", []))
    tpl_res = (template or {}).get("residues", {})

    feats, coords, meta = [], [], []
    for chain_id, res in residues:
        rn = res.id[1]
        c = _repr_atom(res)
        if c is None:
            continue
        which = ("recognition" if rn in rec else "transduction" if rn in trans
                 else "protected" if rn in prot else "scaffold")
        t = tpl_res.get(str(rn), {})
        d_lig = float(np.linalg.norm(LIG - c, axis=1).min()) if LIG is not None else None
        d_dna = float(np.linalg.norm(DNA - c, axis=1).min()) if DNA is not None else None
        row = _one_hot(res.get_resname().strip(), AA3)
        row += _one_hot(which, MASK_TYPES)
        row += [
            float(t.get("class") == "transduction"),
            float(bool(t.get("on_path_significant"))),
            float(t.get("torsion_signal") or 0.0),
            float(t.get("contact_churn") or 0.0),
            # distance features: value if present, plus an explicit "is present" bit so the model
            # never confuses "far" with "absent"
            (d_lig if d_lig is not None else 0.0), float(d_lig is not None),
            (d_dna if d_dna is not None else 0.0), float(d_dna is not None),
        ]
        feats.append(row)
        coords.append(c)
        meta.append({"chain": chain_id, "resnum": rn, "mask": which,
                     "resname": res.get_resname().strip()})
    return (np.array(feats, dtype=np.float32),
            np.array(coords, dtype=np.float32),
            meta)


def radius_edges(coords, radius=CONTACT_RADIUS):
    """Undirected contact graph with |distance| on each edge. Both directions emitted."""
    n = len(coords)
    if n == 0:
        return np.zeros((2, 0), np.int64), np.zeros((0, 1), np.float32)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    src, dst, ew = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < radius:
                src += [i, j]
                dst += [j, i]
                ew += [[D[i, j]], [D[i, j]]]
    return (np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64),
            np.array(ew, dtype=np.float32) if ew else np.zeros((0, 1), np.float32))


def build_state(pdb_path, state_name, masks=None, template=None,
                ligand_smiles=None, ligand_resname=None, ligand_coords=None):
    """One state -> graph dict for the surrogate.

    -> {
        'state', 'residue_features','residue_coords','residue_meta',
        'edge_index','edge_weight',
        'ligand'   : ligand graph (encode_ligand) or None,
        'has_ligand','has_dna'
    }
    The ligand subgraph is attached only for the liganded states (DL, IL). A caller that asks for a
    ligand graph on an apo state is refused rather than handed an empty one, because an apo state
    that carries a ligand graph is exactly the mislabelling the six-state contract forbids.
    """
    rf, rc, rmeta = residue_nodes(pdb_path, masks, template, ligand_resname)
    ei, ew = radius_edges(rc)
    liganded = state_name in ("DL", "IL")
    lig_graph = None
    if liganded:
        if not ligand_smiles:
            raise ValueError("state %s is liganded but no ligand SMILES was given" % state_name)
        lig_graph = encode_ligand(ligand_smiles, coords=ligand_coords)
    elif ligand_smiles is not None and state_name in ("D0", "I0", "D_DNA", "I_DNA"):
        raise ValueError("state %s must be ligand-free; refusing to attach a ligand graph "
                         "(mislabelled state poisons the double difference)" % state_name)
    has_dna = state_name in ("D_DNA", "I_DNA")
    return {"state": state_name,
            "residue_features": rf, "residue_coords": rc, "residue_meta": rmeta,
            "edge_index": ei, "edge_weight": ew,
            "ligand": lig_graph,
            "has_ligand": bool(liganded), "has_dna": bool(has_dna)}


def residue_feature_dim():
    return (len(AA3) + 1) + (len(MASK_TYPES) + 1) + 8


def build_six_graphs(state_paths, masks, template, ligand_smiles, ligand_resname,
                     ligand_coords=None):
    """All six states as graphs, keyed by state name. A missing template path -> None (fail closed,
    same convention as the physics side)."""
    key = {"D0": "X_D", "I0": "X_I", "DL": "X_D_lig", "IL": "X_I_lig",
           "D_DNA": "X_D_DNA", "I_DNA": "X_I_DNA"}
    out = {}
    for st, tkey in key.items():
        p = state_paths.get(tkey)
        if not p:
            out[st] = None
            continue
        smi = ligand_smiles if st in ("DL", "IL") else None
        lc = ligand_coords.get(st) if isinstance(ligand_coords, dict) else ligand_coords
        out[st] = build_state(p, st, masks, template, smi, ligand_resname,
                              lc if st in ("DL", "IL") else None)
    return out
