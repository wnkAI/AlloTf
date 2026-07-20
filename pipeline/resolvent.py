"""Parameter-free directed pocket->DBD response through a per-state resolvent. No training.

The switch question the six energies cannot answer on their own: does a perturbation at the effector
pocket actually propagate, through THIS state's stiffness, into the DBD output direction the native
switch uses? RIFT will later LEARN this operator; here it is the frozen physical baseline that
decides whether the resolvent idea carries any signal before we spend months on the learned version.

An anisotropic network model gives H_s (3N x 3N): nodes are residues (CA), springs run along the
inter-residue unit vectors with a stiffness fixed BY CONTACT TYPE (covalent backbone > metal ~ salt
bridge > H-bond > packing) - a frozen physical ordering, never fit to a phenotype label. The directed
gain

    g_s = v_out^T (H_s + eps I)^-1 u_lig

reads a unit pocket perturbation u_lig into the native DBD direction v_out. eps regularises the six
rigid-body zero modes so the resolvent exists. The ligand-dependent transmission margin is the same
double difference the coupling uses, so its sign convention matches ddG_coup:

    m_trans = (g_IL - g_DL) - (g_I0 - g_D0)

v_out and u_lig are FIXED directions taken from the native D/I and apo/holo structures; only H_s
changes across the four states. The weights live in SPRING and can be scaled for a sensitivity sweep,
but the sweep happens AFTER the switch/non-switch labels are unblinded, never to fit them.
"""
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

from utils.contacts import heavy, POS, NEG, CONTACT, HB

warnings.simplefilter("ignore", PDBConstructionWarning)
_PARSER = PDBParser(QUIET=True)

# Frozen physical spring constants (relative). Ordering is the physics: a covalent bond is far
# stiffer than a van der Waals contact; a metal coordination and a salt bridge sit in between. These
# are NOT tuned to any switch label - they are the same for every scaffold and every variant.
SPRING = {
    "backbone": 10.0,   # i, i+1 covalent peptide bond
    "metal": 6.0,       # metal coordination (e.g. His-Mg in TetR)
    "saltbridge": 4.0,
    "hbond": 3.0,
    "contact": 1.0,     # generic heavy-atom packing (van der Waals)
}
EPS = 1e-2              # rigid-body regulariser; well below the softest real mode


def _ca(chain):
    """[(resnum, CA coord)] for standard residues that have a CA."""
    out = []
    for r in chain:
        if r.id[0] == " " and r.has_id("CA"):
            out.append((r.id[1], r["CA"].coord.astype(float)))
    return out


def _typed_springs(residues):
    """residues: [(resnum, Residue)] -> {(i,j): stiffness} over CA node indices.

    Sums the contact-type contributions between every residue pair; each type contributes its frozen
    SPRING weight (packing scaled by how many heavy-atom contacts, capped so a big residue pair does
    not dominate). Sequence neighbours always get the backbone spring.
    """
    n = len(residues)
    heavies = [np.array([a.coord for a in heavy(r)]) for _, r in residues]
    names = [r.get_resname().strip() for _, r in residues]
    k = {}
    for i in range(n):
        Ai = heavies[i]
        if not len(Ai):
            continue
        for j in range(i + 1, n):
            Aj = heavies[j]
            if not len(Aj):
                continue
            d = np.linalg.norm(Ai[:, None, :] - Aj[None, :, :], axis=2)
            dmin = d.min()
            if dmin >= CONTACT and (residues[j][0] - residues[i][0]) != 1:
                continue
            s = 0.0
            if residues[j][0] - residues[i][0] == 1:
                s += SPRING["backbone"]
            nc = int((d < CONTACT).sum())
            if nc:
                s += SPRING["contact"] * min(nc, 6) / 6.0
                # N/O within H-bond distance -> add an H-bond spring
                No_i = [a.coord for a in heavy(residues[i][1]) if a.element in ("N", "O")]
                No_j = [a.coord for a in heavy(residues[j][1]) if a.element in ("N", "O")]
                if No_i and No_j:
                    dd = np.linalg.norm(np.array(No_i)[:, None, :] - np.array(No_j)[None, :, :], axis=2)
                    if dd.min() < HB:
                        s += SPRING["hbond"]
                if ({names[i], names[j]} & POS) and ({names[i], names[j]} & NEG) and dmin < HB + 0.5:
                    s += SPRING["saltbridge"]
            if s > 0:
                k[(i, j)] = s
    return k


def anm_hessian(coords, springs, eps=EPS):
    """coords: (N,3). springs: {(i,j): k}. -> (3N,3N) ANM Hessian + eps I.

    Standard ANM: each spring contributes k * (e_ij outer e_ij) to the ii and jj superelements and
    minus that to ij/ji, with e_ij the unit vector between the two nodes. eps I lifts the six
    rigid-body zero modes so the resolvent (H+eps I)^-1 exists.
    """
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    for (i, j), k in springs.items():
        d = coords[j] - coords[i]
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        e = d / L
        blk = k * np.outer(e, e)
        H[3*i:3*i+3, 3*i:3*i+3] += blk
        H[3*j:3*j+3, 3*j:3*j+3] += blk
        H[3*i:3*i+3, 3*j:3*j+3] -= blk
        H[3*j:3*j+3, 3*i:3*i+3] -= blk
    H += eps * np.eye(3 * n)
    return H


def directed_gain(H, v_out, u_lig):
    """g = v_out^T (H)^-1 u_lig. Both vectors live in the 3N node space, already normalised."""
    x = np.linalg.solve(H, u_lig)
    return float(v_out @ x)


def _region_direction(index, resnums, disp):
    """Embed a per-residue displacement onto the region's nodes as a normalised 3N direction.

    index: {resnum: node_i}. resnums: the region (pocket or DBD). disp: {resnum: (3,) displacement}.
    Returns a unit 3N vector supported only on the region, or None if the region has no displacement.
    """
    n = len(index)
    v = np.zeros(3 * n)
    for rn in resnums:
        if rn in index and rn in disp:
            v[3*index[rn]:3*index[rn]+3] = disp[rn]
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None
    return v / norm


def _aligned_disp(ref_chain, mob_chain):
    """Per-residue CA displacement mob-ref AFTER removing rigid body (superpose on shared CAs).

    The internal conformational change, not the crystal frame. -> {resnum: (3,) vector}."""
    ref = dict(_ca(ref_chain))
    mob = dict(_ca(mob_chain))
    common = sorted(set(ref) & set(mob))
    if len(common) < 4:
        return {}
    ref_at = [ref_chain[rn]["CA"] for rn in common]
    mob_at = [mob_chain[rn]["CA"] for rn in common]
    sup = Superimposer()
    sup.set_atoms(ref_at, mob_at)
    rot, tran = sup.rotran
    disp = {}
    for rn in common:
        m = mob[rn] @ rot + tran      # mobile CA moved into the reference frame
        disp[rn] = m - ref[rn]
    return disp


def state_hessian(chain, eps=EPS):
    """A protein chain -> (index {resnum: node_i}, coords (N,3), H). Nodes are CA."""
    residues = [(r.id[1], r) for r in chain if r.id[0] == " " and r.has_id("CA")]
    index = {rn: i for i, (rn, _) in enumerate(residues)}
    coords = np.array([r["CA"].coord.astype(float) for _, r in residues])
    springs = _typed_springs(residues)
    return index, coords, anm_hessian(coords, springs, eps)


def transmission_margin(g_D0, g_I0, g_DL, g_IL):
    """(g_IL - g_DL) - (g_I0 - g_D0): ligand-dependent gain toward the native DBD direction.

    Same double-difference topology as ddG_coup, so a positive m_trans means the ligand ADDS
    pocket->DBD transmission in the induced state relative to what it does in the dead state."""
    return (g_IL - g_DL) - (g_I0 - g_D0)


# --- self test: a straight chain, pocket at one end, DBD at the other, connected by a path ---
if __name__ == "__main__":
    # 12 nodes on a line, spacing 3.8 A (CA-CA). Springs only between neighbours = a 1D chain.
    N = 12
    coords = np.array([[3.8 * i, 0.0, 0.0] for i in range(N)])
    full = {(i, i + 1): 5.0 for i in range(N - 1)}
    index = {i: i for i in range(N)}

    # pocket node 0 pushed ALONG the chain (+x, the stiff longitudinal direction); DBD node 11 read
    # along +x. A transverse (y) push would be a zero mode of a collinear chain and carry no signal -
    # correct physics, useless test. Longitudinal motion is what the springs actually transmit.
    u = np.zeros(3 * N); u[0] = 1.0
    v = np.zeros(3 * N); v[3*11 + 0] = 1.0

    H_full = anm_hessian(coords, full)
    g_connected = directed_gain(H_full, v, u)

    cut = dict(full); del cut[(5, 6)]                 # sever the path in the middle
    g_cut = directed_gain(anm_hessian(coords, cut), v, u)

    print("connected chain g = %.4f" % g_connected)
    print("severed  chain g = %.4f" % g_cut)
    assert abs(g_connected) > 10 * abs(g_cut), "cutting the path must collapse the gain"

    # double difference: only the IL state has the extra stiffening that helps transmission
    gD0 = gI0 = gDL = g_cut
    gIL = g_connected
    m = transmission_margin(gD0, gI0, gDL, gIL)
    print("m_trans = %.4f (should be > 0: ligand adds transmission only in induced state)" % m)
    assert m > 0
    print("OK")
