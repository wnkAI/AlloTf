"""Joint sequence + rotamer search by simulated annealing.

Sequence and side-chain conformation must be searched TOGETHER. Picking ARG at a position means
nothing until its chi1..chi4 are placed - the guanidinium either reaches the ligand carboxylate or
it does not, and that is a rotamer question. Optimising sequence first and packing afterwards is
how a design ends up with the right residue in the wrong place.

The energy function is injected, so this module is testable on a synthetic landscape without any
structure, and swappable (internal terms now, Rosetta/FoldX later) without touching the search.

Scale: 11 positions x ~6 residues x up to 81 rotamers. Enumeration is hopeless; SA with restarts
and a move set that mixes residue swaps and rotamer jumps is not.
"""
import math
import numpy as np


class SearchSpace:
    """positions: [pos_id];  allowed: {pos_id: [resname]};  rotamers_of: resname -> [chis]"""

    def __init__(self, positions, allowed, rotamers_of, wt=None):
        self.positions = list(positions)
        self.allowed = {p: list(allowed[p]) for p in self.positions}
        self.rotamers_of = rotamers_of
        self.wt = wt or {}
        for p in self.positions:
            if not self.allowed[p]:
                raise ValueError("position %s has no allowed residue" % p)

    def size(self):
        n = 1
        for p in self.positions:
            n *= sum(max(len(self.rotamers_of(a)), 1) for a in self.allowed[p])
        return n

    def random_state(self, rng):
        s = {}
        for p in self.positions:
            aa = self.allowed[p][rng.randint(len(self.allowed[p]))]
            rots = self.rotamers_of(aa) or [()]
            s[p] = (aa, rots[rng.randint(len(rots))])
        return s

    def wt_state(self, rng):
        s = {}
        for p in self.positions:
            aa = self.wt.get(p, self.allowed[p][0])
            if aa not in self.allowed[p]:
                aa = self.allowed[p][0]
            rots = self.rotamers_of(aa) or [()]
            s[p] = (aa, rots[rng.randint(len(rots))])
        return s


def _move(space, state, rng, p_swap=0.35):
    """Two move types: change the residue (big jump), or re-place the current one (local).
    Residue swaps alone get stuck - a new residue in a bad rotamer looks worse than the old one
    and is rejected, so the search never discovers what that residue could do.
    """
    p = space.positions[rng.randint(len(space.positions))]
    aa, chis = state[p]
    if rng.rand() < p_swap and len(space.allowed[p]) > 1:
        new_aa = aa
        while new_aa == aa:
            new_aa = space.allowed[p][rng.randint(len(space.allowed[p]))]
        rots = space.rotamers_of(new_aa) or [()]
        return p, (new_aa, rots[rng.randint(len(rots))])
    rots = space.rotamers_of(aa) or [()]
    if len(rots) <= 1:
        new_aa = space.allowed[p][rng.randint(len(space.allowed[p]))]
        r2 = space.rotamers_of(new_aa) or [()]
        return p, (new_aa, r2[rng.randint(len(r2))])
    return p, (aa, rots[rng.randint(len(rots))])


def anneal(space, energy_fn, n_steps=20000, t0=5.0, t1=0.05, seed=0,
           start="wt", energy_delta_fn=None):
    """-> (best_state, best_energy, trace). energy_delta_fn(state, pos, new) is used when the
    caller can score incrementally; otherwise the full energy is recomputed."""
    rng = np.random.RandomState(seed)
    state = space.wt_state(rng) if start == "wt" else space.random_state(rng)
    e = energy_fn(state)
    best, best_e = dict(state), e
    trace = []
    for i in range(n_steps):
        T = t0 * (t1 / t0) ** (i / max(n_steps - 1, 1))
        p, new = _move(space, state, rng)
        if energy_delta_fn is not None:
            de = energy_delta_fn(state, p, new)
            e_new = e + de
        else:
            old = state[p]
            state[p] = new
            e_new = energy_fn(state)
            de = e_new - e
            state[p] = old
        if de <= 0 or rng.rand() < math.exp(-de / max(T, 1e-9)):
            state[p] = new
            e = e_new
            if e < best_e:
                best, best_e = dict(state), e
        if i % max(n_steps // 100, 1) == 0:
            trace.append((i, T, e, best_e))
    return best, best_e, trace


def multi_start(space, energy_fn, n_restarts=8, n_steps=20000, seed=0, **kw):
    """-> list[(state, energy)] sorted, deduplicated by sequence."""
    out = {}
    for r in range(n_restarts):
        st, en, _ = anneal(space, energy_fn, n_steps=n_steps, seed=seed + r,
                           start="wt" if r == 0 else "random", **kw)
        key = seq_of(space, st)
        if key not in out or en < out[key][1]:
            out[key] = (st, en)
    return sorted(out.values(), key=lambda x: x[1])


def seq_of(space, state):
    from .aa_filter import one
    return "".join(one(state[p][0]) if state[p][0] else "-" for p in space.positions)


def mutations(space, state):
    return [(p, space.wt.get(p), state[p][0]) for p in space.positions
            if space.wt.get(p) and state[p][0] != space.wt[p]]


def diversify(space, results, max_per_cluster=3, identity=0.9):
    """Keep the search from handing back 50 versions of one mutation family."""
    kept = []
    for st, en in results:
        s = seq_of(space, st)
        n_close = sum(1 for k, _ in kept
                      if sum(a == b for a, b in zip(s, seq_of(space, k))) / len(s) >= identity)
        if n_close < max_per_cluster:
            kept.append((st, en))
    return kept


if __name__ == "__main__":
    # the search must be validated on a landscape whose optimum we KNOW, before it is ever
    # trusted on a real energy function
    POS = list(range(11))
    AAS = ["ALA", "SER", "THR", "VAL", "LEU", "ARG"]
    ROT = {"ALA": [()], "SER": [(-60,), (60,), (180,)], "THR": [(-60,), (60,), (180,)],
           "VAL": [(-60,), (60,), (180,)], "LEU": [(-60, 180), (180, 60)],
           "ARG": [(a, b, c, d) for a in (-60, 180, 60) for b in (-60, 180, 60)
                   for c in (-60, 180, 60) for d in (-90, 90, 180)]}
    TARGET_AA = {p: ("ARG" if p % 3 == 0 else "SER") for p in POS}
    TARGET_ROT = {p: (ROT[TARGET_AA[p]][0]) for p in POS}

    def energy(state):
        e = 0.0
        for p in POS:
            aa, chis = state[p]
            e += 0.0 if aa == TARGET_AA[p] else 2.0
            e += 0.0 if chis == TARGET_ROT[p] else 0.5
        return e

    space = SearchSpace(POS, {p: AAS for p in POS}, lambda a: ROT[a],
                        wt={p: "LEU" for p in POS})
    print("search space size: %.3e states" % space.size())
    best, be, tr = anneal(space, energy, n_steps=20000, seed=0)
    print("annealing: E %.2f -> %.2f   (global optimum = 0.00)" % (tr[0][2], be))
    print("recovered sequence:", seq_of(space, best))
    print("target sequence   :", "".join(TARGET_AA[p][0] for p in POS))
    ok = all(best[p][0] == TARGET_AA[p] for p in POS)
    print("all residues correct:", ok)
    res = multi_start(space, energy, n_restarts=6, n_steps=8000)
    print("multi-start: %d unique sequences, best E = %.2f" % (len(res), res[0][1]))
    print("after diversify:", len(diversify(space, res)))
