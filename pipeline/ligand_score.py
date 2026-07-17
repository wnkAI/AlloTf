"""Does the target ligand PREFER the induced state?

    S_state = E_L(on X_D) - E_L(on X_I)     want > 0

Plus interpretable terms: hbonds, hydrophobic contacts, salt bridges, shape complementarity,
buried unsatisfied polars, ligand clash, buried surface area.

Two different questions live here and must not be collapsed into one number:

  target_binding  E_L_I     does it bind the induced pocket at all?
  state_preference S_state  does it bind the induced pocket BETTER than the DNA-compatible one?

A ligand can bind beautifully in both states and switch nothing (S_state ~ 0), and a ligand can
prefer I strongly while barely binding either (+100 vs +99 - the non-binder that passed the early
gates). rank.py gates them separately for exactly this reason.

S_state and ddG_coup are related but not the same measurement: S_state is an interface energy on
the ligand alone, ddG_coup is a double difference of whole-pose totals. They should agree in sign;
when they do not, the pose or the packing is unstable and the candidate is not trustworthy. That
disagreement is reported, not averaged away.
"""
from .state_builder import linkage, totals

SIGN_S_STATE = +1     # want > 0: ligand is happier on the induced backbone


def interface_terms(backend, pose, ligand_resnum=None, chain="A"):
    """Decomposed protein-ligand interface, via the same backend that scored the states."""
    return backend.interface_energy(pose)


def state_preference(e_l_d, e_l_i):
    """E_L(X_D) - E_L(X_I). Positive = the ligand prefers the induced state."""
    if e_l_d is None or e_l_i is None:
        return None
    return e_l_d - e_l_i


def consistency(s_state, ddg_coup):
    """S_state > 0 and ddG_coup < 0 are the same physical claim seen two ways.

    -> (agree: bool, note). Disagreement means the pose moved during packing or a state failed to
    converge; downstream must not average two contradictory readings into a middling score.
    """
    if s_state is None or ddg_coup is None:
        return False, "missing"
    agree = (s_state > 0) == (ddg_coup < 0)
    if agree:
        return True, "ok"
    return False, ("S_state=%.2f and ddG_coup=%.2f disagree: the interface says one state is "
                   "preferred and the double difference says the other. Pose instability - "
                   "reject rather than average." % (s_state, ddg_coup))


def score(states, cfg=None):
    """states: {state: score_terms} from state_builder.build_six, plus optional
    'interface': {'DL': float, 'IL': float} recorded when the states were built.

    -> dict(state_preference, target_binding, dG_apo, dG_lig, ddG_coupling, terms, agree)
    Fail closed: a missing state yields None, never a zero that reads as "neutral".
    """
    tot = totals(states) if not isinstance(next(iter(states.values()), None), (int, float)) else states
    link = linkage(tot)
    iface = (states.get("interface") or {}) if isinstance(states, dict) else {}
    e_l_d, e_l_i = iface.get("DL"), iface.get("IL")
    s_state = state_preference(e_l_d, e_l_i)
    ddg = link["ddG_coup"] if link else None
    agree, note = consistency(s_state, ddg)

    out = {
        "state_preference": s_state,
        "target_binding": e_l_i,
        "E_L_D": e_l_d,
        "E_L_I": e_l_i,
        "dG_apo": link["dG_apo"] if link else None,
        "dG_lig": link["dG_lig"] if link else None,
        "ddG_coupling": ddg,
        "agree": agree,
        "note": note,
        "terms": {st: (t.get("total_score") if isinstance(t, dict) else t)
                  for st, t in states.items() if st in ("D0", "I0", "DL", "IL")},
    }
    return out


def run(ctx):
    """requires ctx['candidate_states']; produces ctx['ligand_scores']"""
    out = {}
    for cid, st in ctx["candidate_states"].items():
        out[cid] = score(st, ctx.get("cfg"))
    return {"ligand_scores": out}
