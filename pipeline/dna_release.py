"""Surface the DNA readout per candidate for rank.

Both quantities are already computed as double differences inside state_builder, where the six
poses actually live:

    E_DNA_X_D = E(D.DNA) - E(D0)                                apo operator competence (affinity)
    S_release = [E(I.DNA) - E(I0)] - [E(D.DNA) - E(D0)]         raw, topology sign applied in rank

This stage does not recompute them and does not touch structures; it maps them per candidate so
rank can gate on them. The topology sign stays unapplied here on purpose - rank.release_sign()
applies it exactly once, and squaring it would silently un-invert a corepressor.

The old standalone TetR/QacR DBD-DNA geometry check (which read PDBs directly, with hardcoded
paths, and had a `run(tf, dna_pdb, ...)` signature incompatible with the stage contract) now lives
in tools/legacy_dna_geometry_check.py.
"""


def run(ctx):
    """requires ctx['candidate_states'];  produces ctx['dna_scores']"""
    scores = {}
    dropped = {}
    for cid, st in ctx["candidate_states"].items():
        s_rel = st.get("S_release")
        e_dna = st.get("E_DNA_X_D")
        if s_rel is None or e_dna is None:
            # fail closed: a candidate whose DNA states did not build is dropped, never carried
            # with zeros - a zero reads as "neutral" and would sail through the release gate
            dropped[cid] = "missing %s" % ("S_release" if s_rel is None else "E_DNA_X_D")
            continue
        scores[cid] = {"E_DNA_X_D": float(e_dna), "S_release": float(s_rel)}
    if not scores:
        raise RuntimeError("no candidate produced a DNA readout (%d dropped): the DNA states did "
                           "not build" % len(dropped))
    return {"dna_scores": scores, "dna_dropped": dropped}
