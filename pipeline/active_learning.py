"""Optional: fold wet-lab results back in. Target-specific, not a general model.

After round 1 you have (S_i, L_target) -> y_i for a few dozen designs. Those points are worth more
than the whole Sensor-seq corpus for THIS target, because they are on-target.
"""

def update(round_results, cfg):
    """TODO(E): fit f_target(S) on measured designs; propose round 2 by
    uncertainty + Pareto + diversity."""
    raise NotImplementedError
