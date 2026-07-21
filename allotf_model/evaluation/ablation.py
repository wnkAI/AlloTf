"""Ablations that justify each design choice. Each entry flips ONE thing; the training/eval runner
applies it and reports the drop on AUC_sensor_vs_binderonly (the metric that separates a real
allosteric model from a pocket scorer).

Kept as a spec, not a copy of the model: the runner reads these flags and builds the ablated model,
so there is one source of truth for what each ablation means.
"""

ABLATIONS = {
    "full": {},                                          # the model as designed
    "sum_gate": {"gate": "sum"},                         # additive instead of multiplicative gate
    "no_contrast": {"state_contrast": False},            # heads see per-state embeddings, no dh
    "no_mechanistic": {"lambda_mech": 0.0},              # drop the mechanistic constraints
    "single_state": {"states": ["lig"]},                 # only the ligand state (no apo/DNA)
    "no_path_head": {"path_gate": False},                # remove the P_path factor from the gate
    "no_resolvent": {"resolvent": False},                # zero the physics pocket->DBD channel
    "unshared_encoder": {"share_encoder": False},        # separate encoder per state (breaks attribution)
    "no_ranking": {"lambda_rank": 0.0},                  # drop within-TF ranking supervision
}

# what each ablation is meant to demonstrate (goes in the paper table)
CLAIMS = {
    "sum_gate": "multiplicative gating is needed - a binder-only should not be rescued by one factor",
    "no_contrast": "explicit state difference matters - the model must read change, not identity",
    "no_mechanistic": "physics constraints keep the heads consistent",
    "single_state": "multi-state input is required to see switching at all",
    "no_path_head": "path preservation is a distinct necessary condition",
    "no_resolvent": "the physics communication channel adds signal",
    "unshared_encoder": "weight sharing is what lets differences be attributed to state",
    "no_ranking": "within-TF ranking carries the transferable supervision",
}


def apply(base_config, name):
    """Return a copy of base_config with the named ablation applied."""
    if name not in ABLATIONS:
        raise ValueError("unknown ablation '%s'; known: %s" % (name, list(ABLATIONS)))
    cfg = dict(base_config)
    cfg.update(ABLATIONS[name])
    cfg["ablation"] = name
    return cfg
