"""Held-out splits that actually test generalisation, fixed once.

The only splits that mean anything for this model are the ones that hold out a whole GROUP the model
never saw: a TF family, a scaffold, a ligand chemotype, or a mutated position. Random splits leak
near-duplicate mutants of the same protein across train/test and inflate everything.
"""
from collections import defaultdict


def grouped_folds(items, key):
    """Leave-one-group-out. items: list of records; key(record) -> group id.
    Yields (held_out_group, train_idx, test_idx)."""
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[key(it)].append(i)
    for g, test_idx in groups.items():
        train_idx = [i for i in range(len(items)) if i not in set(test_idx)]
        yield g, train_idx, test_idx


def split(items, by):
    """by: one of 'family','scaffold','ligand_class','mutation_position'. Returns list of folds."""
    keymap = {
        "family": lambda r: r.get("family"),
        "scaffold": lambda r: r.get("scaffold"),
        "ligand_class": lambda r: r.get("ligand_class"),
        "mutation_position": lambda r: r.get("mutation_position"),
    }
    if by not in keymap:
        raise ValueError("unknown split '%s'; use one of %s" % (by, list(keymap)))
    return list(grouped_folds(items, keymap[by]))
