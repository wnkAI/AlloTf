"""Held-out splits that actually test cross-biosensor generalisation. Random splits leak near-identical
mutants of one TF across train and test, so only whole-group hold-outs mean anything: a whole scaffold,
a whole structural family, or a whole ligand chemotype. The target TF never appears in training.
"""
from collections import defaultdict


def _scaf(s):
    return s.scaffold_id


def _family(s):
    return s.provenance.get("family", s.scaffold_id)


def _grouped(samples, key):
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[key(s)].append(i)
    for g, test in groups.items():
        train = [i for i in range(len(samples)) if i not in set(test)]
        yield g, train, test


def leave_one_scaffold_out(samples):
    return list(_grouped(samples, _scaf))


def leave_one_family_out(samples):
    return list(_grouped(samples, _family))


def leave_one_ligand_class_out(samples):
    return list(_grouped(samples, lambda s: s.ligand_id))
