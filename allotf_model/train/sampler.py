"""Hierarchical, multi-scaffold sampling: scaffold -> ligand -> functional class -> candidate.

The point is generalisation, so no single transcription factor may dominate. A large-data scaffold
(TtgR) must not swamp the batch just because it has 17k variants. Each scaffold gets equal total
probability; within a scaffold each ligand is equal; within a ligand each functional class is equal;
within a class each candidate is equal. This is implemented as a per-sample weight equal to the
product of the inverse counts at each level, so batches mix scaffolds and see rare ligands / the
functional_sensor class often enough for the within-ligand ranking to have real pairs.
"""
from collections import defaultdict

import torch
from torch.utils.data import Sampler


def _lig(s):
    return s["ligand_id"] if isinstance(s, dict) else s.ligand_id


def _scaf(s):
    return s["scaffold_id"] if isinstance(s, dict) else s.scaffold_id


def _cls(s):
    if isinstance(s, dict):
        return s.get("functional_class")
    return int(s.functional_class) if bool(s.functional_class_label_mask) else -1


class HierarchicalScaffoldSampler(Sampler):
    def __init__(self, specs, num_samples=None, seed=0):
        n_scaf = defaultdict(int)                      # candidates per scaffold
        n_lig = defaultdict(int)                       # per (scaffold, ligand)
        n_cls = defaultdict(int)                       # per (scaffold, ligand, class)
        cells = []
        for s in specs:
            sc, lg, cl = _scaf(s), _lig(s), _cls(s)
            cells.append((sc, lg, cl))
            n_scaf[sc] += 1; n_lig[(sc, lg)] += 1; n_cls[(sc, lg, cl)] += 1
        n_ligands = defaultdict(set)                   # distinct ligands per scaffold
        n_classes = defaultdict(set)                   # distinct classes per (scaffold, ligand)
        for sc, lg, cl in cells:
            n_ligands[sc].add(lg); n_classes[(sc, lg)].add(cl)

        w = []
        for sc, lg, cl in cells:
            # P(i) = 1/S * 1/|ligands_s| * 1/|classes_{s,l}| * 1/n_{s,l,c}; S is constant so drop it
            w.append(1.0 / (len(n_ligands[sc]) * len(n_classes[(sc, lg)]) * n_cls[(sc, lg, cl)]))
        self.weights = torch.tensor(w, dtype=torch.double)
        self.num_samples = num_samples or len(specs)
        self.gen = torch.Generator().manual_seed(seed)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True,
                                      generator=self.gen).tolist())

    def __len__(self):
        return self.num_samples
