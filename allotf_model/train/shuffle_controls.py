"""Build the perturbed samples the mechanistic regularisation needs. The model is re-run on these and
the real must beat them by a margin (see losses.mechanistic_regularization):

  ligand_shuffled - swap in a DIFFERENT ligand; the real target ligand must score as more of a sensor;
  path_shuffled   - scramble the pocket->DBD communication edges; breaking the path must reduce the
                    coupling and the sensor score.
"""
import dataclasses

import torch


def ligand_shuffled(sample, donor):
    """A copy of `sample` carrying `donor`'s ligand graph. Everything else (scaffold, teacher) stays."""
    return dataclasses.replace(
        sample, ligand_atom_features=donor.ligand_atom_features,
        ligand_edge_index=donor.ligand_edge_index, ligand_edge_features=donor.ligand_edge_features,
        ligand_coordinates=donor.ligand_coordinates, ligand_id="SHUFFLED",
        cross_edge_index=torch.zeros(2, 0, dtype=torch.long),
        cross_edge_features=torch.zeros(0, sample.cross_edge_features.shape[1] if sample.cross_edge_features.numel() else 1))


def path_shuffled(sample, seed=0):
    """A copy with the communication senders permuted, so the pocket->DBD path no longer connects the
    pocket to the DBD as it did."""
    ce = sample.communication_edge_index
    if ce.numel() == 0:
        return sample
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(ce.shape[1], generator=g)
    shuffled = torch.stack([ce[0], ce[1][perm]])          # keep receivers, scramble senders
    return dataclasses.replace(sample, communication_edge_index=shuffled)
