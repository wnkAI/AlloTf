"""The data contract that flows into the model: three residue-level state graphs that share their
protein nodes, so a per-residue difference h_lig - h_apo is well defined.

Residue nodes come FIRST in every state, in the same order, so residue i is the same residue in apo,
lig and dna. Ligand nodes exist only in lig / lig_dna; DNA nodes only in dna / lig_dna. Region masks,
the physics resolvent channel, and the pre-computed pocket / DBD-geometry features ride along on the
graph so the heads can mix learned embeddings with physical proxies (never used as labels).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

# node type ids
RESIDUE, LIGAND, DNA, METAL = 0, 1, 2, 3
STATES = ("apo", "lig", "dna", "lig_dna")          # lig_dna is optional
REGIONS = ("pocket", "hinge", "dbd", "dna_interface")


@dataclass
class StateGraph:
    x: torch.Tensor            # (N, Fx) invariant node features
    pos: torch.Tensor          # (N, 3) coordinates
    edge_index: torch.Tensor   # (2, E) long
    edge_attr: torch.Tensor    # (E, Fe)
    node_type: torch.Tensor    # (N,) long, one of RESIDUE/LIGAND/DNA/METAL
    n_res: int                 # residue nodes are rows [0:n_res], aligned across states

    def to(self, device):
        return StateGraph(self.x.to(device), self.pos.to(device), self.edge_index.to(device),
                          self.edge_attr.to(device), self.node_type.to(device), self.n_res)


@dataclass
class MultiStateGraph:
    states: Dict[str, StateGraph]                  # keys subset of STATES; 'apo','lig','dna' required
    region_masks: Dict[str, torch.Tensor]          # region -> (n_res,) bool
    resolvent: torch.Tensor                        # (n_res,) physics pocket->DBD gain per residue
    physics: Dict[str, torch.Tensor] = field(default_factory=dict)   # binding-head proxies
    geom: Dict[str, torch.Tensor] = field(default_factory=dict)      # switch-head structural features
    topology_sign: int = +1                        # +1 ligand-induced release, -1 ligand-enhanced binding
    tf_id: Optional[str] = None                    # for within-TF ranking / family splits
    n_res: int = 0

    def __post_init__(self):
        for s in ("apo", "lig", "dna"):
            if s not in self.states:
                raise ValueError("MultiStateGraph requires state '%s'" % s)
        self.n_res = self.states["apo"].n_res
        for s, g in self.states.items():
            if g.n_res != self.n_res:
                raise ValueError("state %s has %d residue nodes, expected %d (residues must align "
                                 "across states for the difference module)" % (s, g.n_res, self.n_res))

    def to(self, device):
        return MultiStateGraph(
            states={k: v.to(device) for k, v in self.states.items()},
            region_masks={k: v.to(device) for k, v in self.region_masks.items()},
            resolvent=self.resolvent.to(device),
            physics={k: v.to(device) for k, v in self.physics.items()},
            geom={k: v.to(device) for k, v in self.geom.items()},
            topology_sign=self.topology_sign, tf_id=self.tf_id)


def collate(samples):
    """Trivial list batching: the model loops samples and stacks head outputs. Graphs have different
    node/edge counts and different present states, so a padded tensor batch would waste more than it
    saves at this scale; a per-sample loop keeps the equivariant encoder simple and correct."""
    return list(samples)
