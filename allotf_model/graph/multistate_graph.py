"""The data contract that flows into the model: three residue-level state graphs that share their
protein nodes, so a per-residue difference h_lig - h_apo is well defined.

Residue nodes come FIRST in every state, rows [0:n_res], and each carries a stable residue_id
(e.g. chain*100000+resnum). The identity AND ORDER of those ids is enforced to be identical across
states - equal length is not enough, or a permutation would silently corrupt h_lig - h_apo. Ligand
nodes exist only in lig / lig_dna; DNA nodes only in dna / lig_dna. Region masks, the physics
resolvent channel, and pre-computed pocket / DBD-geometry features ride along so the heads can mix
learned embeddings with physical proxies (never used as labels).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

RESIDUE, LIGAND, DNA, METAL = 0, 1, 2, 3
STATES = ("apo", "lig", "dna", "lig_dna")
REGIONS = ("pocket", "hinge", "dbd", "dna_interface")


@dataclass
class StateGraph:
    x: torch.Tensor            # (N, Fx) invariant node features
    pos: torch.Tensor          # (N, 3) coordinates
    edge_index: torch.Tensor   # (2, E) long, interpreted as [receiver i, sender j]  (i <- j)
    edge_attr: torch.Tensor    # (E, Fe) invariant edge features
    node_type: torch.Tensor    # (N,) long
    residue_ids: torch.Tensor  # (n_res,) long stable ids for the residue nodes [0:n_res]
    n_res: int

    def __post_init__(self):
        N = self.x.shape[0]
        if not (0 < self.n_res <= N):
            raise ValueError("n_res=%d must be in (0, N=%d]" % (self.n_res, N))
        if self.residue_ids.shape[0] != self.n_res:
            raise ValueError("residue_ids has %d entries, n_res=%d" % (self.residue_ids.shape[0], self.n_res))
        if self.pos.shape[0] != N or self.node_type.shape[0] != N:
            raise ValueError("pos/node_type must have N=%d rows" % N)
        if self.edge_index.numel() and int(self.edge_index.max()) >= N:
            raise ValueError("edge_index references a node >= N")
        if (self.node_type[: self.n_res] != RESIDUE).any():
            raise ValueError("the first n_res nodes must all be RESIDUE nodes")

    def to(self, device):
        return StateGraph(self.x.to(device), self.pos.to(device), self.edge_index.to(device),
                          self.edge_attr.to(device), self.node_type.to(device),
                          self.residue_ids.to(device), self.n_res)


@dataclass
class MultiStateGraph:
    states: Dict[str, StateGraph]
    region_masks: Dict[str, torch.Tensor]          # region -> (n_res,) bool
    resolvent: torch.Tensor                        # (n_res,)
    physics: Dict[str, torch.Tensor] = field(default_factory=dict)
    geom: Dict[str, torch.Tensor] = field(default_factory=dict)
    topology_sign: int = +1                        # +1 release, -1 enhanced binding
    tf_id: Optional[str] = None
    n_res: int = 0

    def __post_init__(self):
        for s in ("apo", "lig", "dna"):
            if s not in self.states:
                raise ValueError("MultiStateGraph requires state '%s'" % s)
        self.n_res = self.states["apo"].n_res
        ref_ids = self.states["apo"].residue_ids
        for s, g in self.states.items():
            if g.n_res != self.n_res or not torch.equal(g.residue_ids, ref_ids):
                raise ValueError("state '%s' residue ids/order differ from apo: residues must align "
                                 "across states for the difference module" % s)
        for r, m in self.region_masks.items():
            if m.shape[0] != self.n_res:
                raise ValueError("region mask '%s' has %d entries, n_res=%d" % (r, m.shape[0], self.n_res))
        if self.resolvent.shape[0] != self.n_res:
            raise ValueError("resolvent has %d entries, n_res=%d" % (self.resolvent.shape[0], self.n_res))

    def to(self, device):
        return MultiStateGraph(
            states={k: v.to(device) for k, v in self.states.items()},
            region_masks={k: v.to(device) for k, v in self.region_masks.items()},
            resolvent=self.resolvent.to(device),
            physics={k: v.to(device) for k, v in self.physics.items()},
            geom={k: v.to(device) for k, v in self.geom.items()},
            topology_sign=self.topology_sign, tf_id=self.tf_id)


def collate(samples):
    """List batching: the model loops samples and stacks head outputs. Graphs differ in node/edge
    count and in which states are present, so a per-sample loop keeps the encoder simple and correct."""
    return list(samples)
