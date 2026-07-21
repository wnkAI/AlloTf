"""Design-time conformation generators, behind one interface. A backend proposes candidate
structures for (scaffold, mutation, target ligand); it NEVER scores allosteric function - that is
AlloTransfer's job. Backend confidence is a structural confidence, never a functional score.

The pipeline is: generator (OpenDDE / Protenix / docking) -> local refinement (Rosetta) -> ensemble
features -> AlloTransfer ranking. Each backend returns an ensemble of Conformers; the functional
ranker decides which one actually produces the distal DNA-release response.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Conformer:
    ca_coords: np.ndarray          # [N_res, 3]
    ligand_coords: np.ndarray      # [N_atom, 3]
    pocket_idx: np.ndarray         # [n_pocket] residue indices
    confidence: float              # STRUCTURAL confidence in [0,1] (not a functional score)
    seed: int = 0
    source: str = "unknown"
    meta: dict = field(default_factory=dict)


class GeneratorBackend(ABC):
    """One backend = one way to propose conformers. Subclasses wrap an external tool and MUST fail
    loudly if it is unavailable rather than fabricate a structure."""

    name = "base"

    @abstractmethod
    def available(self) -> bool:
        """True only if the underlying tool/weights are actually installed and runnable."""

    @abstractmethod
    def generate(self, scaffold_pdb, mutations, ligand_smiles, n_seeds=8) -> list:
        """(scaffold, {resnum: AA}, ligand) -> list[Conformer]. Raises if not available()."""

    def _require(self):
        if not self.available():
            raise RuntimeError(
                "%s backend is not available (tool/weights not installed). This backend does not "
                "fabricate structures; install it or use another backend." % self.name)
