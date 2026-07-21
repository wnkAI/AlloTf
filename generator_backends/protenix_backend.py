"""Protenix backend - independent all-atom co-folding, used to cross-check OpenDDE and as a stable
backup generator. More mature interface with explicit template / MSA / pocket-constraint support, so
it is the lower-risk fallback. Wraps the external tool; fabricates nothing when absent.
"""
import os

from .base import GeneratorBackend


class ProtenixBackend(GeneratorBackend):
    name = "protenix"

    def __init__(self, env_python=None):
        # e.g. /home/wnk/mamba/envs/protenix/bin/python
        self.env_python = env_python or os.environ.get("PROTENIX_PYTHON")

    def available(self):
        return bool(self.env_python and os.path.exists(self.env_python))

    def generate(self, scaffold_pdb, mutations, ligand_smiles, n_seeds=8):
        self._require()
        # integration contract: call Protenix (with template=scaffold, optional pocket constraints on
        # the mutated pocket) to co-fold the mutant + ligand for n_seeds seeds, parse each complex into
        # a Conformer(source='protenix', meta={version, confidence}). Used to CONFIRM OpenDDE poses.
        raise NotImplementedError(
            "Protenix integration is stubbed: set PROTENIX_PYTHON and implement the call + parser.")
