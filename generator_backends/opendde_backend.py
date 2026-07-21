"""OpenDDE backend - primary all-atom co-folding generator (protein + ligand + nucleic acid).

Wraps the external OpenDDE CLI. It does not ship here; available() reports honestly and generate()
refuses to fabricate a structure when the tool is absent. Wire OPENDDE_HOME to the install.

OpenDDE is a PREVIEW release: CLI/checkpoint/output formats may change between versions, so this
wrapper pins the command in one place and records the version in each Conformer's meta.
"""
import os
import shutil

from .base import GeneratorBackend


class OpenDDEBackend(GeneratorBackend):
    name = "opendde"

    def __init__(self, home=None, checkpoint=None):
        self.home = home or os.environ.get("OPENDDE_HOME")
        self.checkpoint = checkpoint or os.environ.get("OPENDDE_CHECKPOINT")

    def available(self):
        return bool(self.home and os.path.isdir(self.home)
                    and (shutil.which("opendde") or os.path.exists(os.path.join(self.home, "opendde"))))

    def generate(self, scaffold_pdb, mutations, ligand_smiles, n_seeds=8):
        self._require()
        # integration contract (run when OPENDDE_HOME is set):
        #   1. apply `mutations` to the scaffold sequence;
        #   2. call the OpenDDE CLI to co-fold (mutant sequence + ligand_smiles [+ operator DNA])
        #      for n_seeds seeds, writing all-atom complexes;
        #   3. parse each output into a Conformer (ca_coords, ligand_coords, pocket_idx, confidence,
        #      seed, source='opendde', meta={version, plddt/ipTM}).
        raise NotImplementedError(
            "OpenDDE integration is stubbed: set OPENDDE_HOME and implement the CLI call + parser. "
            "This wrapper never fabricates conformers.")
