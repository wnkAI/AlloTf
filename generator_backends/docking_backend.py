"""Docking backend - fast pose expansion around a fixed backbone. Cheap way to enlarge the ligand-
pose part of the ensemble (e.g. smina/vina/rdkit-ETKDG + rigid docking) when full co-folding is too
slow. Backbone stays fixed, so it complements OpenDDE/Protenix rather than replacing them.
"""
import shutil

from .base import GeneratorBackend


class DockingBackend(GeneratorBackend):
    name = "docking"

    def __init__(self, exe="smina"):
        self.exe = exe

    def available(self):
        return shutil.which(self.exe) is not None

    def generate(self, scaffold_pdb, mutations, ligand_smiles, n_seeds=8):
        self._require()
        # integration contract: embed ligand conformers (rdkit), dock into the (mutated) pocket with
        # `self.exe`, parse the top n_seeds poses into Conformers (backbone from scaffold, ligand from
        # the pose, confidence from the docking score mapped to [0,1], source='docking').
        raise NotImplementedError(
            "Docking integration is stubbed: implement the %s call + pose parser." % self.exe)
