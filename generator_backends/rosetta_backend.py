"""Rosetta local refiner - NOT a generator. Takes a proposed conformer and does local backbone /
rotamer relaxation around the pocket so co-folded/docked poses are physically cleaned up before the
ensemble features are read. Available when PyRosetta is importable (the WSL `rosetta` env).
"""


class RosettaRefiner:
    name = "rosetta"

    def available(self):
        try:
            import pyrosetta  # noqa: F401
            return True
        except Exception:
            return False

    def refine(self, pdb_path, pocket_resnums, ligand_params, out_path):
        """Local FastRelax restricted to the pocket (backbone + sidechains near the ligand), jumps
        and the rest frozen; ligand internal torsions free. Returns the refined PDB path.

        Contract (run in the rosetta env): init with the ligand params, load the pose, build a
        MoveMap that opens only pocket residues, FastRelax, dump. Fabricates nothing - if PyRosetta
        is absent this refiner is simply skipped by the orchestrator."""
        if not self.available():
            raise RuntimeError("PyRosetta not available; run the refiner in the rosetta env")
        raise NotImplementedError(
            "Rosetta refinement is stubbed: implement pocket-restricted FastRelax with the ligand "
            "params. The orchestrator skips refinement when this is unavailable.")
