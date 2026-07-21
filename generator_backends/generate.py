"""Orchestrate the design-time generation pipeline:

    generator (OpenDDE primary, Protenix backup, docking for pose expansion)
        -> optional Rosetta local refinement
        -> ensemble features (structural summary, confidence-masked)

The output ensemble features feed the bridge as physics/pose aux - never as a functional score. The
functional decision (does this candidate reproduce the native DNA-release response) is AlloTransfer's.
"""
from .opendde_backend import OpenDDEBackend
from .protenix_backend import ProtenixBackend
from .docking_backend import DockingBackend
from .rosetta_backend import RosettaRefiner
from .ensemble_features import ensemble_features


def default_backends():
    """Preference order; only the available ones are used. OpenDDE primary, Protenix backup."""
    return [OpenDDEBackend(), ProtenixBackend(), DockingBackend()]


def generate_ensemble(scaffold_pdb, mutations, ligand_smiles, n_seeds=8, backends=None, refiner=None):
    """-> (conformers, ensemble_feature_dict). Uses the first available backend; refines with Rosetta
    when available. Raises if NO backend is available (never fabricates)."""
    backends = backends or default_backends()
    usable = [b for b in backends if b.available()]
    if not usable:
        raise RuntimeError(
            "no conformation generator available. Install OpenDDE (OPENDDE_HOME), Protenix "
            "(PROTENIX_PYTHON) or a docking exe; this pipeline does not fabricate structures.")
    conformers = usable[0].generate(scaffold_pdb, mutations, ligand_smiles, n_seeds)

    refiner = refiner or RosettaRefiner()
    if refiner.available():
        for c in conformers:
            pdb = c.meta.get("pdb_path")
            if pdb:
                c.meta["refined"] = refiner.refine(pdb, list(c.pocket_idx),
                                                   c.meta.get("ligand_params"), pdb + ".refined.pdb")
    return conformers, ensemble_features(conformers)
