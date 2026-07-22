"""The structure manifest schema for cross-scaffold pretraining. Each PDB state is one StructureEntry
with the metadata QC actually needs - not a single n_holo count.

The current data/atf_structure_db.csv stores only per-scaffold counts + a few holo/DNA PDB ids; it does
NOT store apo PDB ids, the native effector, resolution, construct mutations, oligomeric state or chain
mapping. So a StructureEntry can only be PARTIALLY populated from the CSV; the missing fields are
marked pending and filled by a PDB re-query + QC pass. Nothing is guessed.
"""
from dataclasses import dataclass, field
from typing import Optional

# topology must be kept: an inducer-release TF and a corepressor-binding TF are opposite directions and
# must never share one directionless label (structure.py already rejects mixing them).
INDUCER_RELEASE = "inducer_release"
COREPRESSOR_BINDING = "corepressor_binding"

# conservatively curated metadata; anything not here is "unknown" and flagged by the audit, not guessed.
SCAFFOLD_META = {
    "TetR": ("TetR_family", INDUCER_RELEASE), "QacR": ("TetR_family", INDUCER_RELEASE),
    "TtgR": ("TetR_family", INDUCER_RELEASE), "RamR": ("TetR_family", INDUCER_RELEASE),
    "EthR": ("TetR_family", INDUCER_RELEASE), "KstR": ("TetR_family", INDUCER_RELEASE),
    "LacI": ("LacI_family", INDUCER_RELEASE), "PurR": ("LacI_family", COREPRESSOR_BINDING),
    "AraC": ("AraC_family", None),            # activator; topology not a simple release/bind - curate
    "CatM": ("LysR_family", None), "BenM": ("LysR_family", None),
    "MarR": ("MarR_family", INDUCER_RELEASE), "BmrR": ("MerR_family", None),
}


@dataclass
class StructureEntry:
    scaffold_id: str
    state: str                         # apo / holo / dna / ternary
    pdb_id: str
    family_id: Optional[str] = None
    topology_mode: Optional[str] = None
    assembly_id: Optional[str] = None
    ligand_resnames: tuple = ()
    native_effector_id: Optional[str] = None
    protein_chains: tuple = ()
    resolution: Optional[float] = None
    sequence_identity: Optional[float] = None
    construct_mutations: tuple = ()
    canonical_mapping: Optional[str] = None      # path to the frozen residue mapping, once built
    qc_status: str = "pending"                   # pending / pass / fail
    exclusion_reasons: tuple = ()

    @classmethod
    def partial(cls, scaffold_id, state, pdb_id):
        fam, topo = SCAFFOLD_META.get(scaffold_id, (None, None))
        reasons = () if scaffold_id in SCAFFOLD_META else ("scaffold metadata not curated",)
        return cls(scaffold_id=scaffold_id, state=state, pdb_id=pdb_id, family_id=fam,
                   topology_mode=topo, qc_status="pending", exclusion_reasons=reasons)
