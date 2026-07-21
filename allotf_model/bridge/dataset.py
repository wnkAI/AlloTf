"""A torch Dataset over TransferSamples. The scaffold-level context (canonical mapping, WT scaffold
PDB, region definitions, frozen native reference) is shared across all candidates of one scaffold and
loaded once; each row's spec carries only the candidate-specific parts (mutation, ligand, labels).
"""
from torch.utils.data import Dataset

from .build_sample import build_transfer_sample


class ScaffoldContext:
    """Everything shared by all candidates of one scaffold - loaded once."""

    def __init__(self, scaffold_id, scaffold_pdb, mapping, regions, native_reference, confidence=None,
                 pair_features=None):
        self.scaffold_id = scaffold_id
        self.scaffold_pdb = scaffold_pdb
        self.mapping = mapping
        self.regions = regions
        self.native_reference = native_reference
        self.confidence = confidence
        self.pair_features = pair_features


class TransferDataset(Dataset):
    """specs: list of dicts with keys candidate_id, ligand_id, ligand_smiles, mutations {ci: AA},
    labels {...}, provenance. contexts: {scaffold_id: ScaffoldContext}."""

    def __init__(self, specs, contexts):
        self.specs = list(specs)
        self.contexts = contexts

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        s = self.specs[i]
        ctx = self.contexts[s["scaffold_id"]]
        return build_transfer_sample(
            sample_id=s.get("sample_id", "%s__%s__%s" % (ctx.scaffold_id, s["candidate_id"], s["ligand_id"])),
            candidate_id=s["candidate_id"], ligand_id=s["ligand_id"],
            scaffold_pdb=ctx.scaffold_pdb, mapping=ctx.mapping, mutations=s.get("mutations", {}),
            regions=ctx.regions, native_reference=ctx.native_reference,
            ligand_smiles=s["ligand_smiles"], physics=s.get("physics"), labels=s.get("labels"),
            confidence=ctx.confidence, pair_features=ctx.pair_features, provenance=s.get("provenance"))
