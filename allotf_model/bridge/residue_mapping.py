"""Canonical residue indexing - the single most error-prone place in the whole bridge.

Every per-residue object (mutation encoding, region masks, communication graph, native response, the
candidate graph) MUST align to `scaffold_id + chain_id + canonical_index`, never to a raw PDB residue
number. PDB numbers carry insertion codes, gaps, engineered tags and apo/holo offsets that silently
break alignment. The canonical index is assigned once per scaffold from its reference structure and
frozen; a ResidueKey keeps the round-trip back to the PDB for provenance.
"""
from dataclasses import dataclass
import json

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.simplefilter("ignore", PDBConstructionWarning)
_PARSER = PDBParser(QUIET=True)


@dataclass(frozen=True)
class ResidueKey:
    scaffold_id: str
    canonical_index: int          # 0..N_res-1, stable per scaffold
    chain_id: str
    pdb_resseq: int
    insertion_code: str = " "


class ResidueMapping:
    """canonical_index <-> (chain_id, pdb_resseq, insertion_code) for one scaffold, built ONCE from
    its reference structure and reused by every extractor."""

    def __init__(self, scaffold_id, keys):
        self.scaffold_id = scaffold_id
        self.keys = list(keys)                                  # ordered by canonical_index
        self._by_pdb = {(k.chain_id, k.pdb_resseq, k.insertion_code): k.canonical_index for k in keys}

    @property
    def n_res(self):
        return len(self.keys)

    def canonical(self, chain_id, pdb_resseq, insertion_code=" "):
        """-> canonical index, or None if this residue is not in the frozen mapping (never guess)."""
        return self._by_pdb.get((chain_id, pdb_resseq, insertion_code))

    @classmethod
    def from_structure(cls, scaffold_id, pdb_path, chains=None):
        """Assign canonical indices to every standard residue with a CA, in structure order."""
        model = next(iter(_PARSER.get_structure(scaffold_id, pdb_path)))
        keys = []
        for ch in model:
            if chains is not None and ch.id not in chains:
                continue
            for r in ch:
                if r.id[0] == " " and r.has_id("CA"):
                    keys.append(ResidueKey(scaffold_id, len(keys), ch.id, r.id[1], r.id[2] or " "))
        if not keys:
            raise ValueError("no standard residues with CA in %s (chains=%s)" % (pdb_path, chains))
        return cls(scaffold_id, keys)

    def save(self, path):
        rows = [dict(canonical_index=k.canonical_index, chain_id=k.chain_id,
                     pdb_resseq=k.pdb_resseq, insertion_code=k.insertion_code) for k in self.keys]
        json.dump({"scaffold_id": self.scaffold_id, "residues": rows}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        keys = [ResidueKey(d["scaffold_id"], r["canonical_index"], r["chain_id"],
                           r["pdb_resseq"], r["insertion_code"]) for r in d["residues"]]
        return cls(d["scaffold_id"], keys)
