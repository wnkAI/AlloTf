"""The native response teacher: a FROZEN, scaffold-level physical description of what the native
ligand does to the distal region, extracted ONCE from the WT apo and native-holo structures.

It is a fixed physical target (per-residue displacement, local contact change, neighbour count -
rotation-invariant channels), saved to disk and only loaded + aligned per candidate. It does NOT
depend on the learned encoder and does NOT drift during training, so "native response" keeps its
physical meaning. (The model's optional EMA teacher only stabilises the learned target
representation; it never redefines this reference.)

Cache layout per scaffold:
    native_references/<scaffold>/{native_response.pt, distal_mask.pt, residue_mapping.json, metadata.json}
"""
import json
import os
import warnings

import numpy as np
import torch
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from .residue_mapping import ResidueMapping

warnings.simplefilter("ignore", PDBConstructionWarning)
_PARSER = PDBParser(QUIET=True)

# per-residue response channels (all rotation/translation invariant)
CHANNELS = ("ca_displacement", "contact_count_change", "n_neighbours_apo", "n_neighbours_holo")
D_TEACHER = len(CHANNELS)
_CONTACT = 8.0     # CA-CA neighbour cutoff for the contact-change channel


def _ca_by_key(pdb_path, mapping):
    """{canonical_index: CA atom} for residues present in the frozen mapping."""
    model = next(iter(_PARSER.get_structure(mapping.scaffold_id, pdb_path)))
    out = {}
    for ch in model:
        for r in ch:
            if r.id[0] == " " and r.has_id("CA"):
                ci = mapping.canonical(ch.id, r.id[1], r.id[2] or " ")
                if ci is not None:
                    out[ci] = r["CA"]
    return out


def _contacts(coords):
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return (d < _CONTACT).sum(1) - 1                          # exclude self


def extract(apo_pdb, holo_pdb, mapping, distal_idx):
    """WT apo + native-holo -> (native_response [N_res, D_TEACHER], confidence [N_res]).

    Superposes holo onto apo on the residues shared OUTSIDE the distal region (a stable frame), then
    reads the distal response. Residues missing from either structure get zero response and zero
    confidence - never a fabricated value."""
    n = mapping.n_res
    apo, holo = _ca_by_key(apo_pdb, mapping), _ca_by_key(holo_pdb, mapping)
    shared = sorted(set(apo) & set(holo))
    distal = set(int(i) for i in distal_idx)
    frame = [i for i in shared if i not in distal] or shared          # align on the stable core
    sup = Superimposer()
    sup.set_atoms([apo[i] for i in frame], [holo[i] for i in frame])
    rot, tran = sup.rotran

    resp = torch.zeros(n, D_TEACHER)
    conf = torch.zeros(n)
    apo_xyz = {i: apo[i].coord.astype(float) for i in apo}
    holo_xyz = {i: holo[i].coord @ rot + tran for i in holo}         # holo moved into the apo frame
    shared_arr = np.array([apo_xyz[i] for i in shared])
    holo_arr = np.array([holo_xyz[i] for i in shared])
    nc_apo = dict(zip(shared, _contacts(shared_arr)))
    nc_holo = dict(zip(shared, _contacts(holo_arr)))
    for i in shared:
        disp = float(np.linalg.norm(holo_xyz[i] - apo_xyz[i]))
        resp[i] = torch.tensor([disp, float(nc_holo[i] - nc_apo[i]),
                                float(nc_apo[i]), float(nc_holo[i])])
        conf[i] = 1.0
    return resp, conf


def _unit_direction(response):
    """Per-residue unit response vector - the native DNA-release DIRECTION the design must align to."""
    return response / (response.norm(dim=1, keepdim=True) + 1e-6)


def _transmission_gain(response, distal_mask, pocket_idx):
    """g_native = ||response_distal|| / ||response_pocket|| - the scaffold's transmission efficiency."""
    dist = response[distal_mask.bool()].norm()
    if pocket_idx is None or len(pocket_idx) == 0:
        return float("nan")
    pk = response[torch.as_tensor(list(pocket_idx), dtype=torch.long)].norm()
    return float(dist / (pk + 1e-6))


class NativeReference:
    def __init__(self, mapping, response, confidence, distal_mask, metadata=None,
                 native_direction=None, native_gain=None, bottleneck_score=None, gain_tuning_mask=None,
                 pocket_idx=None):
        self.mapping = mapping
        self.response = response
        self.confidence = confidence
        self.distal_mask = distal_mask
        self.metadata = metadata or {}
        n = response.shape[0]
        # native DNA-release direction (auto from the frozen response) and scaffold gain
        self.native_direction = native_direction if native_direction is not None else _unit_direction(response)
        self.native_gain = native_gain if native_gain is not None else _transmission_gain(
            response, distal_mask, pocket_idx)
        # transduction residues that MAY be gain-tuned, and which control amplification (default: none
        # flagged until the bottleneck pass fills them - never guessed)
        self.bottleneck_score = bottleneck_score if bottleneck_score is not None else torch.zeros(n)
        self.gain_tuning_mask = gain_tuning_mask if gain_tuning_mask is not None else torch.zeros(n, dtype=torch.bool)

    @classmethod
    def build(cls, scaffold_id, ref_pdb, apo_pdb, holo_pdb, distal_idx, pocket_idx=None, chains=None,
              metadata=None):
        mapping = ResidueMapping.from_structure(scaffold_id, ref_pdb, chains)
        resp, conf = extract(apo_pdb, holo_pdb, mapping, distal_idx)
        dm = torch.zeros(mapping.n_res, dtype=torch.bool)
        dm[torch.as_tensor(list(distal_idx), dtype=torch.long)] = True
        return cls(mapping, resp, conf, dm, metadata, pocket_idx=pocket_idx)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.response, os.path.join(out_dir, "native_response.pt"))
        torch.save(self.distal_mask, os.path.join(out_dir, "distal_mask.pt"))
        torch.save(self.confidence, os.path.join(out_dir, "native_confidence.pt"))
        torch.save(self.native_direction, os.path.join(out_dir, "native_direction.pt"))
        torch.save(self.bottleneck_score, os.path.join(out_dir, "bottleneck_score.pt"))
        torch.save(self.gain_tuning_mask, os.path.join(out_dir, "gain_tuning_mask.pt"))
        self.mapping.save(os.path.join(out_dir, "residue_mapping.json"))
        json.dump({"scaffold_id": self.mapping.scaffold_id, "channels": CHANNELS,
                   "native_gain": self.native_gain, **self.metadata},
                  open(os.path.join(out_dir, "metadata.json"), "w"), indent=2)

    @classmethod
    def load(cls, out_dir):
        mapping = ResidueMapping.load(os.path.join(out_dir, "residue_mapping.json"))
        meta = json.load(open(os.path.join(out_dir, "metadata.json")))

        def opt(name):
            p = os.path.join(out_dir, name)
            return torch.load(p) if os.path.exists(p) else None
        return cls(mapping, torch.load(os.path.join(out_dir, "native_response.pt")),
                   torch.load(os.path.join(out_dir, "native_confidence.pt")),
                   torch.load(os.path.join(out_dir, "distal_mask.pt")), meta,
                   native_direction=opt("native_direction.pt"), native_gain=meta.get("native_gain"),
                   bottleneck_score=opt("bottleneck_score.pt"), gain_tuning_mask=opt("gain_tuning_mask.pt"))
