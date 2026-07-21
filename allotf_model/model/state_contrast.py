"""Make the state change EXPLICIT rather than letting the heads guess it from three embeddings.

Given the shared encoder's per-residue embeddings for apo/lig/dna, this builds h_lig-h_apo and
h_dna-h_apo, pools them globally and over each functional region (pocket, hinge, DBD, DNA interface),
and folds in the physics resolvent channel. The result z is what the heads actually see, so the model
is centred on "what did ligand binding change, and did it reach the DBD", not on any single state.
"""
import torch
import torch.nn as nn

from ..graph.multistate_graph import REGIONS


def _masked_mean(x, mask):
    """Mean of rows selected by a boolean mask; zeros if the region is empty (a missing region is not
    a fabricated signal)."""
    if mask is None or mask.sum() == 0:
        return x.new_zeros(x.shape[1:])
    return x[mask].mean(0)


class StateContrast(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        # per-region gate so the model can learn which region's change matters per TF family
        self.region_gate = nn.Parameter(torch.ones(len(REGIONS)))

    def forward(self, h, region_masks, resolvent):
        """h: {'apo','lig','dna'} -> (n_res, h_dim) residue embeddings. -> feature dict + flat z."""
        h_apo, h_lig, h_dna = h["apo"], h["lig"], h["dna"]
        d_lig = h_lig - h_apo
        d_dna = h_dna - h_apo

        feats = {
            "apo": h_apo.mean(0), "lig": h_lig.mean(0), "dna": h_dna.mean(0),
            "d_lig": d_lig.mean(0), "d_dna": d_dna.mean(0),
        }
        region_vecs = []
        for k, region in enumerate(REGIONS):
            m = region_masks.get(region)
            rv = _masked_mean(d_lig, m) * self.region_gate[k]
            feats["d_lig_%s" % region] = rv
            region_vecs.append(rv)
            feats["d_dna_%s" % region] = _masked_mean(d_dna, m)

        # physics resolvent summary: how much pocket->DBD gain sits on the DBD-reaching residues
        res = resolvent.float()
        feats["resolvent"] = torch.stack([res.mean(), res.abs().max() if res.numel() else res.new_zeros(())])

        z = torch.cat([feats["apo"], feats["lig"], feats["dna"], feats["d_lig"], feats["d_dna"],
                       feats["resolvent"]])
        return feats, z
