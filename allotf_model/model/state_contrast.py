"""Make the state change EXPLICIT rather than letting the heads guess it from three embeddings.

Given the shared encoder's per-residue embeddings for apo/lig/dna, this builds h_lig-h_apo and
h_dna-h_apo, pools them globally and per functional region (pocket, hinge, DBD, DNA interface), keeps
per-STATE region pools (so the switch head can read each state's DBD embedding and compare it to the
DNA-bound reference), and folds in the physics resolvent channel. The result is what the heads see,
so the model is centred on "what did ligand binding change, and did it reach the DBD".
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
        # per-region gate so the model can learn which region's change matters
        self.region_gate = nn.Parameter(torch.ones(len(REGIONS)))

    def forward(self, h, region_masks, resolvent):
        """h: {'apo','lig','dna'} -> (n_res, h_dim). -> feature dict (global, delta, per-state region)."""
        h_apo, h_lig, h_dna = h["apo"], h["lig"], h["dna"]
        d_lig = h_lig - h_apo
        d_dna = h_dna - h_apo

        feats = {
            "apo": h_apo.mean(0), "lig": h_lig.mean(0), "dna": h_dna.mean(0),
            "d_lig": d_lig.mean(0), "d_dna": d_dna.mean(0),
        }
        for k, region in enumerate(REGIONS):
            m = region_masks.get(region)
            feats["d_lig_%s" % region] = _masked_mean(d_lig, m) * self.region_gate[k]
            feats["d_dna_%s" % region] = _masked_mean(d_dna, m)
            # per-STATE region pools (the switch head needs each state's DBD embedding)
            feats["apo_%s" % region] = _masked_mean(h_apo, m)
            feats["lig_%s" % region] = _masked_mean(h_lig, m)
            feats["dna_%s" % region] = _masked_mean(h_dna, m)

        res = resolvent.to(h_apo.dtype)
        rmax = res.abs().max() if res.numel() else res.new_zeros(())
        feats["resolvent"] = torch.stack([res.mean(), rmax])
        return feats
