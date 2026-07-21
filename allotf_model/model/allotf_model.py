"""The full model. Shared encoder over apo/lig/dna -> explicit state contrast -> three heads ->
MULTIPLICATIVE gate.

The gate is the point: S_final = P_bind * P_path * P_functional * sigma(M_switch). A high switch score
cannot rescue a candidate that does not bind; a good binder cannot rescue a broken path. Only a
candidate that binds AND keeps the path AND actually flips scores high - which is exactly the
functional-sensor vs binder-only distinction the model exists to make.
"""
import torch
import torch.nn as nn

from .equivariant_encoder import MultiStateEncoder
from .state_contrast import StateContrast
from .binding_head import BindingHead
from .path_head import PathHead
from .switch_head import SwitchHead


class AlloTFModel(nn.Module):
    def __init__(self, node_in, edge_in, h_dim=128, layers=4,
                 phys_bind_dim=0, path_dim=0, geom_dim=0):
        super().__init__()
        self.encoder = MultiStateEncoder(node_in, edge_in, h_dim, layers)
        self.contrast = StateContrast(h_dim)
        self.binding = BindingHead(2 * h_dim + phys_bind_dim)
        self.path = PathHead(3 * h_dim + 2 + path_dim)
        self.switch = SwitchHead(h_dim + geom_dim)
        self.h_dim = h_dim

    def _res(self, h_full, g):
        return h_full[: g.n_res]                       # residue nodes are rows [0:n_res]

    def forward(self, mg):
        # shared encoder on each present state; keep only residue embeddings for the contrast
        h = {s: self._res(self.encoder(g), g) for s, g in mg.states.items() if s in ("apo", "lig", "dna")}
        feats, _ = self.contrast(h, mg.region_masks, mg.resolvent)

        phys_bind = mg.physics.get("bind", feats["lig"].new_zeros(0))
        path_vec = mg.physics.get("path", feats["lig"].new_zeros(0))
        geom_apo = mg.geom.get("apo", feats["apo"].new_zeros(0))
        geom_lig = mg.geom.get("lig", feats["lig"].new_zeros(0))

        bind_in = torch.cat([feats["lig"], feats["d_lig_pocket"], phys_bind])
        path_in = torch.cat([feats["d_lig"], feats["d_lig_hinge"], feats["d_lig_dbd"],
                             feats["resolvent"], path_vec])
        p_bind_logit = self.binding(bind_in)
        p_path_logit = self.path(path_in)
        m_switch, p_func_logit = self.switch(
            torch.cat([feats["apo"], geom_apo]), torch.cat([feats["lig"], geom_lig]),
            mg.topology_sign)

        p_bind = torch.sigmoid(p_bind_logit)
        p_path = torch.sigmoid(p_path_logit)
        p_func = torch.sigmoid(p_func_logit)
        s_final = p_bind * p_path * p_func * torch.sigmoid(m_switch)

        return {"P_bind": p_bind, "P_path": p_path, "P_functional": p_func,
                "M_switch": m_switch, "S_final": s_final,
                "pooled": {"apo": feats["apo"], "lig": feats["lig"], "dna": feats["dna"]},
                "logits": {"bind": p_bind_logit, "path": p_path_logit, "func": p_func_logit}}
