"""The full model. Shared encoder over apo/lig/dna -> explicit state contrast -> three heads ->
MULTIPLICATIVE gate.

The gate is the point: S_final = P_bind * P_path * P_functional. A high switch score cannot rescue a
candidate that does not bind; a good binder cannot rescue a broken path. P_functional is a monotone
function of the switch margin M_switch (so the switch enters the gate exactly once), and M_switch is
driven by the DNA-bound state's DBD embedding - so all three states affect the score.
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
        self.path = PathHead(4 * h_dim + 2 + path_dim)
        self.switch = SwitchHead(dbd_dim=h_dim, geom_dim=geom_dim)
        self.h_dim = h_dim
        self.phys_bind_dim, self.path_dim, self.geom_dim = phys_bind_dim, path_dim, geom_dim

    def _res(self, h_full, g):
        return h_full[: g.n_res]                       # residue nodes are rows [0:n_res]

    def _opt(self, d, key, dim, ref):
        """Optional feature vector: the CONFIGURED dim as zeros when absent (never a zero-length
        vector), and a size check when present so a wrong-dim feature fails loudly, not silently."""
        v = d.get(key)
        if v is None:
            return ref.new_zeros(dim)
        if v.numel() != dim:
            raise ValueError("feature '%s' has %d dims, model configured for %d" % (key, v.numel(), dim))
        return v.to(ref.dtype)

    def forward(self, mg):
        h = {s: self._res(self.encoder(g), g) for s, g in mg.states.items() if s in ("apo", "lig", "dna")}
        feats = self.contrast(h, mg.region_masks, mg.resolvent)
        ref = feats["lig"]

        phys_bind = self._opt(mg.physics, "bind", self.phys_bind_dim, ref)
        path_vec = self._opt(mg.physics, "path", self.path_dim, ref)
        geom_apo = self._opt(mg.geom, "apo", self.geom_dim, ref)
        geom_lig = self._opt(mg.geom, "lig", self.geom_dim, ref)

        bind_in = torch.cat([feats["lig"], feats["d_lig_pocket"], phys_bind])
        path_in = torch.cat([feats["d_lig"], feats["d_lig_hinge"], feats["d_lig_dbd"],
                             feats["d_lig_dna_interface"], feats["resolvent"], path_vec])
        p_bind_logit = self.binding(bind_in)
        p_path_logit = self.path(path_in)
        m_switch, p_func_logit = self.switch(
            feats["apo_dbd"], feats["lig_dbd"], feats["dna_dbd"], geom_apo, geom_lig,
            mg.topology_sign)

        p_bind = torch.sigmoid(p_bind_logit)
        p_path = torch.sigmoid(p_path_logit)
        p_func = torch.sigmoid(p_func_logit)              # monotone in M_switch; the ONLY switch gate
        s_final = p_bind * p_path * p_func

        return {"P_bind": p_bind, "P_path": p_path, "P_functional": p_func,
                "M_switch": m_switch, "S_final": s_final, "topology_sign": mg.topology_sign,
                "pooled": {"apo": feats["apo"], "lig": feats["lig"], "dna": feats["dna"]},
                "logits": {"bind": p_bind_logit, "path": p_path_logit, "func": p_func_logit}}
