"""Coupling-gain head: predicts the allosteric transmission gain alpha and its uncertainty from the
propagated distal response, the native direction, the hinge/interface representation and the physics
confidence. Deliberately SMALL and interpretable - with few cross-scaffold examples a large head would
just memorise scaffold identity. The per-region contributions are computed analytically (not learned),
so the head predicts a scalar gain + uncertainty while attribution stays transparent.
"""
import torch
import torch.nn as nn


class CouplingGainHead(nn.Module):
    def __init__(self, h_dim, hidden=64):
        super().__init__()
        # pooled distal response + hinge repr + dbd repr + mean physics confidence
        self.net = nn.Sequential(nn.Linear(3 * h_dim + 1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))       # gain_mean, gain_logvar

    def forward(self, dH_distal, hinge_repr, dbd_repr, physics_conf, region_contributions):
        x = torch.cat([dH_distal, hinge_repr, dbd_repr, physics_conf.reshape(1)])
        out = self.net(x)
        return {"gain_mean": out[0], "gain_uncertainty": torch.exp(0.5 * out[1]),
                "gain_region_contributions": region_contributions}
