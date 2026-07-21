"""Head 2 - Path: do the pocket mutations keep the pocket->DBD communication network able to
propagate the state change? Learns which path features matter per TF family (not fixed weights).

Sees the pooled ligand-induced change over the hinge/DBD regions, the physics resolvent gain, and
the path proxies (path centrality, contact churn, calibrated path, template similarity, network
connectivity, state-dependent network change). Outputs P_path in [0,1].
"""
import torch.nn as nn


class PathHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logit
