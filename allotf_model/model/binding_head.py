"""Head 1 - Binding: can the target ligand bind in a reliable pose? Excludes non-binders.

Sees the ligand-state embedding, the pocket-region change, and the PhysPocket proxies (pocket energy
terms, pose stability, specificity). Outputs P_bind in [0,1]. It is a GATE, not a ranker: its job is
to zero out candidates that cannot hold the ligand at all.
"""
import torch.nn as nn


class BindingHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logit; sigmoid applied in the model
