"""State-contrast loss: force the encoder to attend to apo->ligand->DNA change, not to sequence or
template identity.

A small probe must recover which state a pooled embedding came from. If the three states were
encoded identically (the failure mode where the model ignores the ligand and reads off the sequence),
the probe cannot separate them and this loss stays high. It pushes the shared encoder to make the
state itself decodable.
"""
import torch
import torch.nn as nn


class StateContrastLoss(nn.Module):
    def __init__(self, h_dim, n_states=3):
        super().__init__()
        self.probe = nn.Linear(h_dim, n_states)
        self.order = {"apo": 0, "lig": 1, "dna": 2}

    def forward(self, pooled):
        """pooled: {'apo','lig','dna'} -> (h_dim,) each. -> CE that the probe recovers the state."""
        x = torch.stack([pooled[s] for s in ("apo", "lig", "dna")])
        y = x.new_tensor([0, 1, 2], dtype=torch.long)
        return nn.functional.cross_entropy(self.probe(x), y)
