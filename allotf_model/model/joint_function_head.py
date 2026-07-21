"""Joint functional head. One head predicts the functional class directly instead of multiplying
three independent gates. The class that matters most is functional_sensor vs binder-only - only a
model that ranks sensors above binders has learned the response transfer, not just ligand binding.

S_design = P(functional_sensor) = P(apo repression AND target binding AND ligand-induced DNA release).
"""
import torch
import torch.nn as nn

CLASSES = ("functional_sensor", "binder_only", "constitutive_ON", "constitutive_OFF", "non_binder")


class JointFunctionHead(nn.Module):
    def __init__(self, h_dim, ligand_dim, phys_dim, aux_dim, hidden=128):
        super().__init__()
        # inputs: apo DBD, ligand DBD, distal response, ligand vec, [logKd_apo, logKd_target,
        #         ddG_coupling], physics, confidence-masked generator aux
        in_dim = 3 * h_dim + ligand_dim + 3 + phys_dim + aux_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, len(CLASSES)))
        self.sensor_idx = CLASSES.index("functional_sensor")

    def forward(self, apo_dbd, lig_dbd, dH_distal, ligand_vec, release, physics, aux):
        x = torch.cat([apo_dbd, lig_dbd, dH_distal, ligand_vec,
                       release["logKd_apo"].reshape(1), release["logKd_target"].reshape(1),
                       release["ddG_coupling"].reshape(1), physics, aux])
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return {"class_logits": logits, "class_probs": probs, "S_design": probs[self.sensor_idx]}
