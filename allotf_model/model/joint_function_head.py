"""Joint functional head. One head predicts the functional class directly instead of multiplying
three independent gates. The class that matters most is functional_sensor vs binder-only - only a
model that ranks sensors above binders has learned the response transfer, not just ligand binding.

S_design = P(functional_sensor) = P(target binding AND apo repression AND native-direction response
AND useful gain AND ligand-induced DNA release). The four teacher-match scalars (response_alignment,
allosteric_gain, off_path_response, apo_DNA_competence) enter EXPLICITLY so the probability sees the
frozen-teacher match directly, not only through the latent representation. Physics proxies enter
confidence-masked so a not-computed / low-confidence term contributes little and a real 0.0 is not
confused with "missing".
"""
import torch
import torch.nn as nn

CLASSES = ("functional_sensor", "binder_only", "constitutive_ON", "constitutive_OFF", "non_binder")


class JointFunctionHead(nn.Module):
    def __init__(self, h_dim, ligand_dim, phys_dim, hidden=128):
        super().__init__()
        # apo DBD, ligand DBD, distal response, ligand vec, [logKd_apo, logKd_target, ddG], physics,
        # + 4 teacher-match scalars (alignment, gain, off-path, apo competence)
        in_dim = 3 * h_dim + ligand_dim + 3 + phys_dim + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, len(CLASSES)))
        self.sensor_idx = CLASSES.index("functional_sensor")

    def forward(self, apo_dbd, lig_dbd, dH_distal, ligand_vec, release, physics_masked, teacher_match):
        x = torch.cat([apo_dbd, lig_dbd, dH_distal, ligand_vec,
                       release["logKd_apo"].reshape(1), release["logKd_target"].reshape(1),
                       release["ddG_coupling"].reshape(1), physics_masked, teacher_match])
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return {"class_logits": logits, "class_probs": probs, "S_design": probs[self.sensor_idx]}
