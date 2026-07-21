"""Joint functional heads. Predict the two experimental states directly and classify the outcome -
no hand-multiplied gate.

  R_apo : baseline repression competence with no ligand (from the DBD embedding of the protein alone)
  R_lig : transcriptional output with the target ligand (from the ligand-conditioned, propagated DBD)
  M_switch = R_lig - R_apo                                          (sign fixed by the reporter)
  P(y | variant, ligand) over {functional_sensor, constitutive_ON, constitutive_OFF, non_responsive}

The final ranking score is P(functional_sensor) directly - binding/folding/path are auxiliary, not
factors multiplied in. R_apo and R_lig share one repression-competence readout so their difference
isolates the ligand effect, while each is still trained on its own (so the readout's bias is real,
not cancelling).
"""
import torch
import torch.nn as nn

CLASSES = ("functional_sensor", "constitutive_ON", "constitutive_OFF", "non_responsive")


class FunctionalHeads(nn.Module):
    def __init__(self, h_dim, phys_dim, ligand_dim, hidden=128):
        super().__init__()
        self.readout = nn.Sequential(               # shared repression-competence readout g()
            nn.Linear(h_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2 * h_dim + 2 + ligand_dim + phys_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, len(CLASSES)))

    def forward(self, apo_dbd, lig_dbd, ligand_vec, physics):
        r_apo = self.readout(apo_dbd).squeeze(-1)
        r_lig = self.readout(lig_dbd).squeeze(-1)
        m_switch = r_lig - r_apo
        cls_in = torch.cat([apo_dbd, lig_dbd, r_apo.reshape(1), r_lig.reshape(1), ligand_vec, physics])
        class_logits = self.classifier(cls_in)
        return {"R_apo": r_apo, "R_lig": r_lig, "M_switch": m_switch, "class_logits": class_logits}
