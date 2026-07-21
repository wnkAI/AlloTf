"""DNA-release head. Predicts the operator affinity of the DBD in the no-ligand and the
ligand-bound states and their coupling.

    logKd_apo    = g(apo DBD readout)      -> low means the apo mutant still grips the operator
    logKd_target = g(ligand-conditioned DBD readout)
    ddG_coupling = logKd_target - logKd_apo   (RT absorbed; positive = ligand LOWERS DNA affinity)

A functional sensor needs BOTH a low logKd_apo (apo represses) AND ddG_coupling > 0 (ligand releases).
One shared affinity readout scores both states so the coupling isolates the ligand effect; each logKd
is still trained on its own affinity data (Kd/EMSA/MST/BLI/SPR) so the readout's bias is real.
"""
import torch.nn as nn


class DNAReleaseHead(nn.Module):
    def __init__(self, h_dim, hidden=128):
        super().__init__()
        self.affinity = nn.Sequential(
            nn.Linear(h_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, apo_dbd, lig_dbd):
        logkd_apo = self.affinity(apo_dbd).squeeze(-1)
        logkd_target = self.affinity(lig_dbd).squeeze(-1)
        return {"logKd_apo": logkd_apo, "logKd_target": logkd_target,
                "ddG_coupling": logkd_target - logkd_apo}
