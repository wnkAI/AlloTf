"""DNA-release head. Predicts operator affinity in the no-ligand and ligand-bound states, and BOTH the
OFF background and the release margin - not just fold change. Maximising fold change alone selects
false positives that have lost apo DNA binding (no real output); a good sensor must grip the operator
when apo AND release it on ligand.

    logKd_apo    = g(apo DBD readout)              low  -> apo mutant still grips the operator
    logKd_target = g(ligand-conditioned readout)
    ddG_coupling = logKd_target - logKd_apo        > 0  -> ligand LOWERS DNA affinity (releases)
    S_DNA_apo    = sigma(-logKd_apo)               apo DNA-binding competence (the OFF state)
    S_DNA_ligand = sigma(-logKd_target)            ligand-state DNA binding
    release_margin = S_DNA_apo - S_DNA_ligand      dynamic range (needs S_DNA_apo high AND this high)
"""
import torch
import torch.nn as nn


class DNAReleaseHead(nn.Module):
    def __init__(self, h_dim, hidden=128):
        super().__init__()
        self.affinity = nn.Sequential(
            nn.Linear(h_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, apo_dbd, lig_dbd):
        logkd_apo = self.affinity(apo_dbd).squeeze(-1)
        logkd_target = self.affinity(lig_dbd).squeeze(-1)
        s_apo = torch.sigmoid(-logkd_apo)
        s_lig = torch.sigmoid(-logkd_target)
        return {"logKd_apo": logkd_apo, "logKd_target": logkd_target,
                "ddG_coupling": logkd_target - logkd_apo,
                "S_DNA_apo": s_apo, "S_DNA_ligand": s_lig, "release_margin": s_apo - s_lig}
