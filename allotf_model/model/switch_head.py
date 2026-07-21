"""Head 3 - Switch: the core. Is the ligand-bound state less DNA-competent than the apo state?

S_DNA(state) is read from that state's DBD-region embedding AND its offset from the DNA-bound
reference (the dna state's DBD embedding), plus the state's DNA-geometry features. So the DNA-bound
state's learned representation directly drives the score (dS_final/dG_dna != 0), and the scorer is
DBD-focused rather than reading the whole protein. The switch margin is

    M_switch = topology_sign * (S_DNA(apo) - S_DNA(lig))         (+1 release, -1 enhanced binding)

P_functional is a MONOTONE function of M_switch (softplus-positive slope), so the switch BCE trains
M_switch's sign directly and the final gate uses the switch signal exactly once. Disorder-mediated
switching (LacI: the induced DBD melts) enters through the DBD embedding collapsing away from the
DNA-competent reference and through the DNA-compatible-fraction geometry feature.
"""
import torch
import torch.nn as nn


class SwitchHead(nn.Module):
    def __init__(self, dbd_dim, geom_dim, hidden=128):
        super().__init__()
        # S_DNA(state) from [dbd_embed(state), dbd_embed(state) - dbd_embed(dna_ref), geom_state]
        in_dim = 2 * dbd_dim + geom_dim
        # final bias omitted on purpose: S_DNA(apo) and S_DNA(lig) share it, so it cancels in the
        # margin M_switch = S_DNA(apo) - S_DNA(lig) and would be an untrainable dead parameter
        self.s_dna = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1, bias=False))
        # P_functional = sigmoid(softplus(slope) * m_switch + bias): monotone in the margin
        self.slope = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _s_dna(self, dbd_state, dbd_ref, geom_state):
        return self.s_dna(torch.cat([dbd_state, dbd_state - dbd_ref, geom_state])).squeeze(-1)

    def forward(self, dbd_apo, dbd_lig, dbd_dna, geom_apo, geom_lig, topology_sign):
        s_apo = self._s_dna(dbd_apo, dbd_dna, geom_apo)
        s_lig = self._s_dna(dbd_lig, dbd_dna, geom_lig)
        m_switch = topology_sign * (s_apo - s_lig)
        func_logit = nn.functional.softplus(self.slope) * m_switch + self.bias
        return m_switch, func_logit
