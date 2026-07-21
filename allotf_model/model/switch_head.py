"""Head 3 - Switch: the core. Is the ligand-bound state less compatible with DNA than the apo state?

Learns S_DNA for each state from DBD spacing/angle, recognition-helix/operator geometry, DNA-contact
exposure, DNA interface contacts, DBD order/disorder, hinge flexibility, DNA-compatible ensemble
fraction and ligand-induced communication change - NOT from hand weights and NOT from a Rosetta
delta used as a label. The switch margin is M_switch = S_DNA(apo) - S_DNA(lig); it also emits a
functional probability. Disorder-mediated switching (LacI: the induced DBD melts) is handled through
the order/DNA-compatible-fraction features, so the same head covers rigid and disorder mechanisms.
"""
import torch
import torch.nn as nn


class SwitchHead(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        # a SHARED per-state DNA-compatibility scorer: S_DNA(state) from that state's features
        self.s_dna = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        # functional probability from the margin plus the contrast context
        self.func = nn.Sequential(
            nn.Linear(1 + state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, feat_apo, feat_lig, topology_sign):
        """feat_apo / feat_lig: per-state DNA-relevant feature vectors. topology_sign: +1 release,
        -1 enhanced. -> (M_switch, functional_logit)."""
        s_apo = self.s_dna(feat_apo).squeeze(-1)
        s_lig = self.s_dna(feat_lig).squeeze(-1)
        # release (+1): apo binds DNA, ligand releases -> S_DNA(apo) > S_DNA(lig) -> positive margin.
        # enhanced (-1): sign flips so a good ligand-enhanced binder also scores positive.
        m_switch = topology_sign * (s_apo - s_lig)
        func_logit = self.func(torch.cat([m_switch.reshape(1), feat_lig])).squeeze(-1)
        return m_switch, func_logit
