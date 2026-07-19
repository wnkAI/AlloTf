"""Thompson contextual bandit over the GP surrogate.

This is the reinforcement-learning half: it observes the fluorescence from a batch, updates the GP,
and picks the next batch by Thompson sampling the posterior. It is model-based on purpose — the GP
holds what the plates taught it, and selection never runs outside that fitted model, so it cannot
chase a surrogate hole the way a free-running deep policy can.
"""
from . import acquisition, fluorescence as fl


class ThompsonBandit:
    def __init__(self, gp, conc, time, n_samples=64, seed=0):
        self.gp = gp
        self.conc = list(conc)
        self.time = list(time)
        self.n_samples = n_samples
        self.seed = seed

    def observe(self, observations):
        """Refit the GP on all fluorescence seen so far (list of dict(mech,mut,conc,time,scaffold,F))."""
        self.gp.fit(observations)
        return self

    def _candidate_samples(self, candidates):
        cand_samples, meta = {}, {}
        for i, c in enumerate(candidates):
            surfaces = self.gp.sample_surfaces(c["mech"], c["mut"], self.conc, self.time,
                                               c["scaffold"], self.n_samples, self.seed + i,
                                               cid=c["candidate_id"])
            cand_samples[c["candidate_id"]] = [
                fl.objectives(fl.phenotypes(self.conc, self.time, F)) for F in surfaces]
            ph = self.gp.predict_phenotypes(c["mech"], c["mut"], self.conc, self.time,
                                            c["scaffold"], cid=c["candidate_id"])
            meta[c["candidate_id"]] = {"sequence": c.get("sequence", ""), "basal": ph["basal"],
                                       "pred": ph}
        return cand_samples, meta

    def select(self, candidates, n, basal_max=None, min_seq_dist=2):
        """candidates: list of dict(candidate_id, sequence, mech, mut, scaffold).
        -> acquisition.select result, plus the per-candidate mean phenotypes for reporting."""
        if self.gp.gp is None:
            raise RuntimeError("bandit has no fitted GP; call observe() with the first plate first")
        cand_samples, meta = self._candidate_samples(candidates)
        out = acquisition.select(cand_samples, meta, n, basal_max=basal_max,
                                 min_seq_dist=min_seq_dist)
        out["phenotypes"] = {c: meta[c]["pred"] for c in cand_samples}
        return out
