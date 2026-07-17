"""Hierarchical Gaussian-process surrogate: (mechanism, mutation, dose, time, scaffold) -> F(c,t).

Learns the raw fluorescence surface with calibrated uncertainty. It is the whole AI: no deep net,
no per-target retraining. Mechanism weights are shared across scaffolds (one anisotropic
length-scale per mechanism feature, fit on all data at once); a scaffold random effect is carried
by scaffold-indicator dimensions, so a scaffold contributes an offset without redefining the
mechanism map — TtgR is one task among several, not the definition of the response.

The surface, not a phenotype, is what the GP predicts (predict_surface); phenotypes are read off it
by fluorescence.phenotypes, exactly as they are read off a real plate. For selection the joint
posterior is sampled (sample_surfaces) so acquisition/bandit can do multi-objective Thompson
sampling over whole predicted functional surfaces rather than over point estimates.
"""
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from . import fluorescence as fl

# mechanism features, all normalised relative to the scaffold's own WT upstream (Δx)
MECH_FEATURES = ["d_target_binding", "ddG_coupling", "d_apo_dna_affinity", "dna_release",
                 "template_similarity", "decoy_gap", "d_hbond", "d_saltbridge", "strain"]
# mutation features by FUNCTIONAL position (not absolute residue number) + physicochemical deltas
MUT_FEATURES = ["n_recognition", "n_transduction", "n_hinge", "n_dimer", "n_path",
                "d_charge", "d_volume", "d_hydrophobicity", "d_aromaticity", "n_mut"]

C_FLOOR = 1e-4          # log10 of the zero-dose well maps here rather than to -inf


class ResponseGP:
    def __init__(self, scaffolds):
        self.scaffolds = list(dict.fromkeys(scaffolds))    # stable unique order
        self.gp = None
        self._mean = None
        self._std = None

    def featurize(self, mech, mut, conc, t, scaffold):
        """One (candidate, dose, time) -> feature vector. conc is a real concentration; the zero
        dose is floored, not dropped, because the basal well is a data point."""
        log_c = np.log10(max(float(conc), C_FLOOR))
        v = [float(mech.get(k, 0.0)) for k in MECH_FEATURES]
        v += [float(mut.get(k, 0.0)) for k in MUT_FEATURES]
        v += [log_c, float(t)]
        v += [1.0 if scaffold == s else 0.0 for s in self.scaffolds]
        return np.asarray(v, float)

    def _design_matrix(self, observations):
        X, y = [], []
        for o in observations:
            X.append(self.featurize(o["mech"], o["mut"], o["conc"], o["time"], o["scaffold"]))
            y.append(float(o["F"]))
        return np.asarray(X), np.asarray(y)

    def fit(self, observations):
        """observations: list of dict(mech, mut, conc, time, scaffold, F). F is background-subtracted
        fluorescence at that (candidate, dose, time)."""
        X, y = self._design_matrix(observations)
        if len(X) < 3:
            raise ValueError("need >=3 observations to fit the GP, got %d" % len(X))
        self._mean = X.mean(0)
        self._std = X.std(0) + 1e-9
        Xs = (X - self._mean) / self._std
        d = Xs.shape[1]
        kernel = (ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(d),
                  length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1.0, (1e-6, 1e2)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=2, alpha=1e-8)
        self.gp.fit(Xs, y)
        return self

    def _grid_X(self, mech, mut, conc, time, scaffold):
        return np.asarray([self.featurize(mech, mut, c, t, scaffold)
                           for c in conc for t in time])

    def _scale(self, X):
        return (X - self._mean) / self._std

    def predict_surface(self, mech, mut, conc, time, scaffold):
        """-> (F_mean, F_std) as (len(conc), len(time)) arrays."""
        if self.gp is None:
            raise RuntimeError("GP not fitted")
        conc, time = np.asarray(conc, float), np.asarray(time, float)
        m, s = self.gp.predict(self._scale(self._grid_X(mech, mut, conc, time, scaffold)),
                               return_std=True)
        shape = (len(conc), len(time))
        return m.reshape(shape), s.reshape(shape)

    def sample_surfaces(self, mech, mut, conc, time, scaffold, n_samples=64, seed=0):
        """Draw whole surfaces from the JOINT posterior — points on one surface stay correlated, so
        a sampled phenotype is self-consistent. This is what makes Thompson selection over
        phenotypes valid rather than sampling each (c,t) independently.
        -> array (n_samples, len(conc), len(time))."""
        if self.gp is None:
            raise RuntimeError("GP not fitted")
        conc, time = np.asarray(conc, float), np.asarray(time, float)
        Y = self.gp.sample_y(self._scale(self._grid_X(mech, mut, conc, time, scaffold)),
                             n_samples=n_samples, random_state=seed)
        return np.clip(Y.T.reshape(n_samples, len(conc), len(time)), 0.0, None)

    def predict_phenotypes(self, mech, mut, conc, time, scaffold):
        """Mean predicted surface -> phenotypes, plus the mean posterior std as a confidence proxy."""
        F, S = self.predict_surface(mech, mut, conc, time, scaffold)
        ph = fl.phenotypes(conc, time, F)
        ph["pred_uncertainty"] = float(S.mean())
        return ph

    def sample_objectives(self, mech, mut, conc, time, scaffold, n_samples=64, seed=0):
        """Posterior samples -> list of objective vectors (None-dropped). Feeds Thompson selection."""
        surfaces = self.sample_surfaces(mech, mut, conc, time, scaffold, n_samples, seed)
        out = []
        for F in surfaces:
            obj = fl.objectives(fl.phenotypes(conc, time, F))
            if obj is not None:
                out.append(obj)
        return out
