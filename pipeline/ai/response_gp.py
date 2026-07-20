"""Structured Gaussian-process surrogate: (candidate, mechanism, dose, time) -> F(c,t).

SCAFFOLD-SPECIFIC, not cross-scaffold. One project holds one scaffold and the GP is fitted on that
project's own plates. Cross-scaffold mechanism transfer is a result to be earned once several
scaffolds have run, not a property of this code.

THE STATISTICAL UNIT IS THE CANDIDATE, NOT THE WELL. Eight sequences at 5 doses x 7 timepoints
produce 280 rows but still only n_sequence = 8. A plain RBF over one concatenated
(mechanism, mutation, dose, time) vector treats 35 repeated reads of the same protein as 35
independent pieces of evidence about sequence, and the posterior then looks far more confident
about sequence effects than eight designs can justify. The kernel is therefore a PRODUCT over
separable factors plus an explicit candidate random effect:

    K = K_candidate(mechanism, mutation) * K_dose(log c) * K_time(t)
        + K_offset(candidate identity)          <- candidate-level random effect
        + White(replicate noise)

K_offset gives every well of one candidate a shared, correlated offset - which is what a biological
replicate actually is - instead of letting the mean absorb it. Model selection uses
candidate-grouped splits (see grouped_cv_score): a random split would put some wells of a candidate
in train and others in test and report a skill the model does not have on an unseen sequence.

The surface, not a phenotype, is what the GP predicts (predict_surface); phenotypes are read off it
by fluorescence.phenotypes, exactly as they are read off a real plate. For selection the joint
posterior is sampled (sample_surfaces) so acquisition/bandit can do multi-objective Thompson
sampling over whole predicted functional surfaces rather than over point estimates.
"""
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, ConstantKernel, Kernel,
                                              Hyperparameter)

from . import fluorescence as fl

# mechanism features, all normalised relative to the scaffold's own WT upstream (Δx)
MECH_FEATURES = ["d_target_binding", "ddG_coupling", "d_apo_dna_affinity", "dna_release",
                 "template_similarity", "decoy_gap", "d_hbond", "d_saltbridge", "strain"]
# mutation features by FUNCTIONAL position (not absolute residue number) + physicochemical deltas
MUT_FEATURES = ["n_recognition", "n_transduction", "n_hinge", "n_dimer", "n_path",
                "d_charge", "d_volume", "d_hydrophobicity", "d_aromaticity", "n_mut"]

C_FLOOR = 1e-4          # log10 of the zero-dose well maps here rather than to -inf


class _OnDims(Kernel):
    """Apply a kernel to a SUBSET of the columns.

    This is what makes K_candidate x K_dose x K_time expressible: without it every factor would see
    every column and the product would collapse back into one undifferentiated RBF.
    """

    def __init__(self, kernel, dims):
        # stored VERBATIM: sklearn.clone re-instantiates from get_params and then checks the
        # attributes came back identical, so converting dims here (list(dims)) makes the clone
        # fail with "constructor either does not set or modifies parameter dims"
        self.kernel = kernel
        self.dims = dims

    def _cols(self):
        return list(self.dims)

    def get_params(self, deep=True):
        p = {"kernel": self.kernel, "dims": self.dims}
        if deep:
            for k, v in self.kernel.get_params(deep=True).items():
                p["kernel__" + k] = v
        return p

    @property
    def hyperparameters(self):
        out = []
        for h in self.kernel.hyperparameters:
            out.append(Hyperparameter("kernel__" + h.name, h.value_type, h.bounds,
                                      h.n_elements, h.fixed))
        return out

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        cols = self._cols()
        Xs = np.asarray(X)[:, cols]
        Ys = None if Y is None else np.asarray(Y)[:, cols]
        return self.kernel(Xs, Ys, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(np.asarray(X)[:, self._cols()])

    def is_stationary(self):
        return self.kernel.is_stationary()


class ResponseGP:
    def __init__(self, scaffolds):
        # kept for provenance only: this model is fitted per scaffold, it does not pool them
        self.scaffolds = list(dict.fromkeys(scaffolds))
        self.gp = None
        self._mean = None
        self._std = None
        self._cid_index = {}
        self._n_candidates = 0
        self._groups = []

    def featurize(self, mech, mut, conc, t, scaffold, cid=None):
        """One (candidate, dose, time) -> feature vector.

        Column layout is fixed, because the kernel factors address columns by index:
            [0 : n_mech+n_mut)   candidate chemistry/mechanism  -> K_candidate
            [n .. n+1)           log dose                       -> K_dose
            [n+1 .. n+2)         time                           -> K_time
            [n+2 .. )            candidate identity code        -> K_offset (random effect)
        The zero dose is floored, not dropped: the basal well is a data point.
        """
        log_c = np.log10(max(float(conc), C_FLOOR))
        v = [float(mech.get(k, 0.0)) for k in MECH_FEATURES]
        v += [float(mut.get(k, 0.0)) for k in MUT_FEATURES]
        v += [log_c, float(t)]
        v += [float(self._cid_code(cid))]
        return np.asarray(v, float)

    def _cid_code(self, cid):
        """Stable integer per candidate. Only equality matters to K_offset, not the value."""
        if cid is None:
            return -1.0
        if cid not in self._cid_index:
            self._cid_index[cid] = len(self._cid_index)
        return float(self._cid_index[cid])

    # column blocks, resolved once from the feature lists
    @property
    def _dims(self):
        n = len(MECH_FEATURES) + len(MUT_FEATURES)
        return {"cand": list(range(n)), "dose": [n], "time": [n + 1], "cid": [n + 2]}

    def _design_matrix(self, observations):
        X, y, groups = [], [], []
        for o in observations:
            cid = o.get("cid")
            X.append(self.featurize(o["mech"], o["mut"], o["conc"], o["time"],
                                    o.get("scaffold"), cid))
            y.append(float(o["F"]))
            groups.append(cid)
        return np.asarray(X), np.asarray(y), groups

    def _build_kernel(self):
        """K_candidate x K_dose x K_time + K_offset(candidate) + White.

        A product, not a sum over one concatenated vector: dose response and time course are
        separable effects, and multiplying them keeps the model from spending sequence-level
        flexibility explaining a dose curve. K_offset then absorbs the per-candidate shift that all
        35 of its wells share, so those wells stop being counted as independent evidence about
        sequence.
        """
        d = self._dims
        k_cand = ConstantKernel(1.0, (1e-2, 1e3)) * _OnDims(
            RBF(length_scale=np.ones(len(d["cand"])), length_scale_bounds=(1e-2, 1e3)), d["cand"])
        k_dose = _OnDims(RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), d["dose"])
        k_time = _OnDims(RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), d["time"])
        k_offset = ConstantKernel(1.0, (1e-3, 1e3)) * _OnDims(
            RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1.0)), d["cid"])
        return k_cand * k_dose * k_time + k_offset + WhiteKernel(1.0, (1e-6, 1e2))

    def fit(self, observations):
        """observations: list of dict(cid, mech, mut, conc, time, scaffold, F).

        `cid` (the candidate id) is what makes the statistical unit the SEQUENCE rather than the
        well; without it every repeated read counts as a fresh observation of the design.
        """
        X, y, groups = self._design_matrix(observations)
        if len(X) < 3:
            raise ValueError("need >=3 observations to fit the GP, got %d" % len(X))
        n_cand = len({g for g in groups if g is not None})
        self._n_candidates = n_cand or 1
        self._groups = groups
        # standardise everything EXCEPT the identity column: scaling an arbitrary code would make
        # "candidate 0 vs 1" a distance the kernel reads as chemistry
        self._mean = X.mean(0)
        self._std = X.std(0) + 1e-9
        cid_col = self._dims["cid"][0]
        self._mean[cid_col], self._std[cid_col] = 0.0, 1.0
        Xs = (X - self._mean) / self._std
        self.gp = GaussianProcessRegressor(kernel=self._build_kernel(), normalize_y=True,
                                           n_restarts_optimizer=2, alpha=1e-8)
        self.gp.fit(Xs, y)
        return self

    def _grid_X(self, mech, mut, conc, time, scaffold, cid=None):
        return np.asarray([self.featurize(mech, mut, c, t, scaffold, cid)
                           for c in conc for t in time])

    def _scale(self, X):
        return (X - self._mean) / self._std

    def predict_surface(self, mech, mut, conc, time, scaffold, cid=None):
        """-> (F_mean, F_std) as (len(conc), len(time)) arrays."""
        if self.gp is None:
            raise RuntimeError("GP not fitted")
        conc, time = np.asarray(conc, float), np.asarray(time, float)
        m, s = self.gp.predict(self._scale(self._grid_X(mech, mut, conc, time, scaffold, cid)),
                               return_std=True)
        shape = (len(conc), len(time))
        return m.reshape(shape), s.reshape(shape)

    def sample_surfaces(self, mech, mut, conc, time, scaffold, n_samples=64, seed=0, cid=None):
        """Draw whole surfaces from the JOINT posterior — points on one surface stay correlated, so
        a sampled phenotype is self-consistent. This is what makes Thompson selection over
        phenotypes valid rather than sampling each (c,t) independently.
        -> array (n_samples, len(conc), len(time))."""
        if self.gp is None:
            raise RuntimeError("GP not fitted")
        conc, time = np.asarray(conc, float), np.asarray(time, float)
        Y = self.gp.sample_y(self._scale(self._grid_X(mech, mut, conc, time, scaffold, cid)),
                             n_samples=n_samples, random_state=seed)
        return np.clip(Y.T.reshape(n_samples, len(conc), len(time)), 0.0, None)

    def predict_phenotypes(self, mech, mut, conc, time, scaffold, cid=None):
        """Mean predicted surface -> phenotypes, plus the mean posterior std as a confidence proxy."""
        F, S = self.predict_surface(mech, mut, conc, time, scaffold, cid)
        ph = fl.phenotypes(conc, time, F)
        ph["pred_uncertainty"] = float(S.mean())
        return ph

    def grouped_cv_score(self, observations, n_folds=None):
        """Leave-one-CANDIDATE-out score. The only honest estimate of skill on an unseen sequence.

        A random split puts some wells of a candidate in train and the rest in test; the model then
        predicts a sequence it has already seen 30 times and reports a skill it does not have. With
        eight designs, leaving out one candidate at a time is also the largest test set the data
        support.
        -> dict(r2, rmse, n_folds, per_fold). Returns None when there are <2 candidates to split.
        """
        X, y, groups = self._design_matrix(observations)
        uniq = [g for g in dict.fromkeys(groups) if g is not None]
        if len(uniq) < 2:
            return None
        folds = uniq if n_folds is None else uniq[:n_folds]
        per_fold, preds, truth = [], [], []
        for held in folds:
            tr = [i for i, g in enumerate(groups) if g != held]
            te = [i for i, g in enumerate(groups) if g == held]
            if not tr or not te:
                continue
            mean, std = X[tr].mean(0), X[tr].std(0) + 1e-9
            cid_col = self._dims["cid"][0]
            mean[cid_col], std[cid_col] = 0.0, 1.0
            gp = GaussianProcessRegressor(kernel=self._build_kernel(), normalize_y=True,
                                          n_restarts_optimizer=0, alpha=1e-8)
            gp.fit((X[tr] - mean) / std, y[tr])
            p = gp.predict((X[te] - mean) / std)
            rmse = float(np.sqrt(np.mean((p - y[te]) ** 2)))
            per_fold.append({"held_out": held, "n_test": len(te), "rmse": rmse})
            preds.extend(p.tolist())
            truth.extend(y[te].tolist())
        if not per_fold:
            return None
        preds, truth = np.asarray(preds), np.asarray(truth)
        ss_res = float(np.sum((preds - truth) ** 2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2)) or 1e-12
        return {"r2": 1.0 - ss_res / ss_tot,
                "rmse": float(np.sqrt(np.mean((preds - truth) ** 2))),
                "n_folds": len(per_fold), "n_candidates": len(uniq), "per_fold": per_fold}

    def sample_objectives(self, mech, mut, conc, time, scaffold, n_samples=64, seed=0, cid=None):
        """Posterior samples -> list of objective vectors (None-dropped). Feeds Thompson selection."""
        surfaces = self.sample_surfaces(mech, mut, conc, time, scaffold, n_samples, seed, cid=cid)
        out = []
        for F in surfaces:
            obj = fl.objectives(fl.phenotypes(conc, time, F))
            if obj is not None:
                out.append(obj)
        return out
