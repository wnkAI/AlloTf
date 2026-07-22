"""Few-shot calibration to a NEW scaffold. The cross-scaffold backbone is frozen; only a small
classifier is fit on a handful (8-24) of labelled candidates of the new TF. This is the reliable
deployment mode: a cross-scaffold prior + a little scaffold-specific calibration, without retraining.

Features are the model's per-candidate functional read-outs (S_design, the release scalars, the
binding logit, the class logits) - all scaffold-agnostic quantities the shared body already produces.
"""
import numpy as np
import torch


class FewShotCalibrator:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.clf = None
        self.fallback = False

    @torch.no_grad()
    def _features(self, s):
        out = self.model(s.to(self.device))
        base = [float(out["S_design"]), float(out["ddG_coupling"]), float(out["logKd_apo"]),
                float(out["logKd_target"]), float(out["bind_logit"])]
        return base + [float(x) for x in out["class_logits"]]

    def fit(self, samples, labels):
        """samples: TransferSamples of the NEW scaffold. labels: 1 = functional_sensor else 0."""
        X = np.array([self._features(s) for s in samples])
        y = np.asarray(labels, int)
        if len(set(y.tolist())) < 2:
            self.fallback = True                      # cannot calibrate from one class; use raw S_design
            return self
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        return self

    @torch.no_grad()
    def predict(self, samples):
        """-> calibrated P(functional_sensor) per candidate."""
        if self.fallback or self.clf is None:
            return np.array([self._features(s)[0] for s in samples])   # raw S_design
        X = np.array([self._features(s) for s in samples])
        return self.clf.predict_proba(X)[:, 1]
