"""Multi-scaffold trainer. Batches mix scaffolds (via HierarchicalScaffoldSampler), the model is run
per sample (variable residue/atom counts), and the shuffle controls feed the mechanistic term. It logs
BOTH the raw and the weighted contribution of every loss term, so it is visible if response-transfer
starts overpowering the functional task. Checkpointing saves model+optimiser+step for exact resume.
"""
import json
import os
import random

import torch

from ..losses import AlloTransferLoss, sample_labels, mechanistic_penalty
from .shuffle_controls import ligand_shuffled, path_shuffled


class Trainer:
    def __init__(self, model, lr=1e-3, device="cpu", lambdas=None, mech_fraction=0.5, seed=0):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = AlloTransferLoss(lambdas)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mech_fraction = mech_fraction
        self.step = 0
        self.rng = random.Random(seed)

    def _mech(self, samples):
        """Run the shuffle controls on a fraction of the batch -> mean mechanistic penalty."""
        k = max(1, int(len(samples) * self.mech_fraction))
        chosen = self.rng.sample(range(len(samples)), min(k, len(samples)))
        pens = []
        for i in chosen:
            s = samples[i]
            donor = next((samples[j] for j in range(len(samples)) if samples[j].ligand_id != s.ligand_id), None)
            out_real = self.model(s)
            out_lig = self.model(ligand_shuffled(s, donor)) if donor is not None else None
            out_path = self.model(path_shuffled(s, seed=self.step))
            pens.append(mechanistic_penalty(out_real, out_lig, out_path))
        return torch.stack(pens).mean() if pens else None

    def train_step(self, samples):
        samples = [s.to(self.device) for s in samples]
        outputs = [self.model(s) for s in samples]
        labels = [sample_labels(s) for s in samples]
        mech = self._mech(samples)
        total, parts = self.loss_fn(outputs, labels, mech=mech)
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        self.step += 1
        weighted = {k: self.loss_fn.l.get(k, 1.0) * v for k, v in parts.items()}
        return {"step": self.step, "total": float(total), "raw": parts, "weighted": weighted}

    def fit(self, samples, sampler, epochs=1, batch_size=8, log_path=None, ckpt_path=None, ckpt_every=50):
        for _ in range(epochs):
            order = list(iter(sampler))
            for b in range(0, len(order), batch_size):
                batch = [samples[i] for i in order[b:b + batch_size]]
                if not batch:
                    continue
                m = self.train_step(batch)
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(m) + "\n")
                if ckpt_path and self.step % ckpt_every == 0:
                    self.save(ckpt_path)
        return self

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "step": self.step}, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.step = ck["step"]
        return self

    @torch.no_grad()
    def evaluate(self, samples):
        """S_design + functional/binder labels -> the metrics that matter (sensor vs binder-only)."""
        from ..evaluation import functional_metrics
        self.model.eval()
        scores, func, is_binder = [], [], []
        for s in samples:
            out = self.model(s.to(self.device))
            scores.append(float(out["S_design"]))
            y = int(s.functional_class) if bool(s.functional_class_label_mask) else -1
            func.append(1 if y == 0 else 0)               # functional_sensor
            is_binder.append(y in (0, 1))                 # sensor or binder_only both bind
        self.model.train()
        return functional_metrics(scores, func, is_binder)
