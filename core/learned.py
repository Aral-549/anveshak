"""Learned scheduling policy: shared per-band neural scorer trained by the cross-entropy method.

See contracts/learned_policy.md.
"""
import dataclasses
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from core import scenarios
from core.belief import Belief
from core.log import boundary, get_logger

log = get_logger("learned")

FEATURES = ["window", "window_threat", "confirm", "confirm_threat", "staleness", "novel_mean", "novel_ts",
            "novel_evidence", "band_threat", "occ_mean", "long_here", "soon"]
HIDDEN = 16
SOON_SLOTS = 16
TRAIN_SEEDS = range(0, 1000)


def features(t: int, belief: Belief, T_max: int, rng: np.random.Generator) -> np.ndarray:
    B, tol = belief.cfg.n_bands, belief.cfg.period_tol
    X = np.zeros((B, len(FEATURES)))
    for tr in belief.sigs.values():
        if tr.lock is not None and tr.next_window is not None:
            b, d = tr.lock.band, tr.next_window - t
            if abs(d) <= tol:
                X[b, 0] = 1.0
                X[b, 1] = max(X[b, 1], tr.threat_hint)
            elif tol < d <= tol + SOON_SLOTS:
                X[b, 11] = 1.0
        if tr.confirm_until is not None and t <= tr.confirm_until and tr.lock is None and not tr.long:
            b = tr.confirm_band
            X[b, 2] = 1.0
            X[b, 3] = max(X[b, 3], tr.threat_hint)
    an, bn = belief.alpha_novel, belief.beta_novel
    X[:, 4] = np.minimum(belief.staleness(t) / T_max, 2.0)
    X[:, 5] = an / (an + bn)
    X[:, 6] = rng.beta(an, bn)
    X[:, 7] = 1.0 / (an + bn)
    X[:, 8] = belief.band_threat
    X[:, 9] = belief.occ_mean
    X[:, 10] = belief.long_here()
    return X


@dataclasses.dataclass
class LearnedScorer:
    W1: np.ndarray  # [HIDDEN, F]
    b1: np.ndarray  # [HIDDEN]
    W2: np.ndarray  # [HIDDEN]
    b2: float

    @property
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + 1

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W1.T + self.b1) @ self.W2 + self.b2

    def score_fn(self, T_max: int, rng: np.random.Generator):
        return lambda t, belief: self.score(features(t, belief, T_max, rng))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2, [self.b2]])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "LearnedScorer":
        F = len(FEATURES)
        v = np.asarray(v, dtype=float)
        if v.size != HIDDEN * F + 2 * HIDDEN + 1:
            raise ValueError(f"expected {HIDDEN * F + 2 * HIDDEN + 1} parameters, got {v.size}")
        i = HIDDEN * F
        return cls(v[:i].reshape(HIDDEN, F), v[i:i + HIDDEN], v[i + HIDDEN:i + 2 * HIDDEN], float(v[-1]))

    def save(self, path, meta: dict | None = None) -> None:
        Path(path).write_text(json.dumps(dict(features=FEATURES, W1=self.W1.tolist(), b1=self.b1.tolist(),
                                              W2=self.W2.tolist(), b2=self.b2, meta=meta or {}), indent=1))

    @classmethod
    def load(cls, path) -> "LearnedScorer":
        d = json.loads(Path(path).read_text())
        if d.get("features") != FEATURES:
            raise ValueError(f"feature mismatch: weights expect {d.get('features')}, code provides {FEATURES}")
        s = cls(np.asarray(d["W1"], float), np.asarray(d["b1"], float), np.asarray(d["W2"], float), float(d["b2"]))
        if s.W1.shape != (HIDDEN, len(FEATURES)) or s.b1.shape != (HIDDEN,) or s.W2.shape != (HIDDEN,):
            raise ValueError("weight shapes do not match the scorer")
        return s


# ----- training (cross-entropy method) --------------------------------------
def _fitness_job(job) -> float:
    from core.sim import RunConfig, run  # local import keeps worker start-up light
    vec, suite, seed, n_slots, T_max, feat_seed = job
    spec = dataclasses.replace(scenarios.make(suite, seed), n_slots=n_slots)
    scorer = LearnedScorer.from_vector(vec)
    r = run(spec, RunConfig(policy="learned", T_max=T_max, seed=seed),
            score_fn=scorer.score_fn(T_max, np.random.default_rng(feat_seed)))
    w = r.report.weighted_interception_ratio
    return 0.0 if w is None else float(w)


def train(generations: int = 30, population: int = 32, elite: int = 6, scenarios_per_suite: int = 3,
          n_slots: int = 6000, T_max: int = 128, workers: int = 16, seed: int = 0, init_std: float = 0.5,
          out: str | None = None) -> dict:
    assert not set(TRAIN_SEEDS) & set(scenarios.EVAL_SEEDS), "training seeds overlap evaluation seeds"
    rng = np.random.default_rng([seed, 99])
    n = HIDDEN * len(FEATURES) + 2 * HIDDEN + 1
    mean = np.zeros(n)
    mean[:HIDDEN * len(FEATURES)] = rng.normal(0, 0.1, HIDDEN * len(FEATURES))
    mean[HIDDEN * len(FEATURES) + HIDDEN:HIDDEN * len(FEATURES) + 2 * HIDDEN] = rng.normal(0, 0.1, HIDDEN)
    std = np.full(n, init_std)
    curve = []
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for gen in range(generations):
            batch = [(s, int(rng.choice(TRAIN_SEEDS))) for s in scenarios.SUITES for _ in range(scenarios_per_suite)]
            cands = mean + std * rng.standard_normal((population, n))
            jobs = [(c, s, sd, n_slots, T_max, sd * 1000 + gen) for c in cands for s, sd in batch]
            res = list(ex.map(_fitness_job, jobs, chunksize=4)) if ex else [_fitness_job(j) for j in jobs]
            fit = np.asarray(res).reshape(population, len(batch)).mean(axis=1)
            order = np.argsort(-fit, kind="stable")[:elite]
            mean = cands[order].mean(axis=0)
            std = cands[order].std(axis=0) + 0.02
            curve.append(dict(gen=gen, best=round(float(fit[order[0]]), 6), elite_mean=round(float(fit[order].mean()), 6),
                              pop_mean=round(float(fit.mean()), 6)))
            boundary(log, logging.INFO, "generation", **curve[-1])
    finally:
        if ex:
            ex.shutdown()
    result = dict(fitness_curve=curve, params=mean.tolist(),
                  config=dict(generations=generations, population=population, elite=elite,
                              scenarios_per_suite=scenarios_per_suite, n_slots=n_slots, T_max=T_max, seed=seed,
                              init_std=init_std, train_seeds=[TRAIN_SEEDS.start, TRAIN_SEEDS.stop - 1]))
    if out:
        LearnedScorer.from_vector(mean).save(out, meta=result)
    return result
