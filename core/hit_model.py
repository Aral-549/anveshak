"""Supervised hit model: P(a dwell now lands on a new threat event) per band. See contracts/hit_model.md."""
import json
from pathlib import Path

import numpy as np

from core.belief import Belief

HIT_FEATURES = ["window", "window_threat", "soon", "time_to_window", "lock_period", "confirm", "confirm_threat",
                "staleness", "novel_mean", "novel_evidence", "band_threat", "occ_mean", "long_here", "since_novel"]
SOON_SLOTS = 16
TTW_CLIP = (-5, 200)
SINCE_CLIP = 12.0
BENIGN_MAX = 0.1
EVAL_SEED_MIN = 1000


def hit_features(t: int, belief: Belief, T_max: int) -> np.ndarray:
    B, tol = belief.cfg.n_bands, belief.cfg.period_tol
    X = np.zeros((B, len(HIT_FEATURES)))
    X[:, 3] = 1.0
    X[:, 13] = SINCE_CLIP
    soonest = np.full(B, np.inf)
    last_start = np.full(B, -1)
    for tr in belief.sigs.values():
        if tr.event_starts and tr.bands:
            b = tr.bands[-1]
            last_start[b] = max(last_start[b], tr.event_starts[-1])
        if tr.lock is not None and tr.next_window is not None:
            b, d = tr.lock.band, tr.next_window - t
            if abs(d) <= tol:
                X[b, 0] = 1.0
                X[b, 1] = max(X[b, 1], tr.threat_hint)
            elif tol < d <= tol + SOON_SLOTS:
                X[b, 2] = 1.0
            if d < soonest[b]:
                soonest[b] = d
                X[b, 3] = min(max(d, TTW_CLIP[0]), TTW_CLIP[1]) / TTW_CLIP[1]
                X[b, 4] = tr.lock.period / 1000.0
        if tr.confirm_until is not None and t <= tr.confirm_until and tr.lock is None and not tr.long:
            b = tr.confirm_band
            X[b, 5] = 1.0
            X[b, 6] = max(X[b, 6], tr.threat_hint)
    an, bn = belief.alpha_novel, belief.beta_novel
    X[:, 7] = np.minimum(belief.staleness(t) / T_max, 2.0)
    X[:, 8] = an / (an + bn)
    X[:, 9] = 1.0 / (an + bn)
    X[:, 10] = belief.band_threat
    X[:, 11] = belief.occ_mean
    X[:, 12] = belief.long_here()
    seen = last_start >= 0
    X[seen, 13] = np.minimum((t - last_start[seen]) / 1000.0, SINCE_CLIP)
    return X


def label_grid(events, n_slots: int, n_bands: int, first_intercept: dict) -> np.ndarray:
    """y[t, b]: a non-benign event is active in band b at t and was not intercepted before t."""
    y = np.zeros((n_slots, n_bands), dtype=bool)
    for k, e in enumerate(events):
        if e.threat_weight <= BENIGN_MAX:
            continue
        end = min(e.end_exclusive, n_slots)
        if k in first_intercept:
            end = min(end, first_intercept[k] + 1)
        if e.start < end:
            y[e.start:end, e.band] = True
    return y


def assert_train_seeds(seeds) -> None:
    bad = [s for s in seeds if s >= EVAL_SEED_MIN]
    assert not bad, f"evaluation seeds used for training: {bad[:5]}"


class HitModel:
    def __init__(self, booster):
        self.booster = booster

    @classmethod
    def load(cls, path) -> "HitModel":
        names = json.loads(Path(path).read_text()).get("learner", {}).get("feature_names")
        if names != HIT_FEATURES:
            raise ValueError(f"feature mismatch: model has {names}, code expects {HIT_FEATURES}")
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(path))
        booster.set_param({"nthread": 1})
        return cls(booster)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.booster.inplace_predict(np.asarray(X, dtype=np.float32))

    def score_fn(self, T_max: int):
        return lambda t, belief: self.score(hit_features(t, belief, T_max))
