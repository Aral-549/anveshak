"""Dwell scheduler: policies behind one interface, under the revisit floor.

See contracts/scheduler.md. Only the `oracle` policy sees truth (via oracle_scores),
and it is never deployable.
"""
import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.belief import Belief
from core.log import boundary, get_logger

log = get_logger("scheduler")

POLICIES = {"sweep", "permuted_sweep", "bandit", "predictive", "oracle", "custom", "learned", "hybrid"}


@dataclass
class DwellPlan:
    slot: int
    bands: list
    sources: list
    conflicts_dropped: list = field(default_factory=list)


def _top(scores: np.ndarray, exclude: set, k: int) -> list[int]:
    """Top-k band indices by score, highest first, ties to the lowest index."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(b) for b in order if int(b) not in exclude][:k]


class Scheduler:
    def __init__(self, policy: str, n_bands: int, M: int, T_max: int, seed: int = 0,
                 lam: float = 0.5, score_fn=None, hybrid_lam: float = 0.0):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        if M < 1 or M > n_bands:
            raise ValueError(f"need 1 <= M <= n_bands, got M={M}, n_bands={n_bands}")
        if T_max < math.ceil(n_bands / M):
            raise ValueError(f"revisit floor infeasible: T_max={T_max} < ceil({n_bands}/{M})")
        if policy in ("custom", "learned", "hybrid") and score_fn is None:
            raise ValueError(f"policy {policy!r} needs score_fn")
        self.policy, self.B, self.M, self.T_max = policy, n_bands, M, T_max
        self.lam, self.score_fn, self.hybrid_lam = lam, score_fn, hybrid_lam
        self.rng = np.random.default_rng([seed, 11])
        self._ptr = 0
        self._queue: deque = deque()

    # ----- revisit floor (earliest deadline first) ------------------------
    def forced_bands(self, staleness: np.ndarray) -> list[int]:
        slack = self.T_max - staleness
        order = np.lexsort((np.arange(self.B), slack))
        s = slack[order]
        js = np.maximum(s, 0)
        counts = np.searchsorted(s, js, side="right")
        r = int(max(0, (counts - self.M * js).max()))
        return [int(b) for b in order[:min(r, self.M)]]

    # ----- main entry -------------------------------------------------------
    def plan(self, t: int, belief: Belief, oracle_scores: np.ndarray | None = None) -> DwellPlan:
        staleness = belief.staleness(t)
        forced = self.forced_bands(staleness)
        bands, sources = list(forced), ["REVISIT_FLOOR"] * len(forced)
        k = self.M - len(forced)
        dropped: list[int] = []
        if k:
            chosen, srcs, dropped = self._policy(t, belief, set(forced), k, staleness, oracle_scores)
            bands += chosen
            sources += srcs
        plan = DwellPlan(t, bands, sources, dropped)
        boundary(log, logging.DEBUG, "planned", slot=t, bands=bands, sources=sources,
                 staleness_max=int(staleness.max()), conflicts_dropped=dropped)
        return plan

    def _policy(self, t, belief, exclude, k, staleness, oracle_scores):
        p = self.policy
        if p == "sweep":
            return self._sweep(exclude, k), ["SWEEP"] * k, []
        if p == "permuted_sweep":
            return self._permuted(exclude, k), ["SWEEP"] * k, []
        if p == "bandit":
            return _top(self._bandit_scores(belief), exclude, k), ["POLICY"] * k, []
        if p == "predictive":
            return self._predictive(t, belief, exclude, k, staleness)
        if p == "hybrid":
            base = np.asarray(self.score_fn(t, belief), dtype=float)
            if base.shape != (self.B,) or not np.isfinite(base).all():
                boundary(log, logging.WARNING, "policy_fallback", slot=t, reason="non-finite or mis-shaped scores")
                chosen, _, dropped = self._predictive(t, belief, exclude, k, staleness)
                return chosen, ["POLICY_FALLBACK"] * k, dropped
            return self._predictive(t, belief, exclude, k, staleness,
                                    base=base + self.hybrid_lam * staleness / self.T_max)
        if p == "oracle":
            if oracle_scores is None:
                raise ValueError("oracle policy needs oracle_scores")
            scores = oracle_scores if oracle_scores.max() > 0 else staleness.astype(float)
            return _top(np.asarray(scores, dtype=float), exclude, k), ["ORACLE"] * k, []
        scores = np.asarray(self.score_fn(t, belief), dtype=float)
        if scores.shape != (self.B,) or not np.isfinite(scores).all():
            boundary(log, logging.WARNING, "policy_fallback", slot=t, reason="non-finite or mis-shaped scores")
            chosen, _, dropped = self._predictive(t, belief, exclude, k, staleness)
            return chosen, ["POLICY_FALLBACK"] * k, dropped
        return _top(scores, exclude, k), ["POLICY"] * k, []

    def _sweep(self, exclude, k):
        chosen, i = [], 0
        while len(chosen) < k:
            b = (self._ptr + i) % self.B
            i += 1
            if b not in exclude:
                chosen.append(b)
        self._ptr = (chosen[-1] + 1) % self.B
        return chosen

    def _permuted(self, exclude, k):
        chosen = []
        while len(chosen) < k:
            if not self._queue:
                self._queue.extend(int(b) for b in self.rng.permutation(self.B))
            b = self._queue.popleft()
            if b not in exclude and b not in chosen:
                chosen.append(b)
        return chosen

    def _bandit_scores(self, belief: Belief) -> np.ndarray:
        return self.rng.beta(belief.alpha_novel, belief.beta_novel) * belief.band_threat

    def _predictive(self, t, belief, exclude, k, staleness, base=None):
        tol = belief.cfg.period_tol
        window_threat: dict[int, float] = {}
        for tr in belief.sigs.values():
            if tr.lock is not None and tr.next_window is not None and abs(t - tr.next_window) <= tol:
                b = tr.lock.band
                window_threat[b] = max(window_threat.get(b, 0.0), tr.threat_hint)
        confirm_threat: dict[int, float] = {}
        for tr in belief.sigs.values():
            if tr.confirm_until is not None and t <= tr.confirm_until and tr.lock is None and not tr.long:
                b = tr.confirm_band
                confirm_threat[b] = max(confirm_threat.get(b, 0.0), tr.threat_hint)
        scores = (self._bandit_scores(belief) + self.lam * staleness / self.T_max) if base is None else base.copy()
        for b, w in confirm_threat.items():
            scores[b] = 5.0 + w
        for b, w in window_threat.items():
            scores[b] = 10.0 + w
        chosen = _top(scores, exclude, k)
        sources = ["PREDICTED_WINDOW" if b in window_threat else "CONFIRM" if b in confirm_threat else "POLICY"
                   for b in chosen]
        dropped = sorted(b for b in window_threat if b not in chosen and b not in exclude)
        return chosen, sources, dropped
