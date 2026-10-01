"""Belief state learned from the receiver's own hits and misses. See contracts/belief.md.

Never reads truth.
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from core.log import boundary, get_logger
from core.receiver import Observation

log = get_logger("belief")


@dataclass
class BeliefConfig:
    n_bands: int
    gamma: float = 0.99
    period_tol: int = 2
    min_period: int = 40
    max_period: int = 1300
    unlock_misses: int = 3
    max_starts: int = 9          # period search uses at most this many recent event starts
    hop_confidence: float = 0.9  # multi-band signatures are predicted only above this
    hop_min_samples: int = 3
    long_run: int = 12           # consecutive detections that mark a signature as long-duration
    camp_fraction: float = 0.6   # share of slots visited between two starts that counts as a camp
    max_confirms: int = 3        # CONFIRMING entries allowed without a lock


@dataclass
class Lock:
    period: int
    last_start: int
    band: int


@dataclass
class SigTrack:
    event_starts: list = field(default_factory=list)
    lock: Lock | None = None
    next_window: int | None = None
    consecutive_misses: int = 0
    threat_hint: float = 0.0
    bands: list = field(default_factory=list)            # band of each event start
    transitions: dict = field(default_factory=lambda: defaultdict(Counter))
    window_watch: int = 0
    window_detected: bool = False
    run_len: int = 0
    long: bool = False
    confirm_band: int | None = None
    confirm_until: int | None = None
    multiband: bool = False
    confirm_attempts: int = 0
    visits_at_start: int = 0     # band visit count, including the start visit, at the last event start


def find_period(starts, tol: int, min_period: int, max_period: int) -> int | None:
    """Largest period T such that every interval is within tol of a positive multiple of T.

    Near the largest feasible T there is usually a contiguous block of feasible values
    (e.g. 199, 200, 201 for exact 200-slot intervals); the block member with the
    smallest squared residual is returned.
    """
    if len(starts) < 3:
        return None
    iv = np.diff(np.asarray(starts, dtype=np.int64))
    if (iv <= 0).any():
        return None
    upper = min(max_period, int(iv.min()) + tol)

    def residual(T: int) -> float | None:
        n = np.maximum(1, np.rint(iv / T))
        r = np.abs(iv - n * T)
        return float((r ** 2).sum()) if (r <= tol).all() else None

    block = []
    for T in range(upper, min_period - 1, -1):
        r = residual(T)
        if r is not None:
            block.append((r, -T))
        elif block:
            break
    if not block:
        return None
    return -min(block)[1]


class Belief:
    def __init__(self, config: BeliefConfig):
        B = config.n_bands
        if B < 1 or not 0.0 < config.gamma <= 1.0:
            raise ValueError("need n_bands >= 1 and gamma in (0, 1]")
        self.cfg = config
        self.alpha = np.ones(B)
        self.beta = np.ones(B)
        self.alpha_novel = np.ones(B)
        self.beta_novel = np.ones(B)
        self.band_threat = np.full(B, 0.5)
        self._band_threat_seen = np.zeros(B, dtype=bool)
        self.last_visit = np.full(B, -1, dtype=np.int64)
        self.visits = np.zeros(B, dtype=np.int64)
        self._last_visit_sigs: list[frozenset] = [frozenset()] * B
        self.sigs: dict[str, SigTrack] = {}
        self.predictions: list[tuple[int, str, int]] = []

    # ----- read side -------------------------------------------------------
    @property
    def occ_mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    @property
    def novel_mean(self) -> np.ndarray:
        return self.alpha_novel / (self.alpha_novel + self.beta_novel)

    def staleness(self, t: int) -> np.ndarray:
        return t - self.last_visit

    def long_here(self) -> np.ndarray:
        """1.0 where a long-duration signature was detected at the band's last visit."""
        return np.array([float(any(self.sigs[s].long for s in sigs)) for sigs in self._last_visit_sigs])

    # ----- write side ------------------------------------------------------
    def mark_visited(self, t: int, bands) -> None:
        for b in bands:
            self.last_visit[b] = t
            self.visits[b] += 1

    def update(self, obs: Observation) -> None:
        t, g = obs.slot, self.cfg.gamma
        detected_now: set[str] = set()
        for b in sorted(obs.per_band):
            bo = obs.per_band[b]
            hit = bo.detected
            self.alpha[b] = g * self.alpha[b] + hit
            self.beta[b] = g * self.beta[b] + (not hit)

            sigs_here = frozenset(d.signature for d in bo.detections)
            novel = sigs_here - self._last_visit_sigs[b]
            for gone in self._last_visit_sigs[b] - sigs_here:
                self.sigs[gone].run_len = 0
            self.alpha_novel[b] = g * self.alpha_novel[b] + bool(novel)
            self.beta_novel[b] = g * self.beta_novel[b] + (not novel)

            for d in bo.detections:
                tr = self.sigs.setdefault(d.signature, SigTrack())
                tr.threat_hint = max(tr.threat_hint, d.threat_hint)
                tr.run_len = 1 if d.signature in novel else tr.run_len + 1
                if tr.run_len > self.cfg.long_run and not tr.long:
                    tr.long = True
                    tr.confirm_band = tr.confirm_until = None
                    boundary(log, logging.INFO, "long_emitter", slot=t, sig=d.signature, band=b)
                if d.signature in novel:
                    if not self._band_threat_seen[b]:
                        self.band_threat[b] = d.threat_hint
                        self._band_threat_seen[b] = True
                    else:
                        self.band_threat[b] = max(self.band_threat[b], d.threat_hint)
                    # Two novel starts of one signature in one slot (M > 1): keep the lowest band.
                    if d.signature not in detected_now and (not tr.event_starts or tr.event_starts[-1] != t):
                        self._new_event(d.signature, tr, t, b)
                detected_now.add(d.signature)
            self._last_visit_sigs[b] = sigs_here
        self.mark_visited(t, obs.per_band.keys())
        self._score_windows(t, obs, detected_now)
        for tr in self.sigs.values():
            if tr.confirm_until is not None and t > tr.confirm_until:
                tr.confirm_band = tr.confirm_until = None
        boundary(log, logging.DEBUG, "updated", slot=t, bands=list(obs.per_band),
                 detected=sorted(detected_now), locks=sum(1 for s in self.sigs.values() if s.lock))

    # ----- internals -------------------------------------------------------
    def _new_event(self, sig: str, tr: SigTrack, t: int, b: int) -> None:
        prev_start = tr.event_starts[-1] if tr.event_starts else None
        prev_band = tr.bands[-1] if tr.bands else None
        visits_now = int(self.visits[b]) + 1  # this slot's visit is not yet counted
        camped = (prev_band == b and prev_start is not None
                  and (visits_now - tr.visits_at_start) >= self.cfg.camp_fraction * (t - prev_start))
        if prev_band is not None:
            tr.transitions[prev_band][b] += 1
        tr.event_starts.append(t)
        tr.bands.append(b)
        if prev_band is not None and prev_band != b:
            tr.multiband = True
        tr.visits_at_start = visits_now
        if tr.lock is not None and tr.next_window is not None and abs(t - tr.next_window) <= self.cfg.period_tol:
            tr.consecutive_misses = 0

        starts = tr.event_starts[-self.cfg.max_starts:]
        period = None
        for n in range(len(starts), 2, -1):  # longest consistent recent suffix wins
            period = find_period(starts[-n:], self.cfg.period_tol, self.cfg.min_period, self.cfg.max_period)
            if period:
                break
        provisional = False
        if period is None and camped and self.cfg.min_period <= t - prev_start <= self.cfg.max_period:
            period, provisional = t - prev_start, True
        tr.window_watch, tr.window_detected = 0, False
        if period is None:
            if tr.lock is not None:
                boundary(log, logging.INFO, "lock_dropped", slot=t, sig=sig, reason="inconsistent_event")
            tr.lock, tr.next_window = None, None
            if not (tr.long or tr.multiband) and tr.confirm_attempts < self.cfg.max_confirms:
                tr.confirm_attempts += 1
                tr.confirm_band, tr.confirm_until = b, t + self.cfg.max_period + self.cfg.period_tol
            else:
                tr.confirm_band = tr.confirm_until = None
            return
        band = self._predict_band(tr, b)
        was_locked = tr.lock is not None
        tr.confirm_band = tr.confirm_until = None
        tr.confirm_attempts = 0
        tr.lock = Lock(period, t, band if band is not None else b)
        tr.next_window = t + period if band is not None else None
        if not was_locked:
            tr.consecutive_misses = 0
            boundary(log, logging.INFO, "lock_acquired", slot=t, sig=sig, period=period, band=tr.lock.band,
                     provisional=provisional)
        if tr.next_window is not None:
            self.predictions.append((t, sig, tr.next_window))

    def _predict_band(self, tr: SigTrack, b: int) -> int | None:
        if len(set(tr.bands)) == 1:
            return b
        nxt = tr.transitions.get(b)
        if not nxt:
            return None
        total = sum(nxt.values())
        band, count = nxt.most_common(1)[0]
        if total >= self.cfg.hop_min_samples and count / total >= self.cfg.hop_confidence:
            return band
        return None

    def _score_windows(self, t: int, obs: Observation, detected_now: set) -> None:
        tol, K = self.cfg.period_tol, self.cfg.unlock_misses
        for sig, tr in self.sigs.items():
            if tr.lock is None or tr.next_window is None:
                continue
            nw = tr.next_window
            if nw - tol <= t <= nw + tol:
                if tr.lock.band in obs.per_band:
                    tr.window_watch += 1
                if sig in detected_now:
                    tr.window_detected = True
            if t >= nw + tol:
                if tr.window_detected:
                    tr.consecutive_misses = 0
                elif tr.window_watch == 2 * tol + 1:
                    tr.consecutive_misses += 1
                tr.window_watch, tr.window_detected = 0, False
                if tr.consecutive_misses >= K:
                    boundary(log, logging.INFO, "lock_dropped", slot=t, sig=sig, reason="missed_windows")
                    tr.lock, tr.next_window, tr.consecutive_misses = None, None, 0
                    continue
                while tr.next_window + tol <= t:
                    tr.next_window += tr.lock.period
                self.predictions.append((t, sig, tr.next_window))
