"""Simulated RF environment with ground truth. See contracts/environment.md.

Every emitter family occupies at most one band per slot, so per-emitter state is
stored as two [n_emitters, n_slots] arrays (band index or -1, and SNR) instead of
a dense [n_emitters, n_slots, n_bands] cube; `per_emitter_present()` expands it.
"""
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from core.log import boundary, get_logger

log = get_logger("environment")

FAMILIES = {"circular_scan", "agile", "fhss", "bursty", "continuous"}


@dataclass(frozen=True)
class EmitterSpec:
    id: str
    family: str
    threat_weight: float
    params: dict = field(default_factory=dict)
    appear_slot: int = 0
    vanish_slot: int | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    n_bands: int
    n_slots: int
    slot_ms: float
    seed: int
    emitters: list
    snr_floor_db: float = 0.0


@dataclass(frozen=True)
class Event:
    emitter_id: str
    band: int
    start: int
    end_exclusive: int
    threat_weight: float


@dataclass
class Environment:
    spec: ScenarioSpec
    emitter_ids: list
    threat_weights: np.ndarray   # [E]
    emitter_band: np.ndarray     # [E, T] int, -1 when not present (below floor or off)
    emitter_snr_db: np.ndarray   # [E, T] float, SNR when present
    snr_db: np.ndarray           # [T, B] power sum of present emitters, -inf when none
    truth: np.ndarray            # [T, B] bool
    events: list
    event_of: np.ndarray         # [E, T] int, index into events, -1 when not present

    def per_emitter_present(self) -> np.ndarray:
        E, T = self.emitter_band.shape
        out = np.zeros((E, T, self.spec.n_bands), dtype=bool)
        e_idx, t_idx = np.nonzero(self.emitter_band >= 0)
        out[e_idx, t_idx, self.emitter_band[e_idx, t_idx]] = True
        return out

    def band_emitters(self, t: int, bands) -> dict:
        """Present emitters in the given bands at slot t: band -> [(id, snr_db, threat_weight)]."""
        col = self.emitter_band[:, t]
        out: dict = {}
        for e in np.flatnonzero(col >= 0):
            b = int(col[e])
            if b in bands:
                out.setdefault(b, []).append(
                    (self.emitter_ids[e], float(self.emitter_snr_db[e, t]), float(self.threat_weights[e])))
        return out

    def oracle_scores(self, t: int, intercepted: set) -> np.ndarray:
        """Threat weight of un-intercepted events active in each band at slot t (oracle only)."""
        scores = np.zeros(self.spec.n_bands)
        col = self.emitter_band[:, t]
        for e in np.flatnonzero(col >= 0):
            ev = int(self.event_of[e, t])
            if ev not in intercepted:
                scores[col[e]] += self.threat_weights[e]
        return scores


def _lognormal_durations(rng, mean: float, sigma: float, size: int) -> np.ndarray:
    mu = math.log(mean) - sigma ** 2 / 2
    return np.maximum(1, np.rint(rng.lognormal(mu, sigma, size))).astype(np.int64)


def _on_off(rng, n_slots: int, on_mean: float, off_mean: float, sigma: float) -> np.ndarray:
    """Semi-Markov on/off process with log-normal dwell durations."""
    on = np.zeros(n_slots, dtype=bool)
    state = rng.random() < on_mean / (on_mean + off_mean)
    t = 0
    while t < n_slots:
        d = int(_lognormal_durations(rng, on_mean if state else off_mean, sigma, 1)[0])
        if state:
            on[t:t + d] = True
        t += d
        state = not state
    return on


def _require(p: dict, keys, eid: str) -> None:
    missing = [k for k in keys if k not in p]
    if missing:
        raise ValueError(f"emitter {eid}: missing params {missing}")


def _emitter_track(e: EmitterSpec, n_slots: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Returns (band[T] with -1 when silent, snr_db[T]) before the floor is applied."""
    p = e.params
    t = np.arange(n_slots)
    band = np.full(n_slots, -1, dtype=np.int64)
    snr = np.full(n_slots, -np.inf)

    if e.family == "circular_scan":
        _require(p, ("band", "period", "beam_slots", "phase", "snr_main_db", "sidelobe_db"), e.id)
        if p["period"] < 1 or not 1 <= p["beam_slots"] <= p["period"]:
            raise ValueError(f"emitter {e.id}: need 1 <= beam_slots <= period")
        if p["sidelobe_db"] > 0:
            raise ValueError(f"emitter {e.id}: sidelobe_db must be <= 0")
        main = ((t - p["phase"]) % p["period"]) < p["beam_slots"]
        band[:] = p["band"]
        snr[:] = np.where(main, p["snr_main_db"], p["snr_main_db"] + p["sidelobe_db"])
    elif e.family == "continuous":
        _require(p, ("band", "snr_db"), e.id)
        band[:] = p["band"]
        snr[:] = p["snr_db"]
    elif e.family == "bursty":
        _require(p, ("band", "on_mean_slots", "off_mean_slots", "sigma", "snr_db"), e.id)
        on = _on_off(rng, n_slots, p["on_mean_slots"], p["off_mean_slots"], p["sigma"])
        band[on] = p["band"]
        snr[on] = p["snr_db"]
    elif e.family == "agile":
        _require(p, ("bands", "burst_slots", "gap_slots", "order", "snr_db"), e.id)
        bands = list(p["bands"])
        if not bands or p["burst_slots"] < 1 or p["gap_slots"] < 0:
            raise ValueError(f"emitter {e.id}: bad agile params")
        cycle = p["burst_slots"] + p["gap_slots"]
        rel = t - p.get("phase", 0)
        k = np.floor_divide(rel, cycle)
        in_burst = (rel >= 0) & (np.mod(rel, cycle) < p["burst_slots"])
        if p["order"] == "cyclic":
            hop = np.asarray(bands)[np.mod(k, len(bands))]
        elif p["order"] == "uniform":
            n_bursts = int(k.max()) + 1 if n_slots else 0
            hop = np.asarray(bands)[rng.integers(0, len(bands), max(n_bursts, 1))][np.clip(k, 0, None)]
        else:
            raise ValueError(f"emitter {e.id}: order must be cyclic or uniform")
        band[in_burst] = hop[in_burst]
        snr[in_burst] = p["snr_db"]
    elif e.family == "fhss":
        _require(p, ("bands", "sequence_len", "snr_db", "on_mean_slots", "off_mean_slots", "sigma"), e.id)
        seq = rng.choice(np.asarray(p["bands"]), size=p["sequence_len"])
        hop_slots = p.get("hop_slots", 1)
        on = _on_off(rng, n_slots, p["on_mean_slots"], p["off_mean_slots"], p["sigma"])
        hop = seq[(t // hop_slots) % p["sequence_len"]]
        band[on] = hop[on]
        snr[on] = p["snr_db"]
    else:
        raise ValueError(f"emitter {e.id}: unknown family {e.family!r}")
    return band, snr


def _runs(band_row: np.ndarray):
    """Maximal runs of equal, non-negative band values: yields (band, start, end_exclusive)."""
    n = len(band_row)
    if n == 0:
        return
    change = np.flatnonzero(np.diff(band_row)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    for s, e in zip(starts, ends):
        if band_row[s] >= 0:
            yield int(band_row[s]), int(s), int(e)


def build(spec: ScenarioSpec) -> Environment:
    if spec.n_bands < 1 or spec.n_slots < 1 or spec.slot_ms <= 0:
        raise ValueError("n_bands, n_slots must be >= 1 and slot_ms > 0")
    ids = [e.id for e in spec.emitters]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate emitter id")

    E, T, B = len(spec.emitters), spec.n_slots, spec.n_bands
    emitter_band = np.full((E, T), -1, dtype=np.int64)
    emitter_snr = np.full((E, T), -np.inf)
    for i, e in enumerate(spec.emitters):
        if e.family not in FAMILIES:
            raise ValueError(f"emitter {e.id}: unknown family {e.family!r}")
        if not 0.0 <= e.threat_weight <= 1.0:
            raise ValueError(f"emitter {e.id}: threat_weight must be in [0, 1]")
        if e.appear_slot < 0 or (e.vanish_slot is not None and e.vanish_slot <= e.appear_slot):
            raise ValueError(f"emitter {e.id}: need 0 <= appear_slot < vanish_slot")
        bands_used = e.params.get("bands", [e.params.get("band")])
        if any(b is None or not 0 <= b < B for b in bands_used):
            raise ValueError(f"emitter {e.id}: band out of range [0, {B})")
        band, snr = _emitter_track(e, T, np.random.default_rng([spec.seed, i]))
        alive = np.zeros(T, dtype=bool)
        alive[e.appear_slot:e.vanish_slot] = True
        present = alive & (band >= 0) & (snr >= spec.snr_floor_db)
        emitter_band[i, present] = band[present]
        emitter_snr[i, present] = snr[present]

    lin = np.zeros((T, B))
    e_idx, t_idx = np.nonzero(emitter_band >= 0)
    np.add.at(lin, (t_idx, emitter_band[e_idx, t_idx]), 10.0 ** (emitter_snr[e_idx, t_idx] / 10.0))
    with np.errstate(divide="ignore"):
        snr_db = np.where(lin > 0, 10.0 * np.log10(lin), -np.inf)
    truth = snr_db >= spec.snr_floor_db

    raw_events = []
    for i, e in enumerate(spec.emitters):
        for b, s, end in _runs(emitter_band[i]):
            raw_events.append((s, e.id, b, end, e.threat_weight, i))
    raw_events.sort()
    events, event_of = [], np.full((E, T), -1, dtype=np.int64)
    for k, (s, eid, b, end, w, i) in enumerate(raw_events):
        events.append(Event(eid, b, s, end, w))
        event_of[i, s:end] = k

    env = Environment(spec, ids, np.array([e.threat_weight for e in spec.emitters], dtype=float),
                      emitter_band, emitter_snr, snr_db, truth, events, event_of)
    boundary(log, logging.INFO, "built", seed=spec.seed, n_emitters=E, n_slots=T, n_bands=B,
             n_events=len(events), truth_occupancy=round(float(truth.mean()), 5))
    return env
