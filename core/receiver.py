"""ES receiver detection model. See contracts/receiver.md."""
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2


@dataclass(slots=True)
class Detection:
    signature: str
    threat_hint: float
    snr_db: float


@dataclass(slots=True)
class BandObservation:
    detected: bool
    detections: list = field(default_factory=list)
    false_alarm: bool = False


@dataclass(slots=True)
class Observation:
    slot: int
    per_band: dict  # band -> BandObservation, one entry per dwelt band


def pd(snr_db: float, pfa: float) -> float:
    """Square-law detector, non-fluctuating signal: Marcum Q1(sqrt(2 snr), sqrt(-2 ln pfa))."""
    if not 0.0 < pfa <= 1.0:
        raise ValueError(f"pfa must be in (0, 1], got {pfa}")
    snr_lin = 0.0 if snr_db == -math.inf else 10.0 ** (snr_db / 10.0)
    if snr_lin == 0.0:
        return pfa
    return float(ncx2.sf(-2.0 * math.log(pfa), 2, 2.0 * snr_lin))


@lru_cache(maxsize=65536)
def _pd_cached(snr_db_centi: int, pfa: float) -> float:
    return pd(snr_db_centi / 100.0, pfa)


def snr_for_pd(pd_target: float, pfa: float) -> float:
    """Sensitivity: SNR in dB needed for pd_target at the given pfa."""
    if not pfa < pd_target < 1.0:
        raise ValueError("need pfa < pd_target < 1")
    return float(brentq(lambda x: pd(x, pfa) - pd_target, -40.0, 60.0, xtol=1e-9))


class Receiver:
    def __init__(self, n_bands: int, M: int, pfa: float, threat_hint_noise: float, rng: np.random.Generator):
        if n_bands < 1 or M < 1:
            raise ValueError("n_bands and M must be >= 1")
        if not 0.0 < pfa <= 1.0:
            raise ValueError(f"pfa must be in (0, 1], got {pfa}")
        if threat_hint_noise < 0:
            raise ValueError("threat_hint_noise must be >= 0")
        self.n_bands, self.M, self.pfa = n_bands, M, pfa
        self.noise = threat_hint_noise
        self.rng = rng

    def _validate(self, dwell_bands) -> None:
        if not 1 <= len(dwell_bands) <= self.M:
            raise ValueError(f"need 1..{self.M} dwell bands, got {len(dwell_bands)}")
        if len(set(dwell_bands)) != len(dwell_bands):
            raise ValueError(f"duplicate dwell bands {dwell_bands}")
        for b in dwell_bands:
            if not 0 <= b < self.n_bands:
                raise ValueError(f"band {b} out of range [0, {self.n_bands})")

    def observe(self, slot: int, dwell_bands, band_emitters: dict) -> Observation:
        """band_emitters: band -> list of (emitter_id, snr_db, threat_weight), present emitters only."""
        self._validate(dwell_bands)
        rng = self.rng
        per_band = {}
        for b in dwell_bands:
            emitters = band_emitters.get(b, ())
            if emitters:
                dets = []
                for eid, snr_db, w in emitters:
                    p = _pd_cached(round(snr_db * 100), self.pfa) if snr_db != -math.inf else self.pfa
                    if rng.random() < p:
                        hint = w if self.noise == 0 else float(np.clip(w + rng.normal(0.0, self.noise), 0.0, 1.0))
                        dets.append(Detection(eid, hint, snr_db))
                per_band[b] = BandObservation(bool(dets), dets, False)
            else:
                fa = rng.random() < self.pfa
                per_band[b] = BandObservation(fa, [], fa)
        return Observation(slot, per_band)
