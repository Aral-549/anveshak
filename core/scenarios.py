"""Scenario suites S1-S5 (DESIGN.md section 8). Deterministic in (suite, seed).

Evaluation seeds are 1000-1099; training seeds (PPO) are 0-999. Never mix them.
"""
import numpy as np

from core.environment import EmitterSpec, ScenarioSpec

N_BANDS, N_SLOTS, SLOT_MS = 32, 12_000, 10.0
EVAL_SEEDS = range(1000, 1100)
SUITES = ("S1", "S2", "S3", "S4", "S5")


class _Gen:
    def __init__(self, rng: np.random.Generator):
        self.rng, self.n = rng, 0

    def _id(self, prefix):
        self.n += 1
        return f"{prefix}{self.n:02d}"

    def _band(self):
        return int(self.rng.integers(0, N_BANDS))

    def search_radar(self, period=(200, 600), beam=(1, 3), period_value=None, w=0.6, prefix="SR", appear=0):
        r = self.rng
        T = int(period_value if period_value is not None else r.integers(period[0], period[1] + 1))
        return EmitterSpec(self._id(prefix), "circular_scan", w, dict(
            band=self._band(), period=T, beam_slots=int(r.integers(beam[0], beam[1] + 1)),
            phase=int(r.integers(0, T)), snr_main_db=float(r.uniform(12, 25)),
            sidelobe_db=float(r.uniform(-45, -32))), appear_slot=appear)

    def sector_radar(self, w=0.7, prefix="TW", appear=0, period=(50, 200)):
        return self.search_radar(period=period, beam=(2, 4), w=w, prefix=prefix, appear=appear)

    def agile(self, n_bands=(2, 6)):
        r = self.rng
        k = int(r.integers(n_bands[0], n_bands[1] + 1))
        start = int(r.integers(0, N_BANDS - k + 1))
        return EmitterSpec(self._id("AG"), "agile", 0.8, dict(
            bands=list(range(start, start + k)), burst_slots=int(r.integers(2, 6)),
            gap_slots=int(r.integers(20, 81)), order=str(r.choice(["cyclic", "uniform"])),
            phase=int(r.integers(0, 100)), snr_db=float(r.uniform(12, 25))))

    def fhss(self):
        r = self.rng
        k = int(r.integers(4, 13))
        return EmitterSpec(self._id("FH"), "fhss", 0.4, dict(
            bands=[int(b) for b in r.choice(N_BANDS, size=k, replace=False)], sequence_len=int(r.integers(4, 17)),
            on_mean_slots=float(r.uniform(100, 500)), off_mean_slots=float(r.uniform(200, 1500)),
            sigma=0.5, snr_db=float(r.uniform(10, 20))))

    def bursty(self):
        r = self.rng
        return EmitterSpec(self._id("BC"), "bursty", 0.4, dict(
            band=self._band(), on_mean_slots=float(r.uniform(50, 500)),
            off_mean_slots=float(r.uniform(200, 3000)), sigma=0.5, snr_db=float(r.uniform(10, 20))))

    def continuous(self):
        return EmitterSpec(self._id("CW"), "continuous", 0.1, dict(
            band=self._band(), snr_db=float(self.rng.uniform(10, 30))))


def make(suite: str, seed: int) -> ScenarioSpec:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}")
    rng = np.random.default_rng([seed, SUITES.index(suite)])
    g = _Gen(rng)
    em = []
    if suite == "S1":    # sparse search radars + benign beacons
        em += [g.search_radar() for _ in range(6)] + [g.continuous() for _ in range(3)]
    elif suite == "S2":  # frequency agile radars + FHSS / bursty comms
        em += [g.agile() for _ in range(3)] + [g.fhss() for _ in range(2)] + [g.bursty() for _ in range(2)]
        em += [g.continuous() for _ in range(2)] + [g.search_radar() for _ in range(2)]
    elif suite == "S3":  # dense benign background, one pop-up high-threat radar
        em += [g.continuous() for _ in range(8)] + [g.bursty() for _ in range(4)]
        em += [g.search_radar() for _ in range(3)]
        em.append(g.sector_radar(w=1.0, prefix="POPUP", appear=int(rng.integers(2000, 9001))))
    elif suite == "S4":  # synchronisation trap: radar periods are multiples of the 32-slot sweep
        em += [g.search_radar(period_value=32 * int(rng.integers(6, 19)), beam=(1, 2)) for _ in range(5)]
        em += [g.continuous() for _ in range(3)]
    elif suite == "S5":  # randomised mix incl. parameter ranges outside S1-S4
        em += [g.search_radar(period=(150, 1200), beam=(1, 4)) for _ in range(int(rng.integers(2, 9)))]
        em += [g.sector_radar() for _ in range(int(rng.integers(1, 4)))]
        em += [g.agile(n_bands=(2, 8)) for _ in range(int(rng.integers(0, 4)))]
        em += [g.fhss() for _ in range(int(rng.integers(0, 3)))]
        em += [g.bursty() for _ in range(int(rng.integers(0, 5)))]
        em += [g.continuous() for _ in range(int(rng.integers(1, 9)))]
        if rng.random() < 0.5:
            em.append(g.sector_radar(w=1.0, prefix="POPUP", appear=int(rng.integers(2000, 9001))))
    return ScenarioSpec(N_BANDS, N_SLOTS, SLOT_MS, seed, em)
