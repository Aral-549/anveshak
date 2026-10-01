"""Golden cases for contracts/receiver.md. Frozen: add cases, never edit."""
import math

import numpy as np
import pytest

from core.receiver import Receiver, pd, snr_for_pd


@pytest.mark.parametrize("pfa", [1e-6, 1e-4, 1e-2, 0.3])
def test_case1_noise_only_limit(pfa):
    assert pd(-math.inf, pfa) == pytest.approx(pfa, rel=1e-9)


def test_case2_strong_signal():
    assert pd(100.0, 1e-6) == pytest.approx(1.0, abs=1e-12)


def test_case3_monotone():
    vals = [pd(s, 1e-4) for s in np.arange(-10, 30.01, 0.25)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_case4_sensitivity_matches_albersheim():
    assert snr_for_pd(0.9, 1e-6) == pytest.approx(13.1, abs=0.3)


def _rx(pfa=1e-4, M=1, noise=0.0, seed=0):
    return Receiver(n_bands=8, M=M, pfa=pfa, threat_hint_noise=noise,
                    rng=np.random.default_rng(seed))


def test_case5_no_leakage_between_bands():
    obs = _rx().observe(0, [4], {5: [("A", 60.0, 1.0)]})
    assert obs.per_band[4].detections == []


def test_case6_false_alarm_certain():
    b = _rx(pfa=1.0).observe(0, [2], {}).per_band[2]
    assert b.false_alarm is True and b.detected is True and b.detections == []


def test_case7_false_alarm_negligible():
    rx = _rx(pfa=1e-12)
    assert not any(rx.observe(t, [2], {}).per_band[2].false_alarm for t in range(10_000))


def test_case8_duplicate_dwell_rejected():
    with pytest.raises(ValueError):
        _rx(M=2).observe(0, [1, 1], {})


def test_case9_too_many_dwells_rejected():
    with pytest.raises(ValueError):
        _rx(M=1).observe(0, [1, 2], {})


def test_case10_band_out_of_range_rejected():
    with pytest.raises(ValueError):
        _rx().observe(0, [8], {})


def test_case11_threat_hint_exact_without_noise():
    d = _rx().observe(0, [3], {3: [("A", 60.0, 0.7)]}).per_band[3].detections
    assert [(x.signature, x.threat_hint) for x in d] == [("A", 0.7)]


def test_case12_empirical_pd_at_sensitivity():
    snr = snr_for_pd(0.9, 1e-6)
    rx = Receiver(n_bands=1, M=1, pfa=1e-6, threat_hint_noise=0.0, rng=np.random.default_rng(12))
    n = 200_000
    hits = sum(bool(rx.observe(t, [0], {0: [("A", snr, 1.0)]}).per_band[0].detections) for t in range(n))
    assert 0.895 <= hits / n <= 0.905
