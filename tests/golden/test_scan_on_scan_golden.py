"""Golden cases for contracts/scan_on_scan.md. Frozen: add cases, never edit."""
import pytest

from core.scan_on_scan import analyze


def test_case1_coprime_small():
    r = analyze(5, 1, 4, 1)
    assert (r.g, r.L) == (1, 20)
    assert r.frac_phases_intercepting == 1.0
    assert r.mean_interval_slots == 20


def test_case2_commensurate_half_phases():
    r = analyze(6, 1, 4, 1)
    assert (r.g, r.L) == (2, 12)
    assert r.frac_phases_intercepting == 0.5


def test_case3_sync_never():
    r = analyze(6, 1, 4, 1, delta=1)
    assert r.ever_intercepts is False
    assert r.mean_interval_slots is None


def test_case4_sync_intercepting_phase():
    r = analyze(6, 1, 4, 1, delta=0)
    assert r.ever_intercepts is True
    assert r.mean_interval_slots == 12


def test_case5_sync_trap_32_band_sweep():
    r = analyze(320, 2, 32, 1)
    assert r.g == 32
    assert r.frac_phases_intercepting == pytest.approx(0.0625)


def test_case6_coprime_realistic():
    r = analyze(401, 2, 32, 1)
    assert r.g == 1
    assert r.frac_phases_intercepting == 1.0
    assert r.mean_interval_slots == 6416
    assert r.mean_interval_slots_coprime == 6416


def test_case7_always_on_emitter():
    r = analyze(10, 10, 7, 1)
    assert r.frac_phases_intercepting == 1.0
    assert r.mean_interval_slots == 7


@pytest.mark.parametrize("args", [(10, 0, 4, 1), (0, 1, 4, 1), (5, 1, 0, 1), (5, 1, 4, 0)])
def test_case8_zero_rejected(args):
    with pytest.raises(ValueError):
        analyze(*args)


def test_case9_window_longer_than_period_rejected():
    with pytest.raises(ValueError):
        analyze(5, 6, 4, 1)
