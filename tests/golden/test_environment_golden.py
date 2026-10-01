"""Golden cases for contracts/environment.md. Frozen: add cases, never edit."""
import numpy as np
import pytest

from core.environment import EmitterSpec, ScenarioSpec, build


def spec(emitters, n_slots=300, n_bands=8, seed=0):
    return ScenarioSpec(n_bands=n_bands, n_slots=n_slots, slot_ms=10.0, seed=seed, emitters=emitters)


def radar(sidelobe=-40.0, **kw):
    return EmitterSpec(id=kw.get("id", "R"), family="circular_scan", threat_weight=0.6,
                       params=dict(band=5, period=100, beam_slots=2, phase=10,
                                   snr_main_db=30.0, sidelobe_db=sidelobe))


def test_case1_circular_scan_windows():
    env = build(spec([radar()]))
    assert np.flatnonzero(env.truth[:, 5]).tolist() == [10, 11, 110, 111, 210, 211]
    assert len(env.events) == 3
    assert [(e.start, e.end_exclusive) for e in env.events] == [(10, 12), (110, 112), (210, 212)]


def test_case2_continuous():
    env = build(spec([EmitterSpec(id="C", family="continuous", threat_weight=0.1,
                                  params=dict(band=0, snr_db=10.0))], n_slots=50))
    assert env.truth[:, 0].all()
    assert [(e.start, e.end_exclusive) for e in env.events] == [(0, 50)]


def test_case3_popup():
    env = build(spec([EmitterSpec(id="P", family="continuous", threat_weight=1.0,
                                  params=dict(band=3, snr_db=20.0), appear_slot=150)]))
    assert not env.truth[:150, 3].any() and env.truth[150:, 3].all()
    assert [(e.start, e.end_exclusive) for e in env.events] == [(150, 300)]


def test_case4_agile_cyclic():
    env = build(spec([EmitterSpec(id="G", family="agile", threat_weight=0.8,
                                  params=dict(bands=[3, 7], burst_slots=1, gap_slots=0,
                                              order="cyclic", snr_db=20.0))], n_slots=20))
    t = np.arange(20)
    assert (env.truth[:, 3] == (t % 2 == 0)).all()
    assert (env.truth[:, 7] == (t % 2 == 1)).all()
    assert len(env.events) == 20


def test_case5_strong_sidelobe_always_visible():
    env = build(spec([radar(sidelobe=-20.0)]))
    assert env.truth[:, 5].all()
    assert len(env.events) == 1


def test_case6_power_sum_and_separate_attribution():
    ems = [EmitterSpec(id=i, family="continuous", threat_weight=0.1, params=dict(band=2, snr_db=0.0))
           for i in ("C1", "C2")]
    env = build(spec(ems, n_slots=10))
    assert env.snr_db[:, 2] == pytest.approx(10 * np.log10(2))
    assert env.truth[:, 2].all()
    assert sorted(e.emitter_id for e in env.events) == ["C1", "C2"]


def _bursty(seed):
    return build(spec([EmitterSpec(id="B", family="bursty", threat_weight=0.4,
                                   params=dict(band=1, on_mean_slots=20, off_mean_slots=40,
                                               sigma=0.5, snr_db=15.0))], n_slots=2000, seed=seed))


def test_case7_deterministic():
    a, b = _bursty(1), _bursty(1)
    assert np.array_equal(a.truth, b.truth) and np.array_equal(a.snr_db, b.snr_db)
    assert a.events == b.events


def test_case8_seed_changes_stochastic_emitters():
    assert not np.array_equal(_bursty(1).truth, _bursty(2).truth)


def test_case9_invalid_band_and_duplicate_id():
    bad = EmitterSpec(id="X", family="continuous", threat_weight=0.1, params=dict(band=8, snr_db=5.0))
    with pytest.raises(ValueError):
        build(spec([bad]))
    with pytest.raises(ValueError):
        build(spec([radar(), radar()]))


def test_case10_empty():
    env = build(spec([]))
    assert not env.truth.any() and env.events == []


def test_edge_window_wraps_episode_end():
    e = EmitterSpec(id="R", family="circular_scan", threat_weight=0.6,
                    params=dict(band=1, period=100, beam_slots=5, phase=98, snr_main_db=30.0, sidelobe_db=-40.0))
    env = build(spec([e], n_slots=100))
    assert [(x.start, x.end_exclusive) for x in env.events] == [(0, 3), (98, 100)]


def test_edge_vanish_before_appear_rejected():
    e = EmitterSpec(id="C", family="continuous", threat_weight=0.1, params=dict(band=1, snr_db=5.0),
                    appear_slot=50, vanish_slot=50)
    with pytest.raises(ValueError):
        build(spec([e]))
