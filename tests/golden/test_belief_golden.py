"""Golden cases for contracts/belief.md. Frozen: add cases, never edit."""
import pytest

from core.belief import Belief, BeliefConfig, find_period
from core.receiver import BandObservation, Detection, Observation


def cfg(**kw):
    base = dict(n_bands=8, gamma=1.0, period_tol=1, min_period=50, max_period=1000, unlock_misses=3)
    base.update(kw)
    return BeliefConfig(**base)


def obs(t, visits):
    """visits: {band: list of (signature, threat) detections} or {band: 'FA'}."""
    per = {}
    for b, d in visits.items():
        if d == "FA":
            per[b] = BandObservation(detected=True, detections=[], false_alarm=True)
        else:
            dets = [Detection(signature=s, threat_hint=w, snr_db=20.0) for s, w in d]
            per[b] = BandObservation(detected=bool(dets), detections=dets, false_alarm=False)
    return Observation(slot=t, per_band=per)


def hit(t, band=3, sig="A"):
    return obs(t, {band: [(sig, 0.6)]})


def miss(t, band=3):
    return obs(t, {band: []})


def test_case1_occupancy_no_discount():
    b = Belief(cfg())
    for o in (hit(0), hit(1), hit(2), miss(3)):
        b.update(o)
    assert (b.alpha[3], b.beta[3]) == (4.0, 2.0)
    assert b.occ_mean[3] == pytest.approx(2 / 3)


def test_case2_occupancy_discounted():
    b = Belief(cfg(gamma=0.5))
    b.update(hit(0))
    b.update(miss(1))
    assert (b.alpha[3], b.beta[3]) == pytest.approx((0.75, 1.25))
    assert b.occ_mean[3] == pytest.approx(0.375)


def test_case3_never_visited():
    b = Belief(cfg())
    for t in range(10):
        b.update(miss(t, band=0))
    assert b.staleness(9)[5] == 10
    assert b.occ_mean[5] == 0.5


def test_case4_period_from_multiples():
    assert find_period([0, 300, 700], tol=1, min_period=50, max_period=1000) == 100


def test_case4_prediction_through_updates():
    b = Belief(cfg())
    for t in range(0, 702):
        b.update(hit(t, band=4) if t in (0, 300, 700) else miss(t, band=4))
    tr = b.sigs["A"]
    assert tr.lock is not None and tr.lock.period == 100
    assert tr.next_window == 800


def test_case5_one_interval_no_lock():
    assert find_period([0, 300], tol=1, min_period=50, max_period=1000) is None


def test_case6_jitter_within_tolerance():
    assert find_period([0, 301, 700], tol=1, min_period=50, max_period=1000) == 100


def test_case7_largest_consistent_period():
    assert find_period([0, 200, 400, 600], tol=1, min_period=50, max_period=1000) == 200


def test_case8_random_intervals_no_lock():
    assert find_period([0, 137, 291, 503], tol=1, min_period=50, max_period=1000) is None


def test_case9_below_min_period():
    assert find_period([0, 5, 10], tol=1, min_period=50, max_period=1000) is None


def test_case10_contiguous_detections_one_event():
    b = Belief(cfg())
    for t in range(10, 21):
        b.update(hit(t, band=4) if t in (10, 11, 12, 20) else miss(t, band=4))
    assert b.sigs["A"].event_starts == [10, 20]


def _locked_belief():
    b = Belief(cfg())
    for t in range(0, 702):
        if t in (0, 300, 700):
            b.update(hit(t, band=4))
        elif t in (1, 301):
            b.update(miss(t, band=4))  # an empty look between events, so each hit is novel
        else:
            b.update(miss(t, band=0))
    assert b.sigs["A"].lock is not None and b.sigs["A"].next_window == 800
    return b


def test_case11_unlock_after_k_watched_misses():
    b = _locked_belief()
    for t in range(702, 1003):
        watched = any(abs(t - w) <= 1 for w in (800, 900, 1000))
        b.update(miss(t, band=4) if watched else miss(t, band=0))
        if t == 902:
            assert b.sigs["A"].lock is not None and b.sigs["A"].consecutive_misses == 2
    assert b.sigs["A"].lock is None


def test_case12_unwatched_window_is_not_a_miss():
    b = _locked_belief()
    for t in range(702, 1003):
        b.update(miss(t, band=0))
    assert b.sigs["A"].consecutive_misses == 0
    assert b.sigs["A"].lock is not None


def test_case13_false_alarm():
    b = Belief(cfg())
    b.update(obs(0, {6: "FA"}))
    assert b.alpha[6] == 2.0
    assert b.sigs == {}


def test_case14_continuous_never_locks():
    b = Belief(cfg())
    for t in range(0, 97):
        b.update(hit(t, band=4) if t % 32 == 0 else miss(t, band=0))
    assert b.sigs["A"].event_starts == [0]
    assert b.sigs["A"].lock is None


def test_case15_miss_breaks_run():
    b = Belief(cfg())
    for t in range(0, 65):
        if t in (0, 64):
            b.update(hit(t, band=4))
        elif t == 32:
            b.update(miss(t, band=4))
        else:
            b.update(miss(t, band=0))
    assert b.sigs["A"].event_starts == [0, 64]


def test_case16_novel_occupancy():
    b = Belief(cfg())
    b.update(hit(0, band=2))
    b.update(hit(1, band=2))
    b.update(miss(2, band=2))
    assert (b.alpha_novel[2], b.beta_novel[2]) == (2.0, 3.0)


def test_case17_confirming_after_first_hit():
    b = Belief(cfg())
    for t in range(0, 101):
        b.update(hit(t, band=4) if t == 100 else miss(t, band=0))
    tr = b.sigs["A"]
    assert (tr.confirm_band, tr.confirm_until) == (4, 1101)


def test_case18_camp_gives_provisional_lock():
    b = Belief(cfg())
    for t in range(0, 402):
        if t < 100:
            b.update(miss(t, band=0))
        else:
            b.update(hit(t, band=4) if t in (100, 400) else miss(t, band=4))
    tr = b.sigs["A"]
    assert tr.lock is not None and tr.lock.period == 300
    assert tr.next_window == 700
    assert tr.confirm_until is None


def test_case19_no_camp_no_provisional_lock():
    b = Belief(cfg())
    for t in range(0, 402):
        if t in (100, 400):
            b.update(hit(t, band=4))
        elif t == 101:
            b.update(miss(t, band=4))
        else:
            b.update(miss(t, band=0))
    assert b.sigs["A"].lock is None


def test_case20_long_emitter():
    b = Belief(cfg())
    for t in range(0, 21):
        b.update(hit(t, band=4))
    tr = b.sigs["A"]
    assert tr.long is True and tr.confirm_until is None
    b.update(miss(21, band=4))
    b.update(hit(22, band=4))
    assert tr.event_starts == [0, 22]
    assert tr.confirm_until is None


def test_case21_multiband_never_confirms():
    b = Belief(cfg())
    for t in range(0, 101):
        if t == 0:
            b.update(hit(t, band=4))
        elif t == 100:
            b.update(hit(t, band=5))
        else:
            b.update(miss(t, band=0))
    tr = b.sigs["A"]
    assert tr.multiband is True and tr.confirm_until is None


def test_case22_aperiodic_cap():
    b = Belief(cfg())
    for t in range(0, 6001):
        if t in (0, 1999, 4712, 6000):
            b.update(hit(t, band=4))
        elif t in (1, 2000, 4713):
            b.update(miss(t, band=4))
        else:
            b.update(miss(t, band=0))
        if t in (0, 1999, 4712):
            assert b.sigs["A"].confirm_until is not None
    assert b.sigs["A"].confirm_until is None
