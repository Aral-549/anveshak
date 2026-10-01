"""Golden cases for contracts/metrics.md. Frozen: add cases, never edit."""
import numpy as np
import pytest

from core.environment import Event
from core.metrics import compute
from core.receiver import BandObservation, Detection, Observation


def fixture_F():
    events = [Event("A", 0, 0, 2, 1.0), Event("B", 1, 2, 3, 0.5), Event("A", 0, 5, 6, 1.0)]
    truth = np.zeros((6, 2), dtype=bool)
    truth[0:2, 0] = True
    truth[2, 1] = True
    truth[5, 0] = True
    dwells = [[0], [0], [0], [1], [1], [0]]
    observations = []
    for t, (b,) in enumerate(dwells):
        dets = [Detection("A", 1.0, 20.0)] if t in (0, 1, 5) else []
        observations.append(Observation(t, {b: BandObservation(bool(dets), dets, False)}))
    return dict(events=events, truth=truth, dwells=dwells, observations=observations,
                predictions=[], emitter_appear={"A": 0, "B": 0}, slot_ms=10.0)


def test_case1_interception_ratio():
    r = compute(**fixture_F())
    assert r.interception_ratio == pytest.approx(2 / 3)
    assert r.weighted_interception_ratio == pytest.approx(0.8)


def test_case2_pd_pfa():
    r = compute(**fixture_F())
    assert r.pd_empirical == 1.0 and r.pfa_empirical == 0.0


def test_case3_ttfi_and_censoring():
    r = compute(**fixture_F())
    assert r.ttfi_slots == {"A": 0}
    assert r.censored == ["B"]
    assert r.mean_ttfi_slots == 0


def test_case4_reward():
    r = compute(**fixture_F())
    assert r.reward_total == 2.0
    assert r.reward_per_slot == pytest.approx(1 / 3)


def test_case5_intercept_rate():
    assert compute(**fixture_F()).intercept_rate_per_s == pytest.approx(33.333, rel=1e-3)


def test_case6_max_revisit():
    assert compute(**fixture_F()).max_revisit_slots == [3, 4]


def _pred_fixture(preds):
    f = fixture_F()
    f["events"] = [Event("A", 0, 802, 804, 1.0), Event("A", 0, 905, 907, 1.0)]
    f["predictions"] = preds
    f["pred_tol"] = 2
    return f


def test_case7_prediction_scoring():
    r = compute(**_pred_fixture([(0, "A", 800), (0, "A", 900)]))
    assert r.pred_correct_pct == 50.0
    assert r.mean_abs_intercept_time_error_slots == 3.5
    assert r.mean_abs_intercept_time_error_ms == 35.0


def test_case8_unmatched_prediction():
    r = compute(**_pred_fixture([(1000, "A", 1100)]))
    assert len(r.pred_unmatched) == 1
    assert r.pred_correct_pct is None


def test_case9_wrong_signature_does_not_intercept():
    f = fixture_F()
    f["observations"][5] = Observation(5, {0: BandObservation(True, [Detection("Z", 1.0, 20.0)], False)})
    r = compute(**f)
    assert r.interception_ratio == pytest.approx(1 / 3)


def test_case10_zero_events():
    f = fixture_F()
    f["events"] = []
    r = compute(**f)
    assert r.interception_ratio is None
    assert "interception_ratio" in r.undefined


def test_edge_two_receivers_same_event_counted_once():
    events = [Event("A", 0, 0, 1, 1.0)]
    truth = np.zeros((1, 1), dtype=bool)
    truth[0, 0] = True
    o = Observation(0, {0: BandObservation(True, [Detection("A", 1.0, 20.0)], False)})
    r = compute(events=events, truth=truth, dwells=[[0]], observations=[o], predictions=[],
                emitter_appear={"A": 0}, slot_ms=10.0)
    assert r.reward_total == 1.0 and r.interception_ratio == 1.0


def test_edge_detection_after_event_end():
    f = fixture_F()
    f["observations"][2] = Observation(2, {0: BandObservation(True, [Detection("A", 1.0, 20.0)], False)})
    r = compute(**f)
    assert r.interception_ratio == pytest.approx(2 / 3)
