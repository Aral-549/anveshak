"""Golden cases for contracts/hit_model.md. Frozen: add cases, never edit."""
import json

import numpy as np
import pytest

from core.belief import Belief, BeliefConfig, Lock, SigTrack
from core.environment import Event
from core.hit_model import HIT_FEATURES, HitModel, assert_train_seeds, hit_features, label_grid


def belief(n_bands=4):
    return Belief(BeliefConfig(n_bands=n_bands, gamma=1.0, period_tol=2, min_period=50, max_period=1000))


def col(name):
    return HIT_FEATURES.index(name)


def test_case1_fresh_belief():
    X = hit_features(0, belief(), T_max=8)
    assert X.shape == (4, len(HIT_FEATURES))
    expect = dict(window=0, soon=0, time_to_window=1.0, lock_period=0, confirm=0, staleness=0.125, novel_mean=0.5,
                  novel_evidence=0.5, band_threat=0.5, occ_mean=0.5, long_here=0, since_novel=12)
    for k, v in expect.items():
        assert X[:, col(k)] == pytest.approx(np.full(4, v)), k


def _locked(b, band, nw, w=0.7):
    b.sigs["A"] = SigTrack(event_starts=[0, 100, 200], bands=[band] * 3, lock=Lock(100, 200, band),
                           next_window=nw, threat_hint=w)


def test_case2_window_now():
    b = belief()
    _locked(b, 2, 300)
    X = hit_features(300, b, T_max=8)
    assert X[2, col("window")] == 1 and X[2, col("window_threat")] == pytest.approx(0.7)
    assert X[2, col("time_to_window")] == pytest.approx(0.0)
    assert X[2, col("lock_period")] == pytest.approx(0.1)
    assert X[[0, 1, 3], col("window")].tolist() == [0, 0, 0]


def test_case3_soon():
    b = belief()
    _locked(b, 2, 310)
    X = hit_features(300, b, T_max=8)
    assert X[2, col("window")] == 0 and X[2, col("soon")] == 1
    assert X[2, col("time_to_window")] == pytest.approx(0.05)


def test_case4_far_window_clipped():
    b = belief()
    _locked(b, 2, 900)
    X = hit_features(300, b, T_max=8)
    assert X[2, col("soon")] == 0 and X[2, col("time_to_window")] == pytest.approx(1.0)


def test_case5_since_novel():
    b = belief()
    b.sigs["A"] = SigTrack(event_starts=[250], bands=[1], threat_hint=0.6)
    X = hit_features(300, b, T_max=8)
    assert X[1, col("since_novel")] == pytest.approx(0.05)
    assert X[[0, 2, 3], col("since_novel")].tolist() == [12, 12, 12]


def test_edge_two_locks_one_band():
    b = belief()
    b.sigs["A"] = SigTrack(event_starts=[0, 100, 200], bands=[2] * 3, lock=Lock(100, 200, 2), next_window=340,
                           threat_hint=0.4)
    b.sigs["B"] = SigTrack(event_starts=[0, 300, 600], bands=[2] * 3, lock=Lock(300, 600, 2), next_window=320,
                           threat_hint=0.9)
    X = hit_features(300, b, T_max=8)
    assert X[2, col("time_to_window")] == pytest.approx(0.1)   # soonest: 320
    assert X[2, col("lock_period")] == pytest.approx(0.3)      # period of the soonest window's signature


def test_case6_label_until_interception():
    y = label_grid([Event("R", 3, 10, 12, 0.6)], n_slots=20, n_bands=4, first_intercept={})
    assert y[10, 3] and y[11, 3] and not y[12, 3] and not y[10, 2]


def test_case7_label_stops_after_interception():
    y = label_grid([Event("R", 3, 10, 12, 0.6)], n_slots=20, n_bands=4, first_intercept={0: 10})
    assert y[10, 3] and not y[11, 3]


def test_case8_benign_excluded():
    y = label_grid([Event("C", 1, 0, 20, 0.1)], n_slots=20, n_bands=4, first_intercept={})
    assert not y.any()


def test_case9_feature_mismatch_rejected(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"learner": {"feature_names": list(reversed(HIT_FEATURES))}}))
    with pytest.raises(ValueError):
        HitModel.load(p)


def test_case10_eval_seeds_rejected_for_training():
    assert_train_seeds([0, 5, 999])
    with pytest.raises(AssertionError):
        assert_train_seeds([0, 1000])


def test_regression_eval_scores_deployed_model():
    """BUGLOG 2026-10-01: the reported ranking must come from the model file the scheduler loads."""
    import os
    if not os.path.exists("artifacts/hit_model.json"):
        pytest.skip("model artifact not built")
    ev = json.load(open("artifacts/hit_model_eval.json"))
    m = HitModel.load("artifacts/hit_model.json")
    assert ev["n_trees_scored"] == m.booster.num_boosted_rounds()
    assert ev["scored_with"] == "HitModel.load(artifacts/hit_model.json).score"
