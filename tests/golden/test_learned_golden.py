"""Golden cases for contracts/learned_policy.md. Frozen: add cases, never edit."""
import numpy as np
import pytest

from core.belief import Belief, BeliefConfig, Lock, SigTrack
from core.learned import FEATURES, LearnedScorer, features, train
from core.scheduler import Scheduler


def belief(n_bands=4, tol=2):
    return Belief(BeliefConfig(n_bands=n_bands, gamma=1.0, period_tol=tol, min_period=50, max_period=1000))


def col(name):
    return FEATURES.index(name)


def test_case1_fresh_belief():
    X = features(0, belief(), T_max=8, rng=np.random.default_rng(0))
    assert X.shape == (4, len(FEATURES))
    expect = dict(window=0, confirm=0, staleness=1 / 8, novel_mean=0.5, novel_evidence=0.5,
                  band_threat=0.5, occ_mean=0.5, long_here=0, soon=0)
    for k, v in expect.items():
        assert X[:, col(k)] == pytest.approx(np.full(4, v)), k


def _locked(b, band, nw, w):
    b.sigs["A"] = SigTrack(event_starts=[0, 100, 200], lock=Lock(100, 200, band), next_window=nw, threat_hint=w)


def test_case2_window():
    b = belief()
    _locked(b, 2, 300, 0.7)
    X = features(300, b, T_max=8, rng=np.random.default_rng(0))
    assert X[:, col("window")].tolist() == [0, 0, 1, 0]
    assert X[2, col("window_threat")] == pytest.approx(0.7)


def test_case3_soon():
    b = belief()
    _locked(b, 2, 310, 0.7)
    X = features(300, b, T_max=8, rng=np.random.default_rng(0))
    assert X[2, col("window")] == 0 and X[2, col("soon")] == 1


def test_case4_confirm():
    b = belief()
    b.sigs["A"] = SigTrack(event_starts=[250], threat_hint=0.6, confirm_band=1, confirm_until=900)
    X = features(300, b, T_max=8, rng=np.random.default_rng(0))
    assert X[1, col("confirm")] == 1 and X[1, col("confirm_threat")] == pytest.approx(0.6)
    assert X[0, col("confirm")] == 0


def test_case5_staleness_clipped():
    X = features(999, belief(), T_max=128, rng=np.random.default_rng(0))
    assert X[:, col("staleness")] == pytest.approx(np.full(4, 2.0))


def _zero_scorer():
    return LearnedScorer(W1=np.zeros((16, len(FEATURES))), b1=np.zeros(16), W2=np.zeros(16), b2=0.0)


def test_case6_zero_weights_lowest_index():
    b = belief()
    s = _zero_scorer()
    sched = Scheduler("learned", n_bands=4, M=1, T_max=8, score_fn=s.score_fn(T_max=8, rng=np.random.default_rng(0)))
    assert sched.plan(0, b).bands == [0]


def test_case7_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    s = LearnedScorer(W1=rng.normal(size=(16, len(FEATURES))), b1=rng.normal(size=16),
                      W2=rng.normal(size=16), b2=0.3)
    p = tmp_path / "w.json"
    s.save(p)
    X = rng.normal(size=(7, len(FEATURES)))
    assert np.allclose(LearnedScorer.load(p).score(X), s.score(X))


def test_case8_feature_mismatch_rejected(tmp_path):
    import json
    s = _zero_scorer()
    p = tmp_path / "w.json"
    s.save(p)
    d = json.loads(p.read_text())
    d["features"] = list(reversed(d["features"]))
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        LearnedScorer.load(p)


def test_case9_training_reproducible():
    kw = dict(generations=2, population=4, elite=2, scenarios_per_suite=1, n_slots=300, workers=1, seed=0)
    a, b = train(**kw), train(**kw)
    assert a["fitness_curve"] == b["fitness_curve"]


def test_edge_training_seeds_disjoint_from_eval():
    from core import scenarios
    from core.learned import TRAIN_SEEDS
    assert not set(TRAIN_SEEDS) & set(scenarios.EVAL_SEEDS)
