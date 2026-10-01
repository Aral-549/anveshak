"""Golden cases for contracts/scheduler.md. Frozen: add cases, never edit."""
import numpy as np
import pytest

from core.belief import Belief, BeliefConfig, Lock, SigTrack
from core.scheduler import Scheduler


def belief(n_bands, **kw):
    base = dict(n_bands=n_bands, gamma=1.0, period_tol=1, min_period=50, max_period=1000, unlock_misses=3)
    base.update(kw)
    return Belief(BeliefConfig(**base))


def run(sched, b, slots):
    """Drive the scheduler, marking chosen bands visited (no detections)."""
    out = []
    for t in slots:
        p = sched.plan(t, b)
        out.append(p)
        b.mark_visited(t, p.bands)
    return out


def test_case1_sweep_single():
    b = belief(4)
    plans = run(Scheduler("sweep", n_bands=4, M=1, T_max=8), b, range(8))
    assert [p.bands[0] for p in plans] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all(p.sources == ["SWEEP"] for p in plans)


def test_case2_sweep_two_receivers():
    b = belief(4)
    plans = run(Scheduler("sweep", n_bands=4, M=2, T_max=8), b, range(4))
    assert [p.bands for p in plans] == [[0, 1], [2, 3], [0, 1], [2, 3]]


def test_case3_permuted_sweep_blocks():
    seen_orders = set()
    for seed in range(5):
        b = belief(4)
        plans = run(Scheduler("permuted_sweep", n_bands=4, M=1, T_max=8, seed=seed), b, range(12))
        seq = [p.bands[0] for p in plans]
        for k in range(3):
            assert sorted(seq[4 * k:4 * k + 4]) == [0, 1, 2, 3]
            seen_orders.add(tuple(seq[4 * k:4 * k + 4]))
    assert len(seen_orders) > 1


def test_case4_infeasible_floor():
    with pytest.raises(ValueError):
        Scheduler("sweep", n_bands=4, M=1, T_max=3)


def _lock(b, sig, band, t, w):
    b.sigs[sig] = SigTrack(event_starts=[t - 300, t - 200, t - 100], lock=Lock(period=100, last_start=t - 100, band=band),
                           next_window=t, threat_hint=w)


def _fresh(b, t):
    b.last_visit[:] = t - 1


def test_case5_predicted_window():
    b = belief(8)
    _fresh(b, 500)
    _lock(b, "A", 7, 500, 0.6)
    p = Scheduler("predictive", n_bands=8, M=1, T_max=64).plan(500, b)
    assert p.bands == [7] and p.sources == ["PREDICTED_WINDOW"]


def test_case6_conflict_threat_wins():
    b = belief(8)
    _fresh(b, 500)
    _lock(b, "A", 2, 500, 0.6)
    _lock(b, "B", 5, 500, 1.0)
    p = Scheduler("predictive", n_bands=8, M=1, T_max=64).plan(500, b)
    assert p.bands == [5]
    assert 2 in p.conflicts_dropped


def test_case7_floor_overrides_policy():
    b = belief(8)
    _fresh(b, 500)
    b.last_visit[2] = 500 - 64
    _lock(b, "A", 7, 500, 1.0)
    p = Scheduler("predictive", n_bands=8, M=1, T_max=64).plan(500, b)
    assert p.bands == [2] and p.sources == ["REVISIT_FLOOR"]


def test_case8_floor_property_degenerate_policy():
    b = belief(4)
    s = Scheduler("custom", n_bands=4, M=1, T_max=4, score_fn=lambda t, bel: np.array([1.0, 0, 0, 0]))
    for t in range(1000):
        assert b.staleness(t).max() <= 4
        p = s.plan(t, b)
        b.mark_visited(t, p.bands)


def test_case9_bandit_learns():
    b = belief(4)
    b.alpha_novel[:] = 1.0
    b.beta_novel[:] = 21.0
    b.alpha_novel[1], b.beta_novel[1] = 21.0, 1.0
    s = Scheduler("bandit", n_bands=4, M=1, T_max=10**6, seed=3)
    picks = [s.plan(t, b).bands[0] for t in range(1000)]
    assert picks.count(1) >= 950


def test_case12_bandit_ignores_continuous_benign():
    b = belief(2)
    from tests.golden.test_belief_golden import obs
    for t in range(200):
        band = t % 2
        dets = [("C", 0.1)] if band == 0 else ([("R", 0.6)] if t % 10 == 1 else [])
        b.update(obs(t, {band: dets}))
    s = Scheduler("bandit", n_bands=2, M=1, T_max=10**6, seed=0)
    picks = [s.plan(1000 + t, b).bands[0] for t in range(500)]
    assert picks.count(1) > picks.count(0)


def test_case10_oracle():
    b = belief(4)
    p = Scheduler("oracle", n_bands=4, M=1, T_max=64).plan(0, b, oracle_scores=np.array([0, 0.4, 0, 1.0]))
    assert p.bands == [3] and p.sources == ["ORACLE"]


@pytest.mark.parametrize("policy", ["sweep", "permuted_sweep", "bandit", "predictive"])
@pytest.mark.parametrize("M", [1, 2, 3])
def test_case11_output_well_formed(policy, M):
    rng = np.random.default_rng(0)
    b = belief(8)
    s = Scheduler(policy, n_bands=8, M=M, T_max=32, seed=1)
    for t in range(2000):
        b.last_visit[:] = t - rng.integers(1, 40, size=8)
        b.alpha_novel[:] = rng.uniform(0.5, 20, 8)
        b.beta_novel[:] = rng.uniform(0.5, 20, 8)
        p = s.plan(t, b)
        assert len(p.bands) == M == len(set(p.bands)) == len(p.sources)
        assert all(0 <= x < 8 for x in p.bands)


def test_edge_all_due_no_deadlock():
    b = belief(6)
    s = Scheduler("bandit", n_bands=6, M=2, T_max=3)
    p = s.plan(10, b)
    assert p.bands == [0, 1] and p.sources == ["REVISIT_FLOOR", "REVISIT_FLOOR"]


def test_edge_M_greater_than_B():
    with pytest.raises(ValueError):
        Scheduler("sweep", n_bands=2, M=3, T_max=8)


def test_case13_confirm_camp():
    b = belief(8)
    _fresh(b, 500)
    b.sigs["A"] = SigTrack(event_starts=[450], threat_hint=0.6, confirm_band=3, confirm_until=1500)
    p = Scheduler("predictive", n_bands=8, M=1, T_max=64).plan(500, b)
    assert p.bands == [3] and p.sources == ["CONFIRM"]


def test_case14_window_outranks_confirm():
    b = belief(8)
    _fresh(b, 500)
    b.sigs["A"] = SigTrack(event_starts=[450], threat_hint=1.0, confirm_band=3, confirm_until=1500)
    _lock(b, "B", 5, 500, 0.1)
    p = Scheduler("predictive", n_bands=8, M=1, T_max=64).plan(500, b)
    assert p.bands == [5]


def test_case15_hybrid_keeps_window_tier():
    b = belief(8)
    _fresh(b, 500)
    _lock(b, "A", 5, 500, 0.2)
    fn = lambda t, bel: np.array([0, 0.99, 0, 0, 0, 0, 0, 0])
    p = Scheduler("hybrid", n_bands=8, M=1, T_max=64, score_fn=fn).plan(500, b)
    assert p.bands == [5] and p.sources == ["PREDICTED_WINDOW"]


def test_case16_hybrid_model_decides_search():
    b = belief(4)
    _fresh(b, 500)
    fn = lambda t, bel: np.array([0.1, 0.9, 0.2, 0.3])
    p = Scheduler("hybrid", n_bands=4, M=1, T_max=64, score_fn=fn).plan(500, b)
    assert p.bands == [1] and p.sources == ["POLICY"]


def test_case17_hybrid_needs_score_fn():
    with pytest.raises(ValueError):
        Scheduler("hybrid", n_bands=4, M=1, T_max=64)


def test_case18_hybrid_staleness_term():
    b = belief(4)
    _fresh(b, 500)
    b.last_visit[0] = 460
    fn = lambda t, bel: np.array([0.5, 0.6, 0.0, 0.0])
    p = Scheduler("hybrid", n_bands=4, M=1, T_max=64, score_fn=fn, hybrid_lam=1.0).plan(500, b)
    assert p.bands == [0]
