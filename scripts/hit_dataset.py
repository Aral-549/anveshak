"""Build hit-model datasets from simulator missions (contracts/hit_model.md).

    python scripts/hit_dataset.py train   # seeds 0-59 per suite  -> artifacts/hit_data/train.npz
    python scripts/hit_dataset.py val     # seeds 60-79 per suite -> artifacts/hit_data/val.npz
    python scripts/hit_dataset.py eval    # seeds 1000-1019, predictive, no subsampling -> artifacts/hit_data/eval.npz
"""
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import scenarios  # noqa: E402
from core.belief import Belief, BeliefConfig  # noqa: E402
from core.environment import build  # noqa: E402
from core.hit_model import assert_train_seeds, hit_features, label_grid  # noqa: E402
from core.log import boundary, configure, get_logger  # noqa: E402
from core.receiver import Receiver  # noqa: E402
from core.scheduler import Scheduler  # noqa: E402

log = get_logger("hit_dataset")
T_MAX, PFA, NOISE, LAM = 128, 1e-4, 0.05, 0.1
SPLITS = {
    "train": dict(seeds=range(0, 60), stride=4, neg_keep=0.05, behaviour="mix"),
    "val": dict(seeds=range(60, 80), stride=4, neg_keep=0.05, behaviour="mix"),
    "eval": dict(seeds=range(1000, 1020), stride=8, neg_keep=1.0, behaviour="predictive"),
}


def behaviour_for(seed: int, mode: str) -> str:
    if mode != "mix":
        return mode
    return ["predictive", "predictive", "permuted_sweep", "bandit"][seed % 4]


def mission(job):
    suite, seed, stride, neg_keep, mode = job
    spec = scenarios.make(suite, seed)
    env = build(spec)
    B, T = spec.n_bands, spec.n_slots
    policy = behaviour_for(seed, mode)
    rng = np.random.default_rng([seed, 31, scenarios.SUITES.index(suite)])
    rx = Receiver(B, 1, PFA, NOISE, np.random.default_rng([spec.seed, seed, 7]))
    belief = Belief(BeliefConfig(n_bands=B))
    sched = Scheduler(policy, B, 1, T_MAX, seed=seed, lam=LAM)
    emitter_index = {eid: i for i, eid in enumerate(env.emitter_ids)}
    offset = int(rng.integers(0, stride))

    raw_hits, last_hit = np.zeros(B), np.full(B, -1)
    first_intercept: dict[int, int] = {}
    feats, legacy, slots = [], [], []
    for t in range(T):
        if (t - offset) % stride == 0:
            X = hit_features(t, belief, T_MAX)
            assert np.isfinite(X).all(), f"non-finite feature at {suite}/{seed}/{t}"
            vis = belief.visits.astype(float)
            tsec = t * spec.slot_ms / 1000
            L = np.column_stack([
                np.arange(B), 2000 + 500 * np.arange(B) + 250, np.full(B, spec.slot_ms / 1000), np.full(B, tsec),
                vis, raw_hits, np.where(vis > 0, raw_hits / np.maximum(vis, 1), 0.0),
                vis / max(vis.sum(), 1.0), belief.staleness(t) * spec.slot_ms / 1000,
                np.where(last_hit >= 0, (t - last_hit) * spec.slot_ms / 1000, tsec)])
            feats.append(X)
            legacy.append(L)
            slots.append(t)
        plan = sched.plan(t, belief)
        obs = rx.observe(t, plan.bands, env.band_emitters(t, plan.bands))
        belief.update(obs)
        for b, bo in obs.per_band.items():
            if bo.detected:
                raw_hits[b] += 1
                last_hit[b] = t
            for d in bo.detections:
                k = int(env.event_of[emitter_index[d.signature], t])
                first_intercept.setdefault(k, t)

    y_grid = label_grid(env.events, T, B, first_intercept)
    X = np.concatenate(feats)
    Lg = np.concatenate(legacy)
    y = np.concatenate([y_grid[t] for t in slots])
    keep = y | (rng.random(len(y)) < neg_keep)
    w = np.where(y, 1.0, 1.0 / neg_keep)[keep]
    boundary(log, logging.INFO, "mission", suite=suite, seed=seed, policy=policy, rows=int(keep.sum()),
             positives=int(y.sum()))
    return (X[keep].astype(np.float32), y[keep], w.astype(np.float32), Lg[keep].astype(np.float32),
            np.full(keep.sum(), scenarios.SUITES.index(suite), np.int8), np.full(keep.sum(), seed, np.int32))


def main():
    split = sys.argv[1]
    cfg = SPLITS[split]
    if split != "eval":
        assert_train_seeds(cfg["seeds"])
    configure(logging.WARNING)
    jobs = [(s, seed, cfg["stride"], cfg["neg_keep"], cfg["behaviour"]) for s in scenarios.SUITES for seed in cfg["seeds"]]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        parts = list(ex.map(mission, jobs, chunksize=1))
    X, y, w, L, suite, seed = (np.concatenate(p) for p in zip(*parts))
    os.makedirs("artifacts/hit_data", exist_ok=True)
    np.savez_compressed(f"artifacts/hit_data/{split}.npz", X=X, y=y, w=w, legacy=L, suite=suite, seed=seed)
    print(f"{split}: rows {len(y)}, positives {int(y.sum())}, weighted positive rate {(w * y).sum() / w.sum():.5f}")


if __name__ == "__main__":
    main()
