"""Closed-loop run: environment -> receiver -> belief -> scheduler, then metrics."""
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from core.belief import Belief, BeliefConfig
from core.environment import Environment, ScenarioSpec, build
from core.log import boundary, get_logger
from core.metrics import FigureOfMeritReport, compute
from core.receiver import Receiver
from core.scheduler import Scheduler

log = get_logger("sim")


@dataclass
class RunConfig:
    policy: str
    M: int = 1
    T_max: int = 128
    pfa: float = 1e-4
    threat_hint_noise: float = 0.05
    lam: float = 0.5
    hybrid_lam: float = 0.0
    seed: int = 0
    pred_tol: int = 2


@dataclass
class RunResult:
    env: Environment
    dwells: list
    sources: list
    observations: list
    belief: Belief
    report: FigureOfMeritReport

    @property
    def source_counts(self) -> dict:
        return dict(Counter(s for ss in self.sources for s in ss))


def run(spec: ScenarioSpec, cfg: RunConfig, belief_cfg: BeliefConfig | None = None, score_fn=None,
        oracle_weighted_intercepts: float | None = None, env: Environment | None = None) -> RunResult:
    env = env or build(spec)
    B, T = spec.n_bands, spec.n_slots
    rx = Receiver(B, cfg.M, cfg.pfa, cfg.threat_hint_noise, np.random.default_rng([spec.seed, cfg.seed, 7]))
    belief = Belief(belief_cfg or BeliefConfig(n_bands=B))
    sched = Scheduler(cfg.policy, B, cfg.M, cfg.T_max, seed=cfg.seed, lam=cfg.lam, score_fn=score_fn,
                      hybrid_lam=cfg.hybrid_lam)
    is_oracle = cfg.policy == "oracle"
    intercepted: set = set()
    emitter_index = {eid: i for i, eid in enumerate(env.emitter_ids)}

    dwells, sources, observations = [], [], []
    for t in range(T):
        plan = sched.plan(t, belief, env.oracle_scores(t, intercepted) if is_oracle else None)
        obs = rx.observe(t, plan.bands, env.band_emitters(t, plan.bands))
        belief.update(obs)
        if is_oracle:
            for bo in obs.per_band.values():
                for d in bo.detections:
                    intercepted.add(int(env.event_of[emitter_index[d.signature], t]))
        dwells.append(plan.bands)
        sources.append(plan.sources)
        observations.append(obs)

    report = compute(env.events, env.truth, dwells, observations, belief.predictions,
                     {e.id: e.appear_slot for e in spec.emitters}, spec.slot_ms, cfg.pred_tol,
                     oracle_weighted_intercepts)
    boundary(log, logging.INFO, "run_done", seed=spec.seed, policy=cfg.policy, M=cfg.M,
             interception_ratio=report.interception_ratio,
             weighted=report.weighted_interception_ratio, sources=dict(Counter(s for ss in sources for s in ss)))
    return RunResult(env, dwells, sources, observations, belief, report)
