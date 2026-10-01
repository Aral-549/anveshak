"""Benchmark every policy on every suite over held-out seeds; write artifacts/benchmark.json.

Every figure shown anywhere (README, console, deck) must come from this artifact.

    python scripts/benchmark.py                       # full: 5 suites x 100 eval seeds
    python scripts/benchmark.py --seeds 0-29 --out artifacts/tune.json --lam 0.05   # tuning (train seeds only)
"""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scenarios  # noqa: E402
from core.log import configure  # noqa: E402
from core.receiver import snr_for_pd  # noqa: E402
from core.sim import RunConfig, run  # noqa: E402

POLICIES = ["sweep", "permuted_sweep", "bandit", "predictive", "oracle"]


def _one(job):
    suite, seed, policies, M, T_max, lam, hybrid_lam = job
    spec = scenarios.make(suite, seed)
    out = {}
    oracle_w = None
    for pol in sorted(policies, key=lambda p: p != "oracle"):  # oracle first: others are scored against it
        score_fn = None
        if pol in ("learned", "hybrid"):
            from core.hit_model import HitModel
            score_fn = HitModel.load("artifacts/hit_model.json").score_fn(T_max)
        r = run(spec, RunConfig(policy=pol, M=M, T_max=T_max, lam=lam, seed=seed, hybrid_lam=hybrid_lam),
                oracle_weighted_intercepts=oracle_w, score_fn=score_fn)
        rep = r.report
        if pol == "oracle":
            oracle_w = rep.reward_total
        popups = [e.id for e in spec.emitters if e.id.startswith("POPUP")]
        out[pol] = dict(
            ir=rep.interception_ratio, wir=rep.weighted_interception_ratio,
            pd=rep.pd_empirical, pfa=rep.pfa_empirical, rate=rep.intercept_rate_per_s,
            mean_ttfi=rep.mean_ttfi_slots, n_censored=len(rep.censored), n_emitters=len(spec.emitters),
            reward_per_slot=rep.reward_per_slot, pred_correct=rep.pred_correct_pct,
            pred_err=rep.mean_abs_intercept_time_error_slots, pred_n=rep.pred_matched,
            oracle_eff=rep.oracle_efficiency, max_revisit=max(rep.max_revisit_slots),
            popup_ttfi=[rep.ttfi_slots.get(p) for p in popups],
            sources=r.source_counts,
        )
    return suite, seed, out


def _ci(x, rng, n=2000):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                n=int(len(x)))


def aggregate(rows, policies, slot_ms):
    rng = np.random.default_rng(0)
    result = {}
    for suite in sorted({s for s, _, _ in rows}):
        rs = sorted((r for r in rows if r[0] == suite), key=lambda r: r[1])
        per = {}
        for pol in policies:
            vals = [r[2][pol] for r in rs]
            agg = {k: _ci([v[k] for v in vals], rng) for k in
                   ("ir", "wir", "pd", "pfa", "rate", "mean_ttfi", "reward_per_slot", "pred_correct", "pred_err",
                    "oracle_eff")}
            agg["max_revisit_worst"] = int(max(v["max_revisit"] for v in vals))
            agg["censored_share"] = float(sum(v["n_censored"] for v in vals) / sum(v["n_emitters"] for v in vals))
            pops = [t for v in vals for t in v["popup_ttfi"]]
            if pops:
                hit = [t for t in pops if t is not None]
                agg["popup"] = dict(n=len(pops), detected_share=len(hit) / len(pops),
                                    median_ttfi_s=float(np.median(hit)) * slot_ms / 1000 if hit else None,
                                    mean_ttfi_s=float(np.mean(hit)) * slot_ms / 1000 if hit else None)
            src = {}
            for v in vals:
                for k, c in v["sources"].items():
                    src[k] = src.get(k, 0) + c
            tot = sum(src.values())
            agg["dwell_sources"] = {k: round(c / tot, 4) for k, c in sorted(src.items())}
            if pol != "sweep":
                a = [r[2][pol]["wir"] for r in rs]
                b = [r[2]["sweep"]["wir"] for r in rs]
                diff = np.subtract(a, b)
                if len(a) >= 6 and np.any(diff != 0):
                    stat = wilcoxon(a, b)
                    agg["wilcoxon_vs_sweep_wir"] = dict(p=float(stat.pvalue), wins=int((diff > 0).sum()),
                                                        losses=int((diff < 0).sum()), n=len(a))
            per[pol] = agg
        result[suite] = per
    return result


def parse_seeds(s: str):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=f"{scenarios.EVAL_SEEDS.start}-{scenarios.EVAL_SEEDS.stop - 1}")
    ap.add_argument("--suites", default=",".join(scenarios.SUITES))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--T_max", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--hybrid_lam", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="artifacts/benchmark.json")
    args = ap.parse_args()
    configure(logging.WARNING)

    seeds, suites, policies = parse_seeds(args.seeds), args.suites.split(","), args.policies.split(",")
    if "sweep" not in policies:
        policies.insert(0, "sweep")
    jobs = [(s, seed, policies, args.M, args.T_max, args.lam, args.hybrid_lam) for s in suites for seed in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=2))
    artifact = dict(
        generated_by="scripts/benchmark.py", argv=sys.argv[1:], seconds=round(time.time() - t0, 1),
        config=dict(n_bands=scenarios.N_BANDS, n_slots=scenarios.N_SLOTS, slot_ms=scenarios.SLOT_MS,
                    M=args.M, T_max=args.T_max, lam=args.lam, hybrid_lam=args.hybrid_lam, pfa=RunConfig("x").pfa, seeds=[seeds[0], seeds[-1]]),
        sensitivity_db_pd90=snr_for_pd(0.9, RunConfig("x").pfa),
        suites=aggregate(rows, policies, scenarios.SLOT_MS),
        per_seed=[dict(suite=s, seed=seed, **{p: {k: v for k, v in d.items() if k != "sources"} for p, d in out.items()})
                  for s, seed, out in rows],
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=1)
    for suite, per in artifact["suites"].items():
        line = " | ".join(f"{p} {per[p]['wir']['mean']:.3f}" for p in policies)
        print(f"{suite}: wIR {line}")
    print(f"wrote {args.out} in {artifact['seconds']} s")


if __name__ == "__main__":
    main()
