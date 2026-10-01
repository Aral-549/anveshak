"""Slide-2 illustration: one simulated S1 mission, open-loop sweep vs ANVESHAK, same seed."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from core import scenarios  # noqa: E402
from core.sim import RunConfig, run  # noqa: E402

SEED, SUITE, WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3, "S1", 600
spec = scenarios.make(SUITE, SEED)
from core.hit_model import HitModel  # noqa: E402
_fn = HitModel.load("artifacts/hit_model.json").score_fn(128)
runs = {"sweep": run(spec, RunConfig(policy="sweep", seed=SEED)),
        "predictive": run(spec, RunConfig(policy="hybrid", lam=0.1, hybrid_lam=0.1, seed=SEED), score_fn=_fn)}
env = runs["sweep"].env


def hits(r, a, b):
    return [(t, bnd) for t in range(a, b) for bnd, bo in r.observations[t].per_band.items() if bo.detections]


# window late in the mission where ANVESHAK has locked on
_thr = np.zeros(env.truth.shape)
for _e in env.events:
    _thr[_e.start:_e.end_exclusive, _e.band] = np.maximum(_thr[_e.start:_e.end_exclusive, _e.band], _e.threat_weight)


def radar_hits(r, a, b):
    return sum(1 for t, bnd in hits(r, a, b) if _thr[t, bnd] >= 0.5)


best = max(range(6000, spec.n_slots - WIN, 50),
           key=lambda a: radar_hits(runs["predictive"], a, a + WIN) - radar_hits(runs["sweep"], a, a + WIN))
a, b = best, best + WIN

NAVY, BLUE, ORANGE, GREY = "#1F3864", "#0070C0", "#E67E22", "#B8C4D6"
fig, axes = plt.subplots(2, 1, figsize=(4.35, 3.95), dpi=300, sharex=True)
for ax, (pol, label) in zip(axes, [("sweep", "Open-loop sweep"), ("predictive", "ANVESHAK")]):
    r = runs[pol]
    ev_t, ev_b = np.nonzero(env.truth[a:b])
    thr = np.zeros_like(env.truth, dtype=float)
    for e in env.events:
        thr[e.start:e.end_exclusive, e.band] = np.maximum(thr[e.start:e.end_exclusive, e.band], e.threat_weight)
    hot = thr[a:b][ev_t, ev_b] >= 0.5
    t_s = (np.arange(a, b)) * spec.slot_ms / 1000
    ax.scatter(t_s[ev_t[~hot]], ev_b[~hot], s=1.2, marker="s", color=GREY, lw=0, label="benign emitter")
    ax.scatter(t_s[ev_t[hot]], ev_b[hot], s=9, marker="|", color=NAVY, lw=1.1, label="radar beam passing")
    dw = np.array([r.dwells[t][0] for t in range(a, b)])
    ax.plot(t_s, dw, color=BLUE, lw=0.3, alpha=0.4)
    h = hits(r, a, b)
    novel = [(t, bnd) for t, bnd in h if thr[t, bnd] >= 0.5]
    if novel:
        ht, hb = zip(*novel)
        ax.scatter(np.array(ht) * spec.slot_ms / 1000, hb, s=16, color=ORANGE, edgecolor="white", lw=0.4, zorder=5,
                   label="intercept")
    n_int = len({(t, bnd) for t, bnd in novel})
    ax.set_title(f"{label}: {n_int} radar intercepts", fontsize=7, color=NAVY, loc="left", pad=2, fontweight="bold")
    ax.set_ylim(-1, spec.n_bands)
    ax.set_yticks([0, 8, 16, 24, 31])
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.set_ylabel("band (2-18 GHz)", fontsize=6, labelpad=1)
    for s in ax.spines.values():
        s.set_color("#8899AA")
        s.set_linewidth(0.5)
axes[1].set_xlabel("mission time (s)", fontsize=6, labelpad=1)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=BLUE, lw=0.8, alpha=0.6, label="where the receiver listened"),
           Line2D([], [], color=NAVY, marker="|", ls="", ms=5, mew=1.1, label="radar beam passing"),
           Line2D([], [], color=GREY, lw=1.5, label="benign emitter"),
           Line2D([], [], color=ORANGE, marker="o", ls="", ms=4, mec="white", label="intercept")]
fig.legend(handles=handles, fontsize=5.2, loc="lower center", ncol=4, frameon=False, handletextpad=0.3,
           columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.4, h_pad=0.6, rect=(0, 0.045, 1, 1))
fig.savefig("deck/waterfall.png", dpi=300)
print("window", a, b, {p: len(hits(r, a, b)) for p, r in runs.items()})
