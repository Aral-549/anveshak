# ANVESHAK - Smart Scan Scheduler for Electronic Support Receivers
### Smart India Hackathon 2026 | Problem Statement SIH26055 | DRDO
Team Evinco. Working name "ANVESHAK" (Sanskrit: the one who searches) - change freely.

Status: DESIGN - awaiting human review. No implementation code exists yet.
Every module below has a contract in `contracts/`; code starts only after review.

---

## 1. The problem in plain terms

An ES (Electronic Support) receiver has to watch a wide spectrum, e.g. 2-18 GHz,
but it can only listen to a small slice (one band) at a time. Hostile emitters
are only "visible" in short bursts: a search radar's beam sweeps past us for a
few tens of milliseconds every few seconds, a frequency-agile radar jumps between
bands, a radio transmits in bursts. We catch a signal only if we are tuned to the
right band at the right moment.

Today receivers use open-loop sweeps planned from pre-mission intelligence. With
no reliable intelligence, a sweep spends equal time on empty bands, on harmless
always-on emitters and on real threats. It can also fall into lock-step with a
radar's scan so that it never sees that radar at all (section 6).

We build a scheduler that learns, during the mission, where and when to look.
It learns only from its own hits and misses, and it still guarantees that every
band gets searched.

## 2. What we deliver

1. A **simulated RF environment** with ground truth (band x time-slot
   transmission grid) and five emitter families, seed-reproducible.
2. A **receiver model** with probability of detection from SNR (Marcum Q,
   square-law detector), configurable false-alarm rate and N parallel receivers.
3. A **learning scheduler**: belief state (period/phase lock, discounted
   Bernoulli occupancy, hop-pattern model) + policy (predictive heuristic and a
   PPO policy) + a **revisit floor** that no learned policy can override.
4. A **scan-on-scan analysis module** giving exact interception ratio and mean
   intercept time for a periodic receiver against a periodic scanning radar,
   validated against brute force and against simulation.
5. The **figures of merit** requested in the PS, computed from logs against
   truth, with confidence intervals over held-out scenarios.
6. A **web console**: live waterfall (truth vs where we listened vs hits), with
   the open-loop sweep and ANVESHAK running on the same scenario side by side.

## 3. System model

Time is divided into dwell slots. In each slot each receiver is tuned to exactly
one band. All core logic works in integer slots; physical units only exist at
the scenario boundary.

| Parameter | Default | Notes |
|---|---|---|
| Spectrum | 2-18 GHz | configurable |
| Bands `B` | 32 x 500 MHz | receiver instantaneous bandwidth = 1 band |
| Slot (dwell) | 10 ms | retune time folded into the slot |
| Receivers `M` | 1 | 2 = one searcher + one tracker |
| Episode | 12,000 slots (120 s) | |
| Pfa per dwell | 1e-4 | threshold set from Pfa |
| Truth floor | SNR >= 0 dB | below this the emitter is "not present" |
| Revisit floor `T_max` | 128 slots (1.28 s) | every band visited at least this often |

### Emitter families (all parameters randomised per scenario)

| Family | Behaviour in the band x time grid | Threat weight | Range |
|---|---|---|---|
| Circular-scan search radar | fixed band; main beam passes every `T` slots for `tau` slots; sidelobes present if strong enough | 0.6 | T = 2-12 s, beamwidth 1-3 deg -> tau = 1-10 slots |
| Sector / track-while-scan radar | same model, short period | 0.7 | T = 0.5-2 s |
| Frequency-agile radar | bursts that hop across an agile set of 2-8 bands; hop order uniform-random or cyclic | 0.8 | burst 1-5 slots |
| FHSS / bursty comms | semi-Markov on/off (log-normal durations); FHSS hops a set of 4-16 bands per slot, cyclic pseudo-random sequence | 0.4 | on 0.5-5 s, off 2-30 s |
| Benign continuous emitter (beacon, nav, broadcast) | always on in one band | 0.1 | - |
| Pop-up fire-control radar | appears at a random slot, then continuous high-SNR illumination | 1.0 | appear anywhere in episode |

The benign emitters are there on purpose. An open-loop sweep wastes time on
them, and the PS says so directly.

### What the scheduler is allowed to see

Only its own observations: for each slot and each band it dwelt on, whether a
detection happened, and for each detection a **signature id** and a **threat
hint**. These stand in for the ESM front end's pulse deinterleaving and
parameter-based threat assessment, which are out of scope. There is no emitter
library, no count of emitters and no frequency plan. Truth is used only for
scoring and for the training reward.

## 4. Pipeline and stage boundaries

```
 ScenarioSpec ──► [1 environment] ──► TruthGrid + EventLog (ground truth, never shown to policy)
                                             │
       ┌─────────────────── per slot t ──────┼──────────────────────────┐
       │                                     ▼                          │
       │  DwellPlan(t) ──► [2 receiver] ──► Observation(t)              │
       │       ▲                                  │                     │
       │       │                                  ▼                     │
       │  [4 scheduler] ◄── BeliefState(t) ◄── [3 belief update]        │
       │   (policy + revisit floor)                                     │
       └────────────────────────────────────────────────────────────────┘
                                             │
             RunLog (dwells, observations, decisions, predictions)
                                             ▼
                                   [5 metrics] ──► FigureOfMeritReport
       [6 scan_on_scan] (analytic, standalone; also cross-checked against 5)
```

Each arrow is a logged boundary (AGENTS.md rule 5). Logging uses the Python
`logging` module with a JSON formatter, one record per boundary crossing:

```json
{"stage":"scheduler","run_id":"...","slot":4211,"in":{"staleness_max":97,"locks":3},
 "out":{"bands":[7],"source":"PREDICTED_WINDOW","sig":"E3","pred_start":4211}}
```

`source` is always one of `REVISIT_FLOOR | PREDICTED_WINDOW | POLICY | SWEEP |
ORACLE`, so every dwell can be traced back to the rule or model that chose it.

## 5. Scheduler design

### 5.1 Layer 0 - revisit floor (safety invariant)

If a band has not been visited for `T_max` slots, it is visited now, whatever the
policy wants. A learned policy can never stop watching part of the spectrum,
because a new threat can appear anywhere. Same idea as GovScore in PAIMANA: the
model ranks, and a hard rule sets the minimum.
Feasibility requires `T_max >= B / M`; otherwise the configuration is rejected.

### 5.2 Layer 1 - belief state (learned from hits and misses)

Per band and per signature, updated every slot from observations only:

- **Occupancy**: discounted Beta-Bernoulli per band (`alpha, beta`, discount
  `gamma`). Gives a posterior probability that a dwell in this band catches
  something, and it forgets, so the belief keeps up as the environment changes.
- **Period / phase lock** per signature: from the start slots of observed
  events, find the largest period `T` such that every inter-event interval is an
  integer multiple of `T` within tolerance (an approximate GCD). At least 2
  intervals are needed before a lock. Once locked, we predict the next window
  `t_hat = t_last + kT`. If `K` predicted windows in a row are missed, the lock is
  dropped. This turns a scanning radar from something we find by luck into
  something we find on schedule.
- **Hop model** per signature: band-to-band transition counts. For cyclic hop
  sequences this becomes nearly deterministic. For uniform-random hopping it
  stays flat, which correctly tells the scheduler not to chase that emitter.
- **Staleness** per band: slots since last visit.

### 5.3 Layer 2 - policy

Every slot, each band gets a score and the top `M` bands are chosen (subject
to Layer 0):

- **P1 predictive heuristic** (strong non-learned baseline):
  `score_b = sum_sig w_sig * P(window of sig active at t in b) + w_b * ThompsonSample_b + lambda * staleness_b / T_max`
- **P2 PPO policy** (the ML scheduler the PS asks for): the same per-band
  feature vector (staleness, occupancy posterior mean and count, time to
  predicted window, lock confidence, threat hint, time since last hit) goes
  through **one small MLP shared across all bands**. The result is a score per
  band, then a softmax. Sharing weights makes the policy
  permutation-equivariant and independent of the band count, so a model trained
  on 32 bands runs on 64. The revisit floor is applied as an action mask.
  Training runs offline on domain-randomised scenarios (Stable-Baselines3 PPO,
  custom policy). Reward from truth: `+w_e` for the first detection of each
  activity event, `+bonus` for the first-ever detection of an emitter, `-c` per
  dwell.

What the PPO has to learn that P1 does not encode: **when to camp and when to
sweep**. After a first hit on a search radar, staying on that band for up to one
scan period guarantees a second hit and therefore a fast period lock, but it
costs search time everywhere else. That trade-off depends on the scenario, and
it is the most useful thing a learned scheduler can add here.

Emitter lifecycle as the scheduler sees it:
`UNSEEN -> DISCOVERED (1 event) -> CONFIRMING -> LOCKED (period known) -> LOST (K misses) -> CONFIRMING ...`

### 5.4 Baselines (all run on identical seeds)

| Policy | What it represents |
|---|---|
| `sweep` | today's open-loop linear sweep |
| `permuted_sweep` | sweep with a fresh random band order each cycle (no learning, breaks sync) |
| `bandit` | discounted Thompson sampling only (learns from hits/misses, no timing model) |
| `predictive` (P1) | full belief state + heuristic |
| `ppo` (P2) | full belief state + learned policy |
| `oracle` | sees truth; upper bound, never deployable |

Whichever of P1 or P2 wins on held-out scenarios is shipped as the default. If
PPO does not beat P1, we report that and keep it as an ablation. We do not
tune the evaluation until it wins.

## 6. Periodic-scan interception (scan-on-scan)

This covers the PS item "approaches to intercept a periodic scan receiver
optimally". Model: the radar illuminates us for `tau_r` slots every `T_r`
slots, and our receiver visits a given band for `tau_s` slots every `T_s` slots
(a sweep of B bands gives `T_s = B`, `tau_s = 1`). Let `g = gcd(T_r, T_s)`.

Exact discrete results (derivation in `contracts/scan_on_scan.md`):

- An intercept **ever** happens only if some offset `d` in the
  `tau_r + tau_s - 1` window satisfies `g | (Delta_phase + d)`. Over random
  phases, the fraction of phases that ever intercept is
  `min(1, (tau_r + tau_s - 1) / g)`.
- If `g = 1`, intercepts recur with mean interval `T_r * T_s / (tau_r + tau_s - 1)`.

Consequences we demonstrate in the console:

1. **Synchronisation trap.** A 32-band sweep (`T_s = 32`) against a radar with
   `T_r = 320` slots (3.2 s), `tau_r = 2`: `g = 32`, so only **1 in 16** relative
   phases ever intercepts. For the other 15/16 the open-loop receiver never sees
   that radar.
2. **Without lock-step, the physics is still slow.** `T_r = 401`, `tau_r = 2`,
   `T_s = 32`: `g = 1`, mean interval `401 * 32 / 2 = 6,416` slots, i.e. about
   64 s between sightings of a 4 s radar.
3. **Strategy, in order of how much we know:**
   - Nothing known: **permuted sweep**. Randomising band order each cycle removes
     the synchronisation trap, and the mean interval becomes about
     `T_r * B / tau_r` regardless of `T_r`.
   - One hit: **confirm**. Camp or densely revisit that band for one maximum
     scan period to get the second event.
   - Two or more intervals: **phase lock**. Dwell exactly on predicted windows;
     interception ratio for that radar goes to about `Pd`, costing `tau_r` slots
     per scan instead of the full band.
4. Analytic predictions from this module are compared with simulated
   `sweep`/`permuted_sweep` runs. Agreement is itself a reported result, and it
   answers the PS request that "the model should enable prediction of intercept
   time and interception ratio".

## 7. Figures of merit (PS list -> our definition)

| PS asks for | Our definition (see `contracts/metrics.md`) |
|---|---|
| Probability of detection | detections / dwells where truth is present in the dwelt band; plus analytic `Pd(SNR, Pfa)` curve |
| Probability of false alarm | detections with no truth / dwells where truth is absent |
| Sensitivity | SNR needed for Pd = 0.9 at the configured Pfa (analytic), plus the empirical Pd-vs-SNR curve |
| Average intercept rate | intercepted activity events per second, overall and per emitter family |
| Interception ratio | intercepted events / total events, plain and threat-weighted |
| Intercept time | time to first intercept (TTFI) per emitter from its appearance; never-intercepted emitters reported as censored, not dropped |
| Avg reward / cost | mean training reward per slot; cost = dwells spent |
| % correct predictions | locked-emitter window predictions that land within tolerance of the true window start |
| Avg intercept time error | mean abs(predicted window start - true start), slots and ms |
| (added) oracle efficiency | weighted intercepts / oracle weighted intercepts |
| (added) max revisit time | per band; must never exceed `T_max` |

## 8. Evaluation protocol

- **Scenario suites** (each 100 held-out seeds, disjoint from training seeds):
  - S1 sparse radars + benign beacons
  - S2 frequency-agile radars + FHSS comms
  - S3 dense benign background + one pop-up fire-control radar (headline metric:
    TTFI of the pop-up)
  - S4 synchronisation trap (radar periods commensurate with the sweep)
  - S5 fully randomised mix, including parameter ranges never seen in training
    (tests the "no prior intelligence" claim)
- **Statistics**: mean and 95% bootstrap CI per metric; paired Wilcoxon
  signed-rank test of each policy vs `sweep` on the same seeds.
- **Ablations**: remove period lock, remove occupancy, remove revisit floor,
  vary `M`.
- **Measured, not asserted.** Every number in the deck and the console is read
  from an artifact written by `scripts/benchmark.py`. The API returns 404 for
  any figure that has not been produced, the same rule we used in PAIMANA.

## 9. Real-input integration test

Radar truth cannot be obtained legally at a hackathon, so radar scenarios stay
synthetic, and we say so. For the "real input" requirement (AGENTS.md section 6)
we replay **real spectrum occupancy**: an `rtl_power` capture from an RTL-SDR
over, e.g., 400 MHz-1.7 GHz (GSM, LTE, ISM, broadcast) is thresholded into a
band x time truth grid. The unchanged scheduler then runs on it through the
same environment interface. A public crowdsourced dataset (ElectroSense PSD
scans) is the fallback if no SDR is available.

## 10. Tech stack and repo layout

Python 3.12 · NumPy · SciPy (Marcum Q via `ncx2`) · Gymnasium · PyTorch +
Stable-Baselines3 (PPO) · FastAPI + Pydantic v2 (REST + WebSocket stream) ·
Next.js + React + Tailwind (canvas waterfall) · Docker · Azure (same VM as
before).

```
prototype3_evinco/
  DESIGN.md  BUGLOG.md  README.md
  contracts/            environment, receiver, belief, scheduler, metrics, scan_on_scan
  core/                 env.py receiver.py belief.py scheduler.py metrics.py scan_on_scan.py
  rl/                   gym_env.py train_ppo.py policy.py
  scripts/              benchmark.py capture_rtl_power.py
  backend/app/          FastAPI: /simulate /stream /benchmark /scan-on-scan
  frontend/             Next.js console
  tests/golden/         hand-verified cases from each contract (frozen)
  tests/                property + integration tests
  artifacts/            measured benchmark JSON, trained weights
```

## 11. Console (what the judges see)

- **Split waterfall**: bands on the y axis, time scrolling on the x axis. Truth
  is shaded, the receiver's dwell path is a line, hits and misses are marked.
  Left panel is the open-loop sweep, right panel is ANVESHAK, both on the same
  seed.
- **Emitter board**: discovered emitters, lifecycle state, TTFI, locked period,
  countdown to the next predicted window, and a running prediction error.
- **Metric tiles**: interception ratio, pop-up TTFI, Pd, Pfa, oracle efficiency.
  Each tile shows both policies.
- **Scan-on-scan calculator**: enter radar period and beam dwell; see the
  synchronisation map (which phases never intercept) and the mean intercept time
  for sweep, permuted sweep and phase lock.

## 12. Build order

1. `scan_on_scan` + `receiver` Pd model (pure math, fastest to golden-test)
2. `environment` + scenario generator
3. `metrics` (needed before any policy can be judged)
4. `sweep`, `permuted_sweep`, `oracle`, `bandit`
5. `belief` period lock + `predictive` policy
6. `benchmark.py` over S1-S5 -> first real numbers
7. Gym wrapper + PPO training
8. API + console
9. RTL-SDR real-input integration test, deploy

Steps 1-6 give a complete, defensible result without RL. PPO is layered on top,
so the RL training risk cannot sink the project.

## 13. Risks

| Risk | Mitigation |
|---|---|
| PPO does not beat the heuristic | P1 is shipped; PPO is reported honestly as an ablation |
| Simulator too easy (everything looks good) | oracle bound + S5 out-of-range suite + real-input replay |
| Period estimator locks onto 2T or T/2 | largest consistent period + K-miss unlock; ambiguity documented in contract |
| DRDO judges probe domain depth | scan-on-scan theory with exact results, Marcum-Q detection model, cited sources |
| "Deinterleaving/threat hint is cheating" | stated as ESM front-end input; ablation with hint removed and with signature mis-association |

## 14. Open questions for review

1. Is the Evinco team (same Team ID 120057) submitting this, and is the name
   ANVESHAK acceptable?
2. Does anyone have an RTL-SDR or HackRF for the real-input test? If not, we use
   ElectroSense replay.
3. Default `M = 1` receiver - or present `M = 2` (searcher + tracker) as the
   headline configuration?
4. Scope of the idea-stage submission: design + analytic results only, or should
   steps 1-6 be built before the PPT so the deck carries measured numbers (as
   PAIMANA and MarSlick did)?
