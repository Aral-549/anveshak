# Contract: learned_policy

> **Superseded 2026-10-01 by `contracts/hit_model.md`.** The cross-entropy-method
> scorer below was never trained. Policy `learned` now uses the supervised hit
> model. The `features()` / `LearnedScorer` code and its golden tests remain as
> written and still pass, but no reported result depends on them.

## Purpose
The machine-learned scheduling policy. A small neural scorer, shared across all
bands, maps each band's belief features to a score; the scheduler takes the
top-M bands, and the revisit floor still applies (the scorer is plugged in as
`score_fn`, so floor behaviour is inherited from `scheduler`). Trained offline by
direct policy search (cross-entropy method) on the episode reward, using
training seeds only. Online, the policy keeps learning through the belief state
(hits and misses); the network weights are fixed during a mission.

Why the cross-entropy method rather than PPO: the reward (threat-weighted
intercepts over a 12,000-slot episode) is sparse and delayed, and the scorer has
~200 parameters. Population-based search on whole-episode return needs no value
function and no per-step credit assignment, parallelises across CPU cores, and
is reproducible from a seed. PPO remains a documented alternative, not a claim.

## Inputs
- `BeliefState` at slot t, `T_max`, `period_tol`
- weights file (JSON): `{"features": [...names], "W1": [[...]], "b1": [...], "W2": [...], "b2": float, "meta": {...}}`
- `rng` for the Thompson feature

## Features per band (in this order, all finite)
| # | name | definition |
|---|------|------------|
| 0 | window | 1 if a locked signature's predicted window covers t in this band (`abs(t - next_window) <= period_tol`), else 0 |
| 1 | window_threat | max threat hint of such signatures, else 0 |
| 2 | confirm | 1 if a CONFIRMING signature camps on this band at t, else 0 |
| 3 | confirm_threat | max threat hint of such signatures, else 0 |
| 4 | staleness | `min(staleness / T_max, 2)` |
| 5 | novel_mean | `alpha_novel / (alpha_novel + beta_novel)` |
| 6 | novel_ts | one Thompson sample from `Beta(alpha_novel, beta_novel)` |
| 7 | novel_evidence | `1 / (alpha_novel + beta_novel)` |
| 8 | band_threat | `band_threat` |
| 9 | occ_mean | `alpha / (alpha + beta)` |
| 10 | long_here | 1 if a `long` signature was detected at this band's last visit, else 0 |
| 11 | soon | 1 if a locked signature's next window in this band starts within `(period_tol, period_tol + 16]` slots, else 0 |

## Scorer
`score_b = W2 . tanh(W1 x_b + b1) + b2`, hidden width 16. Same weights for every
band, so the policy is permutation-equivariant and independent of `n_bands`.

## Training (scripts/train_learned.py)
- Cross-entropy method over the flat parameter vector: population 32, elite 6,
  diagonal Gaussian (std floor 0.02), initial std 0.5, 30 generations, seed 0.
- Initial mean: all zeros except `W1`/`W2` random with std 0.1 (seeded).
- Fitness of a candidate = mean weighted interception ratio over a batch of 15
  training scenarios (3 per suite S1-S5), 6,000 slots each, fresh batch every
  generation, drawn from seeds 0-999 only. All candidates in a generation see
  the same batch and the same random streams (common random numbers).
- Output: `artifacts/learned_policy.json` with weights, the per-generation
  fitness curve and the git-free config needed to reproduce it.

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | fresh belief, 4 bands, t=0, T_max=8 | features: window 0, confirm 0, staleness 1/8, novel_mean 0.5, novel_evidence 0.5, band_threat 0.5, occ_mean 0.5, long_here 0, soon 0 for every band | |
| 2 | sig A locked in band 2, next_window = t, threat 0.7 | band 2: window 1, window_threat 0.7; others window 0 | |
| 3 | sig A locked in band 2, next_window = t + 10, period_tol 2 | band 2: window 0, soon 1 | |
| 4 | sig A CONFIRMING in band 1, threat 0.6 | band 1: confirm 1, confirm_threat 0.6 | |
| 5 | staleness 1000, T_max 128 | staleness feature 2.0 | clipped |
| 6 | weights with all-zero W2 and b2 = 0 | all scores 0; scheduler picks lowest-index non-forced band | ties rule from scheduler |
| 7 | save then load weights | identical scores on the same features | round trip |
| 8 | weights file missing a feature name, or names in a different order | ValueError at load | no silent feature mismatch |
| 9 | training run twice with the same seed and the same worker count | identical fitness curve | reproducible |

## Edge cases that must be covered
- A score of NaN/inf is handled by the scheduler fallback (`POLICY_FALLBACK`),
  never propagated to a dwell.
- Training seeds are asserted to be disjoint from `scenarios.EVAL_SEEDS`.

## Explicitly out of scope
- The revisit floor -> `scheduler`. Belief updates -> `belief`.
- Online weight updates during a mission.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
