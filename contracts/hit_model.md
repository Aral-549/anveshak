# Contract: hit_model

## Purpose
A supervised model that predicts, for every band at slot t, the probability that
a dwell there now would land on a **new threat event**: an activity event of a
non-benign emitter (threat weight > 0.1) that is present in the band at t and has
not been intercepted yet. It is trained on simulator missions (features from the
receiver's own hits and misses, labels from simulator truth), and plugged into
the scheduler as policy `learned`, with score = predicted probability. The
revisit floor still applies.

Replaces the cross-entropy-method plan in `learned_policy.md` (never trained) and
the externally supplied `smart_scan_xgboost.pkl` (undocumented label, validation
AUC-PR 0.028, different training regime).

## Inputs
- `BeliefState` at slot t, `T_max`
- training: simulator missions from `scenarios.make(suite, seed)`, train seeds
  0-999 only; evaluation seeds 1000-1099 only

## Features per band (`HIT_FEATURES`, in this order, all finite)
| # | name | definition |
|---|------|------------|
| 0 | window | 1 if a locked signature's predicted window covers t in this band (`abs(t - next_window) <= period_tol`) |
| 1 | window_threat | max threat hint of such signatures, else 0 |
| 2 | soon | 1 if a locked signature's next window in this band starts in `(period_tol, period_tol + 16]` slots |
| 3 | time_to_window | `min(next_window - t)` over locked signatures in this band, clipped to [-5, 200], divided by 200; 1.0 if none |
| 4 | lock_period | locked period / 1000 of the soonest such signature, 0 if none |
| 5 | confirm | 1 if a CONFIRMING signature camps on this band at t |
| 6 | confirm_threat | max threat hint of such signatures, else 0 |
| 7 | staleness | `min(staleness / T_max, 2)` |
| 8 | novel_mean | `alpha_novel / (alpha_novel + beta_novel)` |
| 9 | novel_evidence | `1 / (alpha_novel + beta_novel)` |
| 10 | band_threat | `band_threat` |
| 11 | occ_mean | `alpha / (alpha + beta)` |
| 12 | long_here | 1 if a `long` signature was detected at this band's last visit |
| 13 | since_novel | slots since the last observed event start in this band / 1000, clipped to 12; 12 if none |

## Label (training only, from truth)
`y[t, b] = 1` iff some truth event `e` with `e.band == b`, `e.start <= t < e.end_exclusive`,
`e.threat_weight > 0.1`, and `e` not intercepted by the behaviour policy at any
slot `< t`.

## Data and training
- Behaviour policies for data collection, per mission: `predictive` (50%),
  `permuted_sweep` (25%), `bandit` (25%), chosen by seed.
- Rows: every 4th slot (random offset per mission), all 32 bands. All positives
  kept; negatives kept with probability 0.05 and given sample weight 20, so the
  weighted data matches the true class balance.
- Training set: seeds 0-59 per suite (300 missions). Validation for early
  stopping: seeds 60-79 per suite. Grouped by mission, never split within one.
- Model: XGBoost `binary:logistic`, hist, early stopping on validation AUC-PR,
  fixed seed. Saved as native JSON `artifacts/hit_model.json` with feature names.

## Evaluation (all on seeds 1000-1019 per suite, never seen in training)
1. Ranking quality: AUC-PR and ROC-AUC on every-8th-slot rows (no negative
   subsampling) collected under `predictive`, against: the no-skill rate,
   `novel_mean` alone, `window` alone, and `smart_scan_xgboost.pkl` scored on
   its own 10 features computed from the same rows (with the caveat that it was
   trained on a different regime).
2. End-to-end: threat-weighted interception ratio of policy `learned` vs
   `sweep` and `predictive`, same seeds, paired Wilcoxon.
3. Results written to `artifacts/hit_model_eval.json`. "Better" is claimed only
   for the metrics where it holds.

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | fresh belief, 4 bands, t=0, T_max=8 | per band: window 0, soon 0, time_to_window 1.0, lock_period 0, confirm 0, staleness 0.125, novel_mean 0.5, novel_evidence 0.5, band_threat 0.5, occ_mean 0.5, long_here 0, since_novel 12 | |
| 2 | sig A locked in band 2 (period 100), next_window = t = 300, threat 0.7, period_tol 2 | band 2: window 1, window_threat 0.7, time_to_window 0.0, lock_period 0.1 | |
| 3 | same with next_window = 310 | band 2: window 0, soon 1, time_to_window 0.05 | 10/200 |
| 4 | same with next_window = 900 | band 2: soon 0, time_to_window 1.0 | clipped at 200 |
| 5 | sig A last event start at slot 250 in band 1, t = 300 | band 1: since_novel 0.05; other bands 12 | |
| 6 | label: event (band 3, [10, 12), w 0.6), not intercepted | y[10,3] = y[11,3] = 1; y[12,3] = 0; y[10,2] = 0 | |
| 7 | label: same event intercepted at slot 10 | y[10,3] = 1, y[11,3] = 0 | novel only until first interception |
| 8 | label: event with threat 0.1 (benign) | y = 0 everywhere | benign excluded |
| 9 | model file with different feature names or order | ValueError at load | |
| 10 | dataset builder asked for any seed >= 1000 for training | AssertionError | seed hygiene |

## Edge cases that must be covered
- Two locked signatures in one band: `window`/`window_threat` take the max,
  `time_to_window` and `lock_period` come from the soonest window.
- NaN or inf in any feature: AssertionError in the builder (never silently
  trained on).

## Explicitly out of scope
- Online weight updates during a mission (the belief state does the online learning).
- The CRUD app (`scan_crud_app.md`), which will be re-pointed at this model once
  it is accepted.

## Results (2026-10-01, held-out seeds 1000-1019, 100 missions)
Artifacts: `artifacts/hit_model_eval.json`, `artifacts/hit_model_e2e.json`,
`artifacts/hit_model_e2e_hybrid01.json`; tuning on seeds 80-91 in `artifacts/tune_hybrid/`.

Ranking (4.8M band-slot rows, no-skill AUC-PR 0.0088):

| model | AUC-PR | ROC-AUC |
|---|---|---|
| hit model | 0.135 | 0.898 |
| smart_scan_xgboost.pkl (supplied, out of its regime) | 0.010 | 0.491 |
| novel_mean alone | 0.025 | 0.804 |

Scheduling, threat-weighted interception ratio (mean of 20 missions per suite):

| suite | sweep | predictive | learned (model only) | hybrid (hybrid_lam 0.1) | hybrid > predictive |
|---|---|---|---|---|---|
| S1 | 0.060 | 0.493 | 0.418 | 0.459 | 7/20, p 0.52 |
| S2 | 0.052 | 0.119 | 0.158 | 0.158 | 17/20, p < 0.001 |
| S3 | 0.189 | 0.649 | 0.459 | 0.664 | 7/20, p 0.81 |
| S4 | 0.052 | 0.444 | 0.375 | 0.479 | 12/20, p 0.48 |
| S5 | 0.068 | 0.197 | 0.242 | 0.247 | 15/20, p 0.001 |
| mean | 0.084 | 0.381 | 0.330 | 0.402 | 58/100, p 0.06 |

Reading: the model alone is myopic (it undervalues confirm camps) and loses on
radar suites. The hybrid (rules for timing, model for search) is significantly
better than `predictive` on agile/hopping (S2) and mixed (S5) missions and not
significantly different elsewhere. The overall improvement is not significant
at 0.05.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract
- [x] Golden tests exist for every behavior case above
