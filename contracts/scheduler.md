# Contract: scheduler

## Purpose
Chooses which `M` bands to dwell on in the next slot from the `BeliefState`,
under the revisit floor. Hosts every policy (`sweep`, `permuted_sweep`,
`bandit`, `predictive`, `learned`, `hybrid`, `oracle`) behind one interface. Does not update
belief and never sees truth, except the `oracle` policy, which is marked
non-deployable.

## Inputs
- `BeliefState` at slot t
- config: `n_bands` B, `M` (receivers), `T_max` (revisit floor), policy name and
  params (`lambda`, learned-policy weights, seed)
- `oracle` only: truth row and un-intercepted event set for slot t

## Outputs
- `DwellPlan(slot, bands: list[int] of length M, distinct, sources: list[str])`
  where each source is one of `REVISIT_FLOOR | PREDICTED_WINDOW | CONFIRM | POLICY | POLICY_FALLBACK | SWEEP | ORACLE`

## Policy scores
- `bandit`: `score_b = ThompsonSample(Beta(alpha_novel_b, beta_novel_b)) * band_threat_b`
- `predictive`: a band with an active predicted window
  (`|t - next_window| <= period_tol` of a locked signature) scores
  `10 + threat`; else a band that is the `confirm_band` of a CONFIRMING
  signature (`t <= confirm_until`) scores `5 + threat` (source `CONFIRM`); otherwise `score_b = bandit score + lambda * staleness_b / T_max`
- `hybrid`: same tiers as `predictive` (predicted window `10 + threat`, confirm camp
  `5 + threat`), but every other band scores
  `score_fn(t, belief)[b] + hybrid_lam * staleness_b / T_max` (the hit model's
  probability plus an optional staleness term; `hybrid_lam` defaults to 0)
  instead of the bandit + staleness score.
  Sources: `PREDICTED_WINDOW`, `CONFIRM`, else `POLICY`. Needs `score_fn`.
- `oracle`: sum of threat weights of truth events active in the band at t that the
  oracle has not yet intercepted; ties / nothing active -> most stale band

## Rules
1. Feasibility: `T_max >= ceil(B / M)`, else ValueError at construction.
2. Revisit floor first, earliest-deadline-first. Band b must be visited by slot
   `last_visit_b + T_max`; its slack is `s_b = T_max - staleness_b`. The number
   of bands forced now is `r = min(M, max(0, max_{j>=0} (#{b : s_b <= j} - M*j)))`,
   and they are the `r` smallest-slack bands (ties: lowest index). This forces a
   band *before* it is overdue whenever waiting would make some deadline
   unmeetable; a band with `staleness >= T_max` is always forced. Staleness is
   counted so that a band visited at slot t has staleness 1 at slot t+1. (A
   naive "force only when overdue" rule was rejected: with several bands due in
   the same slot it violates the floor - see case 8.)
3. Remaining receivers are filled by the policy, never re-choosing a forced band.
4. Ties in any policy score break on lowest band index. Randomness only via the
   seeded generator.

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | sweep, B=4, M=1, slots 0..7 | bands 0,1,2,3,0,1,2,3; source SWEEP | |
| 2 | sweep, B=4, M=2, slots 0..3 | [0,1],[2,3],[0,1],[2,3] | |
| 3 | permuted_sweep, B=4, M=1, seed s, slots 0..11 | each block of 4 slots is a permutation of {0,1,2,3}; blocks differ for some seed | |
| 4 | B=4, M=1, T_max=3 | ValueError | 3 < 4 |
| 5 | predictive, B=8, M=1, T_max=64, sig A locked in band 7 with next_window = t, no band due | band 7, source PREDICTED_WINDOW | |
| 6 | predictive, two locked sigs predicted at t, bands 2 (w 0.6) and 5 (w 1.0), M=1 | band 5; band 2 logged as conflict_dropped | threat weight wins |
| 7 | any policy wanting band 7, band 2 staleness = T_max | band 2, source REVISIT_FLOOR | floor overrides the model |
| 8 | degenerate policy that always scores band 0 highest, B=4, M=1, T_max=4, 1,000 slots | no band ever has staleness > 4 | property |
| 9 | bandit, band 1 with alpha_novel=21, beta_novel=1, all others alpha_novel=1, beta_novel=21, equal band_threat, fixed seed | band 1 in >= 95% of 1,000 plan calls (floor disabled via T_max = 10^6) | learns from hits/misses |
| 12 | bandit, band 0 with only non-novel hits (continuous benign emitter), band 1 with novel hits | band 1 preferred | a harmless always-on emitter does not attract the bandit |
| 13 | predictive, sig A CONFIRMING in band 3 (confirm_until >= t), no windows, no band due | band 3, source CONFIRM | camp after a first hit |
| 14 | predictive, predicted window in band 5 and a CONFIRMING sig in band 3, M=1 | band 5 | a known window outranks a camp |
| 15 | hybrid, predicted window in band 5, score_fn favours band 1 | band 5, source PREDICTED_WINDOW | rules keep the timing tiers |
| 16 | hybrid, no windows, no confirm, no band due, score_fn = [0.1, 0.9, 0.2, 0.3] | band 1, source POLICY | model decides the search |
| 18 | hybrid, hybrid_lam 1.0, T_max 64, score_fn = [0.5, 0.6, 0, 0], band 0 staleness 40, band 1 staleness 1, no windows | band 0 (0.5 + 40/64 > 0.6 + 1/64) | staleness term spreads search |
| 17 | hybrid without score_fn | ValueError at construction | |
| 10 | oracle, truth shows un-intercepted events in bands 1 (w 0.4) and 3 (w 1.0), M=1 | band 3, source ORACLE | |
| 11 | any policy | output bands distinct, length M, all in [0,B) | property over 10,000 random belief states |

## Edge cases that must be covered
- All bands due at once (e.g. slot 0 with T_max small): the most stale first,
  then lowest index; must not deadlock.
- Learned policy returning NaN scores: fall back to `predictive` for that slot and
  log `source = POLICY_FALLBACK`. Counts as a bug to investigate, not silence.
- `M > B`: ValueError.

## Explicitly out of scope
- Belief updates -> `belief`. Metric computation -> `metrics`.
- Learned-policy features and training -> `contracts/learned_policy.md`.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
