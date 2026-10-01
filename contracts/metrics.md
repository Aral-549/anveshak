# Contract: metrics

## Purpose
Computes every figure of merit in the PS from a completed run (truth events +
dwell log + observation log + prediction log). This is the only module allowed
to produce numbers that appear in the console, the README or the deck.
Aggregation across seeds (CIs, Wilcoxon) lives in `scripts/benchmark.py`, which
calls this module per run.

## Inputs
- `events`: list of `Event(emitter_id, band, start, end_exclusive, threat_weight)` (from `environment`)
- `truth`: bool `[n_slots, n_bands]`
- `dwells`: list per slot of dwelt bands
- `observations`: list per slot of `Observation`
- `predictions`: list of `(slot_made, signature, predicted_start)` (from `belief`)
- `emitter_appear`: map `emitter_id -> appear_slot`
- `slot_ms`, `pred_tol` (slots), optional `oracle_weighted_intercepts`

## Outputs
`FigureOfMeritReport`:
- `interception_ratio` = intercepted events / events; `weighted_interception_ratio`
- An event is **intercepted** if some slot in `[start, end)` has a dwell on its
  band with a detection whose signature is its emitter
- `pd_empirical` = dwells with a true detection / dwells where truth is present in
  the dwelt band; `pfa_empirical` = false alarms / dwells where truth is absent
- `intercept_rate_per_s` = intercepted events / (n_slots * slot_ms / 1000)
- `ttfi_slots` per emitter = first intercepting slot - appear_slot; emitters
  never intercepted go into `censored` (list) and are excluded from
  `mean_ttfi`, never silently set to 0 or n_slots
- `reward_total`, `reward_per_slot`: +threat_weight for the first detection of
  each event only
- `cost_dwells` = total dwells (n_slots * M)
- For each prediction, the true start = the start of the event of that
  signature, among events with `start >= slot_made`, whose start is nearest to
  `predicted_start` (ties: earlier). Unmatched predictions (no later event) go
  into `pred_unmatched`. `pred_correct_pct` = share of matched predictions with
  abs error <= pred_tol; `mean_abs_intercept_time_error` in slots and ms
- `max_revisit_slots` per band: largest gap between consecutive visits, treating
  slot -1 as a virtual visit (same convention as `belief` staleness) and slot
  `n_slots` as a virtual visit at the end
- `oracle_efficiency` = weighted intercepts / oracle's, when provided

## Behavior cases (input -> expected output)
Shared fixture F: n_slots 6, 2 bands, slot_ms 10, M=1.
Events: E1 (emitter A, band 0, [0,2), w 1.0); E2 (emitter B, band 1, [2,3), w 0.5);
E3 (emitter A, band 0, [5,6), w 1.0). Both emitters appear at slot 0.
Dwells: 0,0,0,1,1,0. Detections of A at slots 0, 1, 5; nothing else.

| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | F | interception_ratio = 2/3; weighted = 2.0/2.5 = 0.8 | E2 missed (receiver on band 0 at slot 2) |
| 2 | F | pd_empirical = 3/3 = 1.0; pfa_empirical = 0/3 = 0.0 | present dwells: slots 0,1,5; absent: 2,3,4 |
| 3 | F | ttfi A = 0; B censored; mean_ttfi = 0 over 1 emitter | |
| 4 | F | reward_total = 2.0 (slot 0 and slot 5; slot 1 is the same event as slot 0); reward_per_slot = 1/3 | |
| 5 | F | intercept_rate_per_s = 2 / 0.06 = 33.33 | |
| 6 | F | max_revisit_slots: band 0 = 3 (last visit slot 2 -> next slot 5), band 1 = 4 (start -> slot 3) | |
| 7 | A has events starting 802 and 905; predictions (0,A,800), (0,A,900); pred_tol 2 | pred_correct_pct = 50.0; mean abs error = 3.5 slots = 35 ms | |
| 8 | prediction with no later event of that signature | goes to pred_unmatched, not counted as wrong or right | |
| 9 | detection in the event's band but of a different signature | does not intercept that event | attribution by signature |
| 10 | zero events | interception_ratio = None (not 0, not 1) | undefined, stated |

## Edge cases that must be covered
- M = 2 with both receivers on bands of the same event: counted once.
- Detection one slot after an event ends (the belief's gap merging does not
  apply here): does not intercept that event.
- Every ratio with a zero denominator returns None and a reason string.

## Explicitly out of scope
- Cross-seed aggregation, CIs, significance tests -> `scripts/benchmark.py`.
- Analytic predictions -> `scan_on_scan`.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
