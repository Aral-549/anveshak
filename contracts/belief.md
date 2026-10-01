# Contract: belief

## Purpose
Maintains everything the scheduler knows, built only from past observations:
per-band occupancy posterior, per-band staleness, per-signature event history,
period/phase lock and next-window prediction, and hop transitions. Never reads
truth. Does not choose dwells (`scheduler`).

## Inputs
- `Observation` for slot t (from `receiver`)
- `DwellPlan` for slot t (which bands were visited, so misses are known)
- config: `n_bands`, `gamma` (discount in (0,1]), `period_tol` (slots,
  default 2), `min_period` (slots), `max_period` (slots), `unlock_misses` K
  (default 3)

## Outputs
- `BeliefState`:
  - per band: `alpha`, `beta`, `occ_mean = alpha/(alpha+beta)`, `visits`,
    `staleness` (slots since last visit; `t+1` if never visited),
    `alpha_novel`, `beta_novel` (same update, driven by novel hits only),
    `band_threat` (max threat hint of any signature that produced a novel hit in
    the band; 0.5 before any)
  - per signature: `event_starts` (list), `band_history`, `lock`:
    `None` or `(period, last_start, band)`, `next_window` (predicted start slot
    or None), `consecutive_misses`, `threat_hint` (running max), `run_len`,
    `long`, `confirm_band`, `confirm_until`
  - `predictions`: list of `(slot_made, signature, predicted_start)` for scoring
    by `metrics`

## Rules
- **Novel detection / observed event start**: a detection of signature `s` in
  band `b` at slot t is *novel* (starts a new observed event) iff `s` was not
  detected in `b` at the previous visit to `b`, or `b` had no previous visit.
  This is what stops a continuous emitter seen on every sweep from looking like
  a periodic one: it produces exactly one event start, never a lock.
- Novel occupancy: a visit is a novel hit iff it contains at least one novel
  detection; `alpha_novel/beta_novel` update like occupancy on that bit.
- Occupancy (only the dwelt band is updated): hit ->
  `alpha = gamma*alpha + 1, beta = gamma*beta`; miss ->
  `alpha = gamma*alpha, beta = gamma*beta + 1`. Prior (1, 1). A false alarm
  counts as a hit for occupancy (the belief cannot tell the difference) but
  creates no signature.
- Period lock: using the observed event starts of a signature, find the
  **largest** integer `T` in `[min_period, max_period]` such that every
  consecutive interval `I` satisfies `min_n |I - n*T| <= period_tol` with
  `n >= 1`. At least 2 intervals are required. The prediction is the first
  `last_start + k*T` after the current slot.
- **Long emitters**: `run_len` counts consecutive visits to a signature's band
  on which it was detected (reset by a visit without it). When
  `run_len > long_run` (default 12 slots, longer than any main-beam dwell), the
  signature is marked `long = True` permanently (continuous, bursty comms, or a
  radar seen through its sidelobes). Long signatures never enter CONFIRMING.
- **Multiband**: a signature whose event starts span 2 or more bands is
  `multiband = True` (agile radar, FHSS); it never enters CONFIRMING, since its
  timing is predicted by the hop model instead.
- **Aperiodic**: each entry into CONFIRMING increments `confirm_attempts`; after
  `max_confirms` (default 3) entries without a lock the signature stops being
  confirmed. The counter resets when a lock is acquired.
- **CONFIRMING**: a novel event start of a signature that is not locked, not
  long, not multiband and not aperiodic sets `confirm_band = b`, `confirm_until = t + max_period + period_tol`.
  Cleared on lock, on becoming long, or once `t > confirm_until`.
- **Provisional lock**: when a new event start does not yield a period from
  `find_period` (e.g. only 2 starts), and the previous start was in the same
  band, and the band was visited on at least 60% of the slots between the two
  starts (a camp), the last interval `I` becomes the period if
  `min_period <= I <= max_period`. Rationale: while camped, the first recurrence
  seen *is* the period (a miss from the floor interrupting the camp can only
  yield a multiple of it, whose predictions still hit).
- Unlock: if the scheduler dwelt on the locked band throughout a predicted
  window of length `period_tol*2+1` centred on `next_window` and saw no detection
  of that signature, `consecutive_misses += 1`. At K, `lock = None`.

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | gamma=1, band 3: hit, hit, hit, miss | alpha=4, beta=2, occ_mean=2/3 | |
| 2 | gamma=0.5, band 3: hit, miss | alpha=0.75, beta=1.25, occ_mean=0.375 | (1,1)->(1.5,0.5)->(0.75,1.25) |
| 3 | band never visited by slot 9 | staleness = 10, occ_mean = 0.5 | |
| 4 | sig A starts [0, 300, 700], tol 1, min 50, max 1000 | lock period 100; at slot 701, next_window = 800 | intervals 3T, 4T |
| 5 | sig A starts [0, 300] | lock None | only 1 interval |
| 6 | sig A starts [0, 301, 700], tol 1 | lock period 100 | residuals 1, 1 |
| 7 | sig A starts [0, 200, 400, 600] | lock period 200 | largest consistent T; ambiguity vs 100 is documented, not a bug |
| 8 | sig A starts [0, 137, 291, 503], tol 1, min 50 | lock None | intervals 137, 154, 212 share no period >= 50 |
| 9 | sig A starts [0, 5, 10], min_period 50 | lock None | below physical minimum |
| 10 | band 4 visited every slot 10..20; A detected at 10, 11, 12, 20 | event_starts [10, 20] | 10-12 is one event; visit at 13 had no A |
| 14 | band 4 visited at 0, 32, 64, 96, A detected every visit | event_starts [0], lock None | continuous-looking emitter never locks |
| 15 | band 4 visited at 0, 32, 64; A detected at 0 and 64 only | event_starts [0, 64] | miss at 32 breaks the run |
| 16 | gamma=1, band 2: novel hit, non-novel hit, miss | alpha_novel=2, beta_novel=3 | only novel hits count |
| 17 | A (threat 0.6) first detected at slot 100 in band 4, max_period 1000, tol 1 | confirm_band 4, confirm_until 1101 | CONFIRMING |
| 18 | band 4 visited every slot 100..401, A detected at 100 and 400 only | lock period 300 (provisional), next_window 700, confirm cleared | camp measures the period |
| 19 | band 4 visited only at 100, 101, 400; A at 100 and 400 | lock None | not a camp |
| 21 | A detected in band 4 at slot 0 and (novel) in band 5 at slot 100 | multiband = True, confirm_until None | agile |
| 22 | A has 3 unlocked CONFIRMING entries (starts 0, 1999, 4712 - no common period - visits elsewhere between) and a 4th novel start at 6000 | confirm_until None after 6000 | aperiodic cap |
| 20 | band 4 visited every slot 0..20, A detected every slot | long = True, confirm cleared; after a gap, a new novel start of A does not set confirm | beacon-like |
| 11 | locked A, 3 fully watched predicted windows with no A | lock None after the third | K = 3 |
| 12 | locked A, predicted window NOT watched (scheduler elsewhere) | consecutive_misses unchanged | absence of evidence is not a miss |
| 13 | false alarm in band 6 | band 6 hit for occupancy; no new signature | |

## Edge cases that must be covered
- Intervals of 0 (two event starts of one signature in the same slot, e.g. in
  two bands with M=2): keep the earliest-band one only.
- Signature seen in several bands (agile): lock is per signature, and
  `band` in the lock is the band of the last event. Predictions for agile
  emitters are only emitted when the hop model's top transition probability
  is >= 0.9 (cyclic hopping); otherwise no prediction.
- Belief update must be O(bands + signatures) per slot: no re-scan of history
  except on a new event start.

## Explicitly out of scope
- Choosing dwells -> `scheduler`. Scoring predictions -> `metrics`.
- Staggered-PRI / jittered-scan radars: tolerance only; documented limitation.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
