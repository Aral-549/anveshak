# Contract: environment

## Purpose
Turns a `ScenarioSpec` into ground truth: a per-slot, per-band SNR grid, a
boolean transmission grid and an event log of every activity event of every
emitter. Deterministic given the seed. Does not decide what the receiver
detects (`receiver`) and never exposes truth to a policy (`scheduler` receives
only observations).

## Inputs
- `ScenarioSpec`: `n_bands` (int >= 1), `n_slots` (int >= 1), `slot_ms` (float > 0),
  `snr_floor_db` (float, default 0.0), `seed` (int), `emitters`: list of
  `EmitterSpec`
- `EmitterSpec`: `id` (str, unique), `family` in {`circular_scan`, `agile`,
  `fhss`, `bursty`, `continuous`}, `threat_weight` in [0,1], `appear_slot`
  (int >= 0, default 0), `vanish_slot` (int or None), plus family parameters:
  - `circular_scan`: `band`, `period`, `beam_slots` (tau), `phase`,
    `snr_main_db`, `sidelobe_db` (<= 0, relative to main)
  - `agile`: `bands` (list), `burst_slots`, `gap_slots`, `order` in
    {`uniform`, `cyclic`}, `snr_db`
  - `fhss`: `bands` (list), `sequence_len`, `snr_db`, plus bursty on/off params
  - `bursty`: `band`, `on_mean_slots`, `off_mean_slots`, `sigma` (log-normal), `snr_db`
  - `continuous`: `band`, `snr_db`

## Outputs
- `snr_db`: float array `[n_slots, n_bands]`, `-inf` where nothing is present;
  where several emitters share a cell, powers add in linear units
- `truth`: bool array `[n_slots, n_bands]` = `snr_db >= snr_floor_db`
- `per_emitter_present`: bool array `[n_emitters, n_slots, n_bands]` (same floor
  rule, per emitter; used for event attribution)
- `events`: list of `Event(emitter_id, band, start, end_exclusive, threat_weight)`,
  one per maximal run of consecutive present slots of one emitter in one band

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | circular_scan band 5, period 100, beam 2, phase 10, main 30 dB, sidelobe -40 dB, n_slots 300 | truth[:,5] true exactly at slots 10,11,110,111,210,211; 3 events | sidelobe at -10 dB is below the 0 dB floor |
| 2 | continuous band 0, 10 dB, n_slots 50 | truth[:,0] all true; 1 event [0,50) | |
| 3 | continuous band 3, appear_slot 150, n_slots 300 | truth[:150,3] false, truth[150:,3] true; 1 event [150,300) | pop-up |
| 4 | agile bands [3,7], cyclic, burst 1, gap 0 | slot t present in band 3 if t even, band 7 if t odd; every slot a separate event | runs are per band |
| 5 | case 1 but sidelobe -20 dB | truth[:,5] true in every slot; 1 event | 10 dB sidelobe >= floor: radar always visible, never "scans" |
| 6 | two continuous emitters, both band 2, 0 dB each | snr_db[:,2] = 10*log10(2) (about 3.01 dB); truth true; 2 events (one per emitter) | power sum; attribution kept separate |
| 7 | same spec, same seed, twice | bit-identical outputs | determinism |
| 8 | bursty emitter, seeds 1 and 2 | outputs differ | randomness flows from seed only |
| 9 | band index >= n_bands, or duplicate emitter id | ValueError | |
| 10 | empty emitter list | truth all false, events empty | |

## Edge cases that must be covered
- Scan window wrapping the episode end (phase + beam > n_slots): truncated, no
  index error, event ends at `n_slots`.
- `vanish_slot <= appear_slot`: ValueError.
- Emitter present in the last slot only: 1 event `[n_slots-1, n_slots)`.
- Generator must not use global RNG state (`numpy.random.default_rng(seed)` only).

## Explicitly out of scope
- Detection randomness, Pd, false alarms -> `receiver`.
- Deinterleaving pulses into signatures -> assumed done by the ESM front end;
  `receiver` reports `emitter_id` as the signature.
- Real-spectrum replay: `scripts/capture_rtl_power.py` produces the same
  `truth` / `snr_db` shapes and bypasses the emitter models.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
