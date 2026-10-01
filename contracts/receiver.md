# Contract: receiver

## Purpose
Given where the receivers are tuned in a slot and the truth for that slot,
produce what the ES receiver actually reports: detections (with signature and
threat hint) and false alarms. Owns the Pd/Pfa model. Does not choose where to
tune (`scheduler`) and does not keep history (`belief`).

## Detection model
Square-law detector, non-fluctuating signal, single look:
`Pd(snr_lin, Pfa) = Q1(sqrt(2*snr_lin), sqrt(-2*ln(Pfa)))`, implemented as
`scipy.stats.ncx2.sf(-2*ln(Pfa), df=2, nc=2*snr_lin)`.
Each present emitter in the dwelt band is detected independently with its own
Pd. If the band is empty, a false alarm fires with probability `Pfa`.

## Inputs
- `dwell_bands`: list of distinct ints, `1 <= len <= M`, each in `[0, n_bands)`
- `slot`: int
- `per_emitter_snr_db`: for the slot, map `band -> list[(emitter_id, snr_db, threat_weight)]`
  (only emitters present per the environment floor)
- `pfa`: float in (0, 1]
- `threat_hint_noise`: float >= 0 (std of Gaussian noise added to threat weight, clipped to [0,1])
- `rng`: `numpy.random.Generator`

## Outputs
- `Observation(slot, per_band: {band -> BandObservation})`
- `BandObservation`: `detected` (bool), `detections`: list of
  `(signature, threat_hint, snr_db)`, `false_alarm` (bool)
  - `signature` = `emitter_id`; `None` for a false alarm
  - `detected = len(detections) > 0 or false_alarm`

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | Pd function, snr_lin = 0, any Pfa | Pd = Pfa exactly | noise-only limit |
| 2 | Pd function, snr_db = 100, Pfa = 1e-6 | Pd = 1.0 (to 1e-12) | |
| 3 | Pd function, fixed Pfa, snr_db from -10 to 30 | strictly non-decreasing | |
| 4 | Pd function, Pfa = 1e-6, Pd = 0.9 inverse | SNR within 0.3 dB of 13.1 dB | Albersheim N=1 gives 13.11 dB |
| 5 | dwell [4], emitter only in band 5 at 60 dB | band 4: no detections | no leakage between bands |
| 6 | dwell [2], empty band, Pfa = 1.0 | false_alarm = True, detected = True, detections empty | |
| 7 | dwell [2], empty band, Pfa tiny (1e-12), 10,000 trials | zero false alarms | |
| 8 | dwell [1,1] | ValueError (duplicate) | |
| 9 | len(dwell) > M | ValueError | |
| 10 | dwell [n_bands] | ValueError | |
| 11 | threat_hint_noise = 0 | threat_hint equals the emitter's threat_weight | |
| 12 | 200,000 trials at the case-4 SNR | empirical Pd in [0.895, 0.905] | statistical, fixed seed |

## Edge cases that must be covered
- Two emitters in one dwelt band: each gets its own Bernoulli draw; possible to
  detect one and miss the other.
- A present emitter AND a false alarm in the same band in one slot: false alarms
  are drawn only when the band is truly empty, so this cannot happen by
  construction.
- `snr_db = -inf` never reaches the receiver (filtered by the environment floor);
  if it does, Pd = Pfa, no crash.

## Explicitly out of scope
- Multi-pulse integration, fluctuating (Swerling) targets: fixed as a documented
  simplification.
- Where to tune -> `scheduler`. History and learning -> `belief`.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
