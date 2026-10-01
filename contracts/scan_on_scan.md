# Contract: scan_on_scan

## Purpose
Analytic interception model for a periodic receiver against a periodic scanning
emitter in integer dwell slots: whether an intercept can ever happen, what
fraction of relative phases intercept, and the mean interval between intercepts.
Pure math, no simulation. Cross-checking against simulation is `metrics` +
`scripts/benchmark.py`, not this module.

## Model
- Emitter window: slots `{phi_r + k*T_r + i : k in Z, 0 <= i < tau_r}`
- Receiver window (on that band): slots `{phi_s + m*T_s + j : m in Z, 0 <= j < tau_s}`
- An intercept is a slot belonging to both sets. `g = gcd(T_r, T_s)`,
  `L = lcm(T_r, T_s)`, `Delta = (phi_s - phi_r) mod g`.

Derivation: overlap requires `k*T_r - m*T_s = (phi_s - phi_r) + (j - i)`. The
left side ranges over exactly the multiples of `g`. `j - i` takes the
`tau_r + tau_s - 1` consecutive values `-(tau_r-1) .. (tau_s-1)`. So an intercept
exists iff one of those values `d` satisfies `g | (Delta + d)`. The number of
such `d` is `n_d`, and intercepting windows recur `n_d` times per `L` slots.

## Inputs
- `T_r, tau_r, T_s, tau_s`: positive ints, `tau_r <= T_r`, `tau_s <= T_s`
- `delta`: optional int, relative phase; if omitted, results are averaged over
  uniform phase

## Outputs
- `ScanOnScanResult`: `g`, `L`, `ever_intercepts` (bool, only when `delta` is
  given), `frac_phases_intercepting` in [0,1] = `min(1, (tau_r+tau_s-1)/g)`,
  `mean_interval_slots` = `L / n_d` for the given `delta` (None when `n_d = 0`),
  and `mean_interval_slots_coprime` = `T_r*T_s/(tau_r+tau_s-1)` when `g = 1`

## Behavior cases (input -> expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | T_r=5,tau_r=1,T_s=4,tau_s=1 | g=1, L=20, frac=1.0, mean_interval=20 | radar 0,5,10,15,20; rx 0,4,..,20 -> coincide at 0 and 20 |
| 2 | T_r=6,tau_r=1,T_s=4,tau_s=1 | g=2, L=12, frac=0.5 | rx phases 0,2 intercept; 1,3 never (odd vs even slots) |
| 3 | case 2 with delta=1 | ever_intercepts=False, mean_interval=None | synchronisation: never |
| 4 | case 2 with delta=0 | ever_intercepts=True, mean_interval=12 | coincide at 0, 12 |
| 5 | T_r=320,tau_r=2,T_s=32,tau_s=1 | g=32, frac=0.0625 | sync trap: 1 phase in 16 ever intercepts |
| 6 | T_r=401,tau_r=2,T_s=32,tau_s=1 | g=1, frac=1.0, mean_interval=6416 | 401*32/2 |
| 7 | T_r=10,tau_r=10,T_s=7,tau_s=1 | frac=1.0, mean_interval=7 | emitter always on -> every receiver visit hits |
| 8 | tau_r=0 or T_r=0 | ValueError | |
| 9 | tau_r > T_r | ValueError | |

## Edge cases that must be covered
- Property test: for all T_r, T_s <= 40 and all tau, delta, the analytic
  `ever_intercepts` and `mean_interval_slots` equal brute-force enumeration over
  one full `L` period. This is the main guarantee.
- `tau_r + tau_s - 1 >= g` gives frac exactly 1.0, never above.
- Very large coprime periods (T_r=1e6+3, T_s=64): no overflow; `L` computed with
  Python ints.

## Explicitly out of scope
- Random / permuted sweeps (stochastic). Their intercept statistics are measured
  by `metrics` over simulation runs.
- Detection probability < 1 inside a window. `receiver` owns Pd; the analytic
  result here is the geometric upper bound (Pd = 1).
- Non-integer or jittered periods (radar stagger). Documented as a limitation.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [ ] Implementation matches this contract
- [ ] Golden tests exist for every behavior case above
