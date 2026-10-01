"""Analytic scan-on-scan interception model. See contracts/scan_on_scan.md."""
from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class ScanOnScanResult:
    g: int
    L: int
    frac_phases_intercepting: float
    ever_intercepts: bool | None
    mean_interval_slots: float | None
    mean_interval_slots_coprime: float | None


def _check(name: str, value, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")


def overlapping_window_pairs(T_r: int, tau_r: int, T_s: int, tau_s: int, delta: int) -> int:
    """Number of (emitter window, receiver window) pairs that share a slot, per L slots.

    Each valid value v = delta + d (d in [-(tau_r-1), tau_s-1], g | v) maps to exactly
    one window pair per L, and distinct values mod L give distinct pairs.
    """
    g = gcd(T_r, T_s)
    L = T_r * T_s // g
    lo, hi = delta - (tau_r - 1), delta + (tau_s - 1)
    multiples = hi // g - (lo - 1) // g
    return min(multiples, L // g)


def analyze(T_r: int, tau_r: int, T_s: int, tau_s: int, delta: int | None = None) -> ScanOnScanResult:
    for name, v in (("T_r", T_r), ("tau_r", tau_r), ("T_s", T_s), ("tau_s", tau_s)):
        _check(name, v)
    if tau_r > T_r or tau_s > T_s:
        raise ValueError("window length cannot exceed its period")
    if delta is not None and (not isinstance(delta, int) or isinstance(delta, bool)):
        raise ValueError("delta must be an int")

    g = gcd(T_r, T_s)
    L = T_r * T_s // g
    span = tau_r + tau_s - 1
    frac = min(g, span) / g
    coprime = L / min(span, L) if g == 1 else None

    if delta is None:
        return ScanOnScanResult(g, L, frac, None, coprime, coprime)
    n = overlapping_window_pairs(T_r, tau_r, T_s, tau_s, delta)
    return ScanOnScanResult(g, L, frac, n > 0, (L / n) if n else None, coprime)
