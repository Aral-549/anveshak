"""Figures of merit from a completed run. See contracts/metrics.md.

The only module allowed to produce numbers shown in the console, README or deck.
"""
import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from core.log import boundary, get_logger

log = get_logger("metrics")


@dataclass
class FigureOfMeritReport:
    n_slots: int
    n_events: int
    n_intercepted: int
    interception_ratio: float | None
    weighted_interception_ratio: float | None
    pd_empirical: float | None
    pfa_empirical: float | None
    intercept_rate_per_s: float
    ttfi_slots: dict
    censored: list
    mean_ttfi_slots: float | None
    reward_total: float
    reward_per_slot: float
    cost_dwells: int
    pred_correct_pct: float | None
    mean_abs_intercept_time_error_slots: float | None
    mean_abs_intercept_time_error_ms: float | None
    pred_matched: int
    pred_unmatched: list
    max_revisit_slots: list
    oracle_efficiency: float | None
    event_intercepted: list = field(repr=False, default_factory=list)
    undefined: dict = field(default_factory=dict)


def _ratio(num: float, den: float, name: str, undefined: dict, reason: str) -> float | None:
    if den == 0:
        undefined[name] = reason
        return None
    return num / den


def compute(events, truth, dwells, observations, predictions, emitter_appear, slot_ms,
            pred_tol: int = 2, oracle_weighted_intercepts: float | None = None) -> FigureOfMeritReport:
    n_slots, n_bands = len(dwells), truth.shape[1]
    undefined: dict = {}

    detected: dict[tuple[int, int], set] = {}
    n_present = n_true_det = n_absent = n_fa = 0
    for t, obs in enumerate(observations):
        for b, bo in obs.per_band.items():
            sigs = {d.signature for d in bo.detections}
            if sigs:
                detected[(t, b)] = sigs
            if truth[t, b]:
                n_present += 1
                n_true_det += bool(sigs)
            else:
                n_absent += 1
                n_fa += bo.false_alarm

    intercepted_flags, first_hit = [], {}
    for ev in events:
        hit_slot = None
        for t in range(ev.start, min(ev.end_exclusive, n_slots)):
            if ev.emitter_id in detected.get((t, ev.band), ()):
                hit_slot = t
                break
        intercepted_flags.append(hit_slot is not None)
        if hit_slot is not None:
            first_hit[ev.emitter_id] = min(first_hit.get(ev.emitter_id, hit_slot), hit_slot)

    n_int = sum(intercepted_flags)
    w_all = sum(ev.threat_weight for ev in events)
    w_int = sum(ev.threat_weight for ev, f in zip(events, intercepted_flags) if f)
    ttfi = {eid: first_hit[eid] - emitter_appear.get(eid, 0) for eid in sorted(first_hit)}
    all_emitters = set(emitter_appear) | {ev.emitter_id for ev in events}
    censored = sorted(all_emitters - set(first_hit))

    # Predictions: nearest true event start of that signature at or after the slot it was made.
    starts_by_sig = defaultdict(list)
    for ev in events:
        starts_by_sig[ev.emitter_id].append(ev.start)
    for s in starts_by_sig.values():
        s.sort()
    errors, unmatched = [], []
    for made, sig, pred in predictions:
        s = starts_by_sig.get(sig, [])
        i = bisect.bisect_left(s, made)
        cands = s[i:]
        if not cands:
            unmatched.append((made, sig, pred))
            continue
        j = bisect.bisect_left(cands, pred)
        near = [c for c in (cands[j - 1] if j > 0 else None, cands[j] if j < len(cands) else None) if c is not None]
        true = min(near, key=lambda c: (abs(c - pred), c))
        errors.append(abs(pred - true))
    correct = sum(e <= pred_tol for e in errors)

    visits = defaultdict(list)
    for t, bands in enumerate(dwells):
        for b in bands:
            visits[b].append(t)
    max_revisit = []
    for b in range(n_bands):
        v = [-1] + visits[b] + [n_slots]
        max_revisit.append(int(np.diff(v).max()))

    duration_s = n_slots * slot_ms / 1000.0
    report = FigureOfMeritReport(
        n_slots=n_slots,
        n_events=len(events),
        n_intercepted=n_int,
        interception_ratio=_ratio(n_int, len(events), "interception_ratio", undefined, "no events"),
        weighted_interception_ratio=_ratio(w_int, w_all, "weighted_interception_ratio", undefined, "no event weight"),
        pd_empirical=_ratio(n_true_det, n_present, "pd_empirical", undefined, "no dwell on a present signal"),
        pfa_empirical=_ratio(n_fa, n_absent, "pfa_empirical", undefined, "no dwell on an empty band"),
        intercept_rate_per_s=n_int / duration_s,
        ttfi_slots=ttfi,
        censored=censored,
        mean_ttfi_slots=float(np.mean(list(ttfi.values()))) if ttfi else None,
        reward_total=float(w_int),
        reward_per_slot=float(w_int) / n_slots if n_slots else 0.0,
        cost_dwells=sum(len(d) for d in dwells),
        pred_correct_pct=_ratio(100.0 * correct, len(errors), "pred_correct_pct", undefined, "no matched predictions"),
        mean_abs_intercept_time_error_slots=float(np.mean(errors)) if errors else None,
        mean_abs_intercept_time_error_ms=float(np.mean(errors)) * slot_ms if errors else None,
        pred_matched=len(errors),
        pred_unmatched=unmatched,
        max_revisit_slots=max_revisit,
        oracle_efficiency=(w_int / oracle_weighted_intercepts) if oracle_weighted_intercepts else None,
        event_intercepted=intercepted_flags,
        undefined=undefined,
    )
    if not ttfi:
        undefined["mean_ttfi_slots"] = "no emitter intercepted"
    boundary(log, logging.INFO, "computed", n_slots=n_slots, n_events=len(events), n_intercepted=n_int,
             interception_ratio=report.interception_ratio, censored=len(censored), predictions=len(predictions))
    return report
