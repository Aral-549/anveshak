# Contract: scan_crud_app

## Purpose
A small web app around the ANVESHAK hit model (`artifacts/hit_model.json`,
`contracts/hit_model.md`): create, read, update and delete **band-state
records** (one band's 14 belief features), score each record with the model on
create and update, snapshot all bands of a simulated mission into records, and
rank bands to recommend where the receiver should dwell next. Training the model
is out of scope (`scripts/train_hit_model.py`).

Revised 2026-10-01: the first draft served `smart_scan_xgboost.pkl`; that model
scored at chance on simulator data (AUC-PR 0.010 vs 0.0088 no-skill) and was
replaced by the hit model (0.135) at the user's request. The app never
unpickles anything.

## Model pinned by this contract
- `artifacts/hit_model.json`, sha256
  `ddc7a3a62a36120b715049933d979fa8ffd51f09abdab181f9482348154468b9`,
  loaded through `core.hit_model.HitModel` (feature names checked at load).
- The app refuses to start if the file is missing or its hash differs from the
  one in `artifacts/hit_model.sha256` (written alongside the model).

## Inputs
- `BandStateIn` (create / update body):
  - `band`: int, 0 <= band <= 255
  - `window`, `soon`, `confirm`, `long_here`: int in {0, 1}
  - `window_threat`: float in [0, 1]; must be 0 when `window` = 0
  - `confirm_threat`: float in [0, 1]; must be 0 when `confirm` = 0
  - `time_to_window`: float in [-0.025, 1]
  - `lock_period`: float in [0, 2]
  - `staleness`: float in [0, 2]
  - `novel_mean`, `band_threat`, `occ_mean`: float in [0, 1]
  - `novel_evidence`: float in (0, 1]
  - `since_novel`: float in [0, 12]
  - `note`: optional str, at most 500 chars
- `SnapshotIn`: `suite` in {S1..S5}, `seed` int in [0, 100000],
  `slot` int in [1, 12000], `policy` in {sweep, predictive, hybrid}
  (default hybrid).

## Outputs
- `BandStateOut`: `id`, every `BandStateIn` field, `probability` (float in
  [0, 1]), `source` (`"manual"` or `"mission <suite>/<seed> @ slot <slot> (<policy>)"`),
  `model_sha256`, `created_at`, `updated_at` (ISO-8601 UTC).
- Persistence: SQLite file `artifacts/scan_app.db` (path overridable with env
  `SCAN_APP_DB`), one table `band_states`.

## Endpoints
| Method | Path | Behaviour |
|---|---|---|
| POST | `/api/band-states` | validate, score, store; 201 + record |
| GET | `/api/band-states` | list, newest first; optional `?band=`, `?source=`, `?limit=` (default 100, clamped to 1..1000) |
| GET | `/api/band-states/{id}` | one record; 404 if missing |
| PUT | `/api/band-states/{id}` | full replace of the fields, re-score, bump `updated_at`; keeps `created_at` and `source`; 404 if missing |
| DELETE | `/api/band-states/{id}` | 204; 404 if missing |
| POST | `/api/band-states/snapshot` | run the simulator to `slot` with `policy`, store one record per band from the belief at that slot; 201 + list |
| GET | `/api/recommendation?k=3` | latest record (highest id) per band, top-k by probability, ties to the lower band; empty list when no records |
| GET | `/api/health` | model sha256, feature names, record count |
| GET | `/` | single-page UI: create/edit form, records table with edit/delete, snapshot form, recommendation panel |

## Behavior cases (input -> expected output)
Reference rows (all other features as listed):
- FRESH: window 0, window_threat 0, soon 0, time_to_window 1.0, lock_period 0,
  confirm 0, confirm_threat 0, staleness 0.125, novel_mean 0.5,
  novel_evidence 0.5, band_threat 0.5, occ_mean 0.5, long_here 0, since_novel 12
- DUE: window 1, window_threat 0.6, soon 0, time_to_window 0.0,
  lock_period 0.3, confirm 0, confirm_threat 0, staleness 0.25,
  novel_mean 0.1, novel_evidence 0.02, band_threat 0.6, occ_mean 0.1,
  long_here 0, since_novel 0.3
- BEACON: as FRESH but staleness 0.25, novel_mean 0.02, novel_evidence 0.01,
  band_threat 0.1, occ_mean 0.98, long_here 1

| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | POST FRESH band 4 | 201; probability 0.0157600 (±1e-6); source "manual" | |
| 2 | POST DUE band 7 | probability 0.9697804 (±1e-6) | a locked radar due now |
| 3 | POST BEACON band 2 | probability 0.0027849 (±1e-6) | always-on benign emitter |
| 4 | POST FRESH with window 0 and window_threat 0.5 | 422 | inconsistent |
| 5 | POST with staleness 2.5, or novel_evidence 0, or band -1, or window 2 | 422 | |
| 6 | GET /api/band-states/999999 | 404 | |
| 7 | PUT case-1 record with the DUE fields | 200; probability 0.9697804; source still "manual"; created_at unchanged; updated_at >= created_at | re-scored |
| 8 | DELETE an id, then GET it; DELETE it again | 204, then 404, then 404 | |
| 9 | records: band 4 FRESH, band 7 DUE, band 2 BEACON; GET /api/recommendation?k=2 | bands [7, 4] in that order | |
| 10 | then POST band 7 BEACON (newer); GET /api/recommendation?k=1 | band 4 | only the latest record per band counts |
| 11 | GET /api/recommendation on an empty table | 200, [] | |
| 12 | POST snapshot S1 / seed 1000 / slot 500 / hybrid | 201; 32 records, bands 0..31, all source "mission S1/1000 @ slot 500 (hybrid)"; every probability in [0, 1] | |
| 13 | the same snapshot twice | identical feature values and probabilities | deterministic |
| 14 | POST snapshot with suite "S9" or slot 0 or slot 12001 | 422 | |
| 15 | GET /api/band-states?band=7 | only band-7 records, newest first | |
| 16 | GET /api/band-states?limit=5000 | at most 1000 records, no error | clamped |
| 17 | POST with note `<script>x</script>` | stored verbatim; the UI renders it as text | no injection |

## Edge cases that must be covered
- Start-up fails loudly (RuntimeError) if the model file is missing, its feature
  names differ, or its sha256 differs from `artifacts/hit_model.sha256`.
- Two apps on different DB paths do not share records (tests use temp DBs).
- Snapshot of a slot before any visit still returns 32 finite records.
- Non-finite numbers in a request body (`NaN`, `Infinity`) -> 422 with a
  readable error, never a 500 (added after BUGLOG 2026-10-02).

## Configuration
- Served with an app factory (`uvicorn backend.app:create_app --factory`);
  importing `backend.app` has no side effects (no DB file is created).
- `ANVESHAK_CORS_ORIGINS`: comma-separated origins allowed to call the API from
  a separate frontend dev server. Unset means no CORS headers (same-origin only).

| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 18 | `ANVESHAK_CORS_ORIGINS=http://localhost:3000`, preflight `OPTIONS /api/band-states` from that origin | 200 with `access-control-allow-origin: http://localhost:3000` | |
| 19 | variable unset, same request | no `access-control-allow-origin` header | |
| 20 | `import backend.app` | no `artifacts/scan_app.db` created | no import side effects |

## Logging (AGENTS.md rule 5)
JSON log line at each boundary: request validated -> features built -> model
scored -> row written / deleted, with record id, band and probability; snapshot
logs suite, seed, slot, policy and record count.

## Explicitly out of scope
- Training or retraining the model.
- Authentication, multi-user access, deployment.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract
- [x] Golden tests exist for every behavior case above (case 17's "renders as
  text" checked by screenshot: the UI builds every cell with `textContent`)
