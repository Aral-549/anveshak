# Bug Log

Every entry here must result in a permanent case added to `tests/golden/`
before it's marked resolved. A patched bug without a regression case is not
resolved - it's just hidden until the next rewrite.

---

## 2026-10-01 — Hit-model evaluation scored a different model than the one deployed
- **Symptom:** `artifacts/hit_model_eval.json` reported AUC-PR 0.1354, but the
  scheduler, benchmarks and deck use `HitModel.score`, which gives 0.1351 on
  the same rows.
- **Root cause:** `scripts/train_hit_model.py` scored with
  `iteration_range=(0, best_iteration + 1)` (301 trees), while the saved
  `artifacts/hit_model.json` holds all 361 trees from early stopping and
  `HitModel.score` uses every tree. The ranking result therefore described a
  slightly different model.
- **Stage/module:** training/evaluation script vs `core/hit_model.HitModel` (serving path).
- **Fix:** evaluation now loads the saved file through `HitModel.load(...).score`
  and records `n_trees_scored`; results regenerated with `--eval-only`
  (0.1351, still 15x chance; deck figures unchanged at their displayed precision).
- **Regression case added:** `tests/golden/test_hit_model_golden.py` —
  `test_regression_eval_scores_deployed_model`
- **Status:** fixed / verified

---

## 2026-10-02 — CRUD app returns 500 on NaN / Infinity in a request body
- **Symptom:** `POST /api/band-states` with `"staleness": NaN` (valid for
  Python's JSON parser) raised `ValueError: Out of range float values are not JSON
  compliant` and the client got HTTP 500 instead of 422. Found in the adversarial pass.
- **Root cause:** validation correctly rejected the value, but FastAPI's default
  422 handler echoes each offending `input` back in the response, and the
  JSON encoder refuses to serialise NaN/inf.
- **Stage/module:** `backend/app.py`, request validation -> error response.
- **Fix:** a custom `RequestValidationError` handler that converts non-finite
  floats in the error detail to strings before encoding.
- **Regression case added:** `tests/golden/test_scan_crud_app_golden.py` —
  `test_regression_non_finite_body_is_422`
- **Status:** fixed / verified
