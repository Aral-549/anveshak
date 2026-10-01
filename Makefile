# Common tasks. Everything runs through uv, so `make setup` is the only install step.
export OMP_NUM_THREADS ?= 1
UV ?= uv

.PHONY: setup test app app-dev data train eval bench deck

setup:            ## create .venv with all dependencies (Python 3.12)
	$(UV) sync

test:             ## golden tests (~15 s)
	$(UV) run pytest tests/golden -q

app:              ## band-state console on http://localhost:8030
	$(UV) run uvicorn backend.app:create_app --factory --port 8030

app-dev:          ## same, auto-reload, CORS open to a frontend dev server on :3000
	ANVESHAK_CORS_ORIGINS=http://localhost:3000 $(UV) run uvicorn backend.app:create_app --factory --port 8030 --reload

data:             ## regenerate hit-model datasets from the simulator (~3 min)
	$(UV) run python scripts/hit_dataset.py train
	$(UV) run python scripts/hit_dataset.py val
	$(UV) run python scripts/hit_dataset.py eval

train:            ## retrain the hit model and re-pin its hash (needs `make data`)
	$(UV) run python scripts/train_hit_model.py
	sha256sum artifacts/hit_model.json | cut -d' ' -f1 > artifacts/hit_model.sha256

eval:             ## re-score the current model on held-out data without retraining
	$(UV) run python scripts/train_hit_model.py --eval-only

bench:            ## end-to-end benchmark on held-out missions (~10 min on 16 cores)
	$(UV) run python scripts/benchmark.py --seeds 1000-1019 --policies sweep,predictive,hybrid --lam 0.1 --hybrid_lam 0.1 --out artifacts/bench_latest.json

deck:             ## rebuild the SIH deck from the measured artifacts
	$(UV) run python deck/illustration.py 5
	$(UV) run python deck/build_deck.py
	cd deck/out && soffice --headless --convert-to pdf SIH2026_SIH26055_ANVESHAK.pptx
