# ANVESHAK

Smart scan scheduler for an Electronic Support (ES) receiver. Smart India Hackathon 2026, problem statement SIH26055 (DRDO). Team Evinco.

An ES receiver can only listen to one slice of the spectrum at a time. ANVESHAK decides which slice to listen to next. It learns from its own hits and misses, measures each radar's scan period, dwells exactly when a beam is due back, and never leaves any band unwatched for more than 1.28 s.

Everything here runs on a simulator that has ground truth for every band in every 10 ms slot, so you don't need any radio hardware.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and `make`. uv installs Python 3.12 and every dependency into `.venv`.

```bash
git clone <repo-url> && cd anveshak
make setup      # uv sync
make test       # 156 golden tests, about 15 s
make app        # console on http://localhost:8030
```

Without make: `uv sync`, then `uv run pytest tests/golden -q`, then `uv run uvicorn backend.app:create_app --factory --port 8030`.

## Where things are

| Path | What it is |
|---|---|
| `core/` | The simulator and scheduler: emitters, receiver, belief state, scheduling policies, metrics, the hit model |
| `backend/app.py` | FastAPI app: CRUD for band-state records, scoring, mission snapshots, recommendations |
| `backend/static/index.html` | The current frontend: one HTML file, plain JS, no build step |
| `scripts/` | Dataset generation, model training, benchmark |
| `artifacts/` | The trained model (`hit_model.json` + its sha256) and every measured result the deck uses |
| `contracts/` | One spec per module: inputs, outputs, and input-to-expected-output cases. Read the matching one before changing a module |
| `tests/golden/` | Tests written from the contracts. Add cases; don't edit or delete existing ones |
| `BUGLOG.md` | Every bug found so far, its root cause, and the regression test that pins it |
| `deck/` | Builds the SIH deck from the artifacts (`make deck`) |
| `DESIGN.md` | The full design: problem, system model, scheduler, evaluation plan |
| `AGENTS.md` | Workflow rules for people and AI coding agents. It's short, read it once |

## If you're working on the model

The model is an XGBoost classifier in `core/hit_model.py`. For each band at each slot it predicts the chance that listening there now catches a new threat signal. Its 14 inputs come only from what the receiver has observed. The training labels come from simulator truth. The spec, with current results, is `contracts/hit_model.md`.

The loop:

```bash
make data    # build train/val/eval sets from simulated missions (about 3 min, ~120 MB, not in git)
make train   # train, save artifacts/hit_model.json, re-pin its sha256
make eval    # re-score the saved model on held-out data
make bench   # run it as a scheduler on held-out missions (about 10 min on 16 cores)
```

Rules that keep the numbers honest:

- Train and tune on seeds 0-999. Seeds 1000 and up are held out for reporting. `assert_train_seeds` enforces this, so please don't work around it.
- "Better" is measured two ways: ranking quality on held-out rows (AUC-PR, where no-skill is 0.0088), and the threat-weighted interception ratio when the model drives the scheduler (`make bench`). A model can improve the first and not the second. That already happened once. The model on its own ranks well but undervalues camping on a band after a first hit, which is why the shipped policy, `hybrid`, keeps the rules for timing and uses the model only to choose where to search.
- If you change the features, edit `HIT_FEATURES`, `hit_features()` and the contract together, then retrain. The app refuses to load a model whose feature list or hash doesn't match.

Current numbers (held-out missions):

| | Value |
|---|---|
| Hit model AUC-PR | 0.135 (chance 0.0088) |
| Interception ratio, open-loop sweep | 0.084 |
| Interception ratio, rules only | 0.381 |
| Interception ratio, hybrid (rules + model) | 0.402 |

Hybrid clearly beats rules alone on frequency-hopping and mixed missions (p < 0.001 on each). On radar-only missions the difference isn't significant. Good next steps: a model that values camping (multi-step targets or RL), faster first intercept of pop-up radars, and results with two receivers.

## If you're working on the frontend

The UI is `backend/static/index.html`, served at `/`. It's plain JS talking to the API below, and you can rebuild it in whatever framework you like.

To run a separate dev server, such as Next.js on port 3000, start the API with CORS open to it:

```bash
make app-dev    # API on :8030 with reload, CORS allows http://localhost:3000
```

Interactive API docs are at http://localhost:8030/docs.

| Method | Path | What it does |
|---|---|---|
| POST | `/api/band-states` | Create a record. The model scores it |
| GET | `/api/band-states?band=&source=&limit=` | List records, newest first |
| GET / PUT / DELETE | `/api/band-states/{id}` | Read, update (re-scores), delete |
| POST | `/api/band-states/snapshot` | Run a simulated mission to a slot and store all 32 bands |
| GET | `/api/recommendation?k=3` | Latest record per band, ranked by probability |
| GET | `/api/health` | Model hash, feature names, record count |

Validation errors come back as 422 with a `detail` list (field and message). The exact field ranges are in `contracts/scan_crud_app.md`.

The most valuable missing piece is a live waterfall view: bands on one axis, time on the other, showing where the receiver listened, the true signals, and the intercepts, with the plain sweep and ANVESHAK side by side on the same mission. The deck has a static version (`deck/waterfall.png`, made by `deck/illustration.py`). A new endpoint that returns one mission's dwells and truth for a time window would feed it. Write that endpoint's contract cases first.

## How we work

The short version of `AGENTS.md`:

1. Before changing a module, read its contract in `contracts/`. For something new, write the contract (input to expected-output cases) first.
2. Don't write the tests for code you just wrote in the same sitting. Take a separate pass whose goal is to break it.
3. `tests/golden/` is frozen. Add cases; don't edit or delete existing ones.
4. A bug isn't fixed until it's in `BUGLOG.md` with a regression test.
5. No emojis anywhere.

## Known gaps

- Simulated data only. Replaying real spectrum recordings (RTL-SDR or ElectroSense) was planned and isn't built.
- The receiver is handed each signal's emitter ID and threat level. Separating pulses into emitters and assessing threat are assumed, not built.
- Benchmarks use one receiver.
- Not deployed anywhere yet.
