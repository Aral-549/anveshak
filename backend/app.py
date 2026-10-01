"""ANVESHAK band-state CRUD app around the hit model. See contracts/scan_crud_app.md.

    uv run uvicorn backend.app:create_app --factory --port 8030      # then open http://localhost:8030

Set ANVESHAK_CORS_ORIGINS=http://localhost:3000 (comma-separated) to call the API from a separate
frontend dev server.
"""
import hashlib
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from core import scenarios
from core.belief import Belief, BeliefConfig
from core.environment import build
from core.hit_model import HIT_FEATURES, HitModel, hit_features
from core.log import boundary, configure, get_logger
from core.receiver import Receiver
from core.scheduler import Scheduler
from core.sim import RunConfig

ROOT = Path(__file__).resolve().parent.parent
log = get_logger("crud_app")
LIMIT_MAX = 1000


class BandStateIn(BaseModel):
    band: int = Field(ge=0, le=255)
    window: Literal[0, 1]
    window_threat: float = Field(ge=0, le=1)
    soon: Literal[0, 1]
    time_to_window: float = Field(ge=-0.025, le=1)
    lock_period: float = Field(ge=0, le=2)
    confirm: Literal[0, 1]
    confirm_threat: float = Field(ge=0, le=1)
    staleness: float = Field(ge=0, le=2)
    novel_mean: float = Field(ge=0, le=1)
    novel_evidence: float = Field(gt=0, le=1)
    band_threat: float = Field(ge=0, le=1)
    occ_mean: float = Field(ge=0, le=1)
    long_here: Literal[0, 1]
    since_novel: float = Field(ge=0, le=12)
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _consistent(self):
        if self.window == 0 and self.window_threat != 0:
            raise ValueError("window_threat must be 0 when window is 0")
        if self.confirm == 0 and self.confirm_threat != 0:
            raise ValueError("confirm_threat must be 0 when confirm is 0")
        return self


class BandStateOut(BandStateIn):
    id: int
    probability: float
    source: str
    model_sha256: str
    created_at: str
    updated_at: str


class SnapshotIn(BaseModel):
    suite: Literal["S1", "S2", "S3", "S4", "S5"]
    seed: int = Field(ge=0, le=100_000)
    slot: int = Field(ge=1, le=scenarios.N_SLOTS)
    policy: Literal["sweep", "predictive", "hybrid"] = "hybrid"


COLUMNS = ["band", *HIT_FEATURES, "note", "probability", "source", "model_sha256", "created_at", "updated_at"]


def _json_safe(x):
    """Make validation-error detail encodable: NaN/inf become strings (BUGLOG 2026-10-02)."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _load_model(model_path: Path, hash_path: Path) -> tuple[HitModel, str]:
    if not model_path.exists():
        raise RuntimeError(f"model file missing: {model_path}")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected = hash_path.read_text().strip() if hash_path.exists() else None
    if digest != expected:
        raise RuntimeError(f"model hash {digest} does not match {hash_path} ({expected})")
    try:
        return HitModel.load(model_path), digest
    except ValueError as e:
        raise RuntimeError(str(e)) from e


def create_app(db_path: str | None = None, model_path: str | None = None, hash_path: str | None = None) -> FastAPI:
    model, digest = _load_model(Path(model_path or ROOT / "artifacts/hit_model.json"),
                                Path(hash_path or ROOT / "artifacts/hit_model.sha256"))
    db_file = db_path or os.environ.get("SCAN_APP_DB") or str(ROOT / "artifacts/scan_app.db")
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    lock = threading.Lock()
    feature_cols = ", ".join(f"{f} REAL NOT NULL" for f in HIT_FEATURES)
    with lock, conn:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS band_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT, band INTEGER NOT NULL, {feature_cols}, note TEXT,
            probability REAL NOT NULL, source TEXT NOT NULL, model_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    configure(logging.INFO)
    app = FastAPI(title="ANVESHAK band-state console", version="1.0")
    origins = [o.strip() for o in os.environ.get("ANVESHAK_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        detail = _json_safe(list(exc.errors()))
        boundary(log, logging.INFO, "rejected", path=request.url.path, errors=len(detail))
        return JSONResponse(status_code=422, content={"detail": detail})

    def score_rows(rows: np.ndarray) -> np.ndarray:
        return np.clip(model.score(rows), 0.0, 1.0)

    def to_out(row: sqlite3.Row) -> dict:
        d = dict(row)
        for f in ("window", "soon", "confirm", "long_here"):
            d[f] = int(d[f])
        return d

    def fetch(rid: int) -> dict:
        with lock:
            row = conn.execute("SELECT * FROM band_states WHERE id = ?", (rid,)).fetchone()
        if row is None:
            raise HTTPException(404, f"band state {rid} not found")
        return to_out(row)

    def insert(records: list[dict]) -> list[dict]:
        ts = _now()
        ids = []
        with lock, conn:
            for r in records:
                vals = [r[c] if c in r else None for c in COLUMNS[:-2]] + [ts, ts]
                cur = conn.execute(f"INSERT INTO band_states ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                                   vals)
                ids.append(cur.lastrowid)
        return [fetch(i) for i in ids]

    @app.post("/api/band-states", response_model=BandStateOut, status_code=201)
    def create(body: BandStateIn):
        rec = body.model_dump()
        x = np.array([[rec[f] for f in HIT_FEATURES]], dtype=float)
        boundary(log, logging.INFO, "validated", band=rec["band"])
        rec.update(probability=float(score_rows(x)[0]), source="manual", model_sha256=digest)
        out = insert([rec])[0]
        boundary(log, logging.INFO, "written", id=out["id"], band=out["band"], probability=out["probability"])
        return out

    @app.get("/api/band-states", response_model=list[BandStateOut])
    def list_states(band: int | None = None, source: str | None = None, limit: int = Query(100)):
        limit = min(max(limit, 1), LIMIT_MAX)
        sql, args = "SELECT * FROM band_states WHERE 1=1", []
        if band is not None:
            sql, args = sql + " AND band = ?", args + [band]
        if source is not None:
            sql, args = sql + " AND source = ?", args + [source]
        with lock:
            rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
        return [to_out(r) for r in rows]

    @app.get("/api/band-states/{rid}", response_model=BandStateOut)
    def read(rid: int):
        return fetch(rid)

    @app.put("/api/band-states/{rid}", response_model=BandStateOut)
    def update(rid: int, body: BandStateIn):
        fetch(rid)
        rec = body.model_dump()
        prob = float(score_rows(np.array([[rec[f] for f in HIT_FEATURES]], dtype=float))[0])
        sets = ", ".join(f"{c} = ?" for c in ["band", *HIT_FEATURES, "note", "probability", "model_sha256", "updated_at"])
        vals = [rec["band"], *[rec[f] for f in HIT_FEATURES], rec["note"], prob, digest, _now(), rid]
        with lock, conn:
            conn.execute(f"UPDATE band_states SET {sets} WHERE id = ?", vals)
        boundary(log, logging.INFO, "updated", id=rid, band=rec["band"], probability=prob)
        return fetch(rid)

    @app.delete("/api/band-states/{rid}", status_code=204)
    def delete(rid: int):
        with lock, conn:
            n = conn.execute("DELETE FROM band_states WHERE id = ?", (rid,)).rowcount
        if n == 0:
            raise HTTPException(404, f"band state {rid} not found")
        boundary(log, logging.INFO, "deleted", id=rid)
        return Response(status_code=204)

    @app.post("/api/band-states/snapshot", response_model=list[BandStateOut], status_code=201)
    def snapshot(body: SnapshotIn):
        X = _mission_features(body, model)
        probs = score_rows(X)
        source = f"mission {body.suite}/{body.seed} @ slot {body.slot} ({body.policy})"
        recs = []
        for b in range(X.shape[0]):
            rec = {f: float(X[b, i]) for i, f in enumerate(HIT_FEATURES)}
            for f in ("window", "soon", "confirm", "long_here"):
                rec[f] = int(rec[f])
            rec.update(band=b, note=None, probability=float(probs[b]), source=source, model_sha256=digest)
            recs.append(rec)
        out = insert(recs)
        boundary(log, logging.INFO, "snapshot", suite=body.suite, seed=body.seed, slot=body.slot, policy=body.policy,
                 records=len(out))
        return out

    @app.get("/api/recommendation", response_model=list[BandStateOut])
    def recommendation(k: int = Query(3, ge=1, le=256)):
        with lock:
            rows = conn.execute("""SELECT * FROM band_states WHERE id IN (SELECT MAX(id) FROM band_states GROUP BY band)
                                   ORDER BY probability DESC, band ASC LIMIT ?""", (k,)).fetchall()
        return [to_out(r) for r in rows]

    @app.get("/api/health")
    def health():
        with lock:
            n = conn.execute("SELECT COUNT(*) FROM band_states").fetchone()[0]
        return dict(model_sha256=digest, features=HIT_FEATURES, records=n)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app


def _mission_features(body: SnapshotIn, model: HitModel) -> np.ndarray:
    """Run a simulated mission up to `slot` and return every band's hit-model features at that slot."""
    cfg = RunConfig(policy=body.policy, lam=0.1, hybrid_lam=0.1, seed=body.seed)
    spec = scenarios.make(body.suite, body.seed)
    env = build(spec)
    rx = Receiver(spec.n_bands, cfg.M, cfg.pfa, cfg.threat_hint_noise,
                  np.random.default_rng([spec.seed, cfg.seed, 7]))
    belief = Belief(BeliefConfig(n_bands=spec.n_bands))
    score_fn = model.score_fn(cfg.T_max) if body.policy == "hybrid" else None
    sched = Scheduler(cfg.policy, spec.n_bands, cfg.M, cfg.T_max, seed=cfg.seed, lam=cfg.lam, score_fn=score_fn,
                      hybrid_lam=cfg.hybrid_lam)
    for t in range(body.slot):
        plan = sched.plan(t, belief)
        belief.update(rx.observe(t, plan.bands, env.band_emitters(t, plan.bands)))
    X = hit_features(body.slot, belief, cfg.T_max)
    assert np.isfinite(X).all()
    return X

