"""Golden cases for contracts/scan_crud_app.md. Frozen: add cases, never edit."""
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app

FRESH = dict(window=0, window_threat=0, soon=0, time_to_window=1.0, lock_period=0, confirm=0, confirm_threat=0,
             staleness=0.125, novel_mean=0.5, novel_evidence=0.5, band_threat=0.5, occ_mean=0.5, long_here=0,
             since_novel=12)
DUE = dict(window=1, window_threat=0.6, soon=0, time_to_window=0.0, lock_period=0.3, confirm=0, confirm_threat=0,
           staleness=0.25, novel_mean=0.1, novel_evidence=0.02, band_threat=0.6, occ_mean=0.1, long_here=0,
           since_novel=0.3)
BEACON = dict(FRESH, staleness=0.25, novel_mean=0.02, novel_evidence=0.01, band_threat=0.1, occ_mean=0.98, long_here=1)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(db_path=str(tmp_path / "t.db"))) as c:
        yield c


def post(c, band, row, **extra):
    return c.post("/api/band-states", json=dict(row, band=band, **extra))


def test_case1_fresh(client):
    r = post(client, 4, FRESH)
    assert r.status_code == 201
    assert r.json()["probability"] == pytest.approx(0.0157600, abs=1e-6)
    assert r.json()["source"] == "manual"


def test_case2_due(client):
    assert post(client, 7, DUE).json()["probability"] == pytest.approx(0.9697804, abs=1e-6)


def test_case3_beacon(client):
    assert post(client, 2, BEACON).json()["probability"] == pytest.approx(0.0027849, abs=1e-6)


def test_case4_inconsistent_window_threat(client):
    assert post(client, 4, dict(FRESH, window_threat=0.5)).status_code == 422


@pytest.mark.parametrize("bad", [dict(staleness=2.5), dict(novel_evidence=0), dict(window=2)])
def test_case5_out_of_range(client, bad):
    assert post(client, 4, dict(FRESH, **bad)).status_code == 422


def test_case5_negative_band(client):
    assert post(client, -1, FRESH).status_code == 422


def test_case6_missing(client):
    assert client.get("/api/band-states/999999").status_code == 404


def test_case7_update_rescores(client):
    rec = post(client, 4, FRESH).json()
    r = client.put(f"/api/band-states/{rec['id']}", json=dict(DUE, band=4))
    assert r.status_code == 200
    out = r.json()
    assert out["probability"] == pytest.approx(0.9697804, abs=1e-6)
    assert out["source"] == "manual" and out["created_at"] == rec["created_at"]
    assert out["updated_at"] >= out["created_at"]


def test_case8_delete(client):
    rid = post(client, 4, FRESH).json()["id"]
    assert client.delete(f"/api/band-states/{rid}").status_code == 204
    assert client.get(f"/api/band-states/{rid}").status_code == 404
    assert client.delete(f"/api/band-states/{rid}").status_code == 404


def test_case9_and_10_recommendation(client):
    post(client, 4, FRESH)
    post(client, 7, DUE)
    post(client, 2, BEACON)
    assert [x["band"] for x in client.get("/api/recommendation?k=2").json()] == [7, 4]
    post(client, 7, BEACON)
    assert [x["band"] for x in client.get("/api/recommendation?k=1").json()] == [4]


def test_case11_empty_recommendation(client):
    r = client.get("/api/recommendation")
    assert r.status_code == 200 and r.json() == []


def test_case12_and_13_snapshot(client):
    body = dict(suite="S1", seed=1000, slot=500, policy="hybrid")
    a = client.post("/api/band-states/snapshot", json=body)
    assert a.status_code == 201
    recs = a.json()
    assert sorted(x["band"] for x in recs) == list(range(32))
    assert all(x["source"] == "mission S1/1000 @ slot 500 (hybrid)" for x in recs)
    assert all(0 <= x["probability"] <= 1 for x in recs)
    b = client.post("/api/band-states/snapshot", json=body).json()
    strip = lambda rs: sorted((tuple((k, v) for k, v in r.items() if k not in ("id", "created_at", "updated_at")) for r in rs))
    assert strip(recs) == strip(b)


@pytest.mark.parametrize("bad", [dict(suite="S9"), dict(slot=0), dict(slot=12001)])
def test_case14_bad_snapshot(client, bad):
    body = dict(dict(suite="S1", seed=1000, slot=500, policy="hybrid"), **bad)
    assert client.post("/api/band-states/snapshot", json=body).status_code == 422


def test_case15_filter_by_band(client):
    post(client, 7, FRESH)
    post(client, 3, FRESH)
    post(client, 7, DUE)
    rows = client.get("/api/band-states?band=7").json()
    assert [r["band"] for r in rows] == [7, 7] and rows[0]["id"] > rows[1]["id"]


def test_case16_limit_clamped(client):
    post(client, 1, FRESH)
    r = client.get("/api/band-states?limit=5000")
    assert r.status_code == 200 and len(r.json()) == 1


def test_case17_note_stored_verbatim(client):
    note = "<script>x</script>"
    rid = post(client, 1, FRESH, note=note).json()["id"]
    assert client.get(f"/api/band-states/{rid}").json()["note"] == note


def test_edge_startup_fails_on_hash_mismatch(tmp_path, monkeypatch):
    bad = tmp_path / "hash"
    bad.write_text("0" * 64)
    with pytest.raises(RuntimeError):
        create_app(db_path=str(tmp_path / "t.db"), hash_path=str(bad))


def test_edge_separate_dbs(tmp_path):
    with TestClient(create_app(db_path=str(tmp_path / "a.db"))) as a, \
         TestClient(create_app(db_path=str(tmp_path / "b.db"))) as b:
        post(a, 1, FRESH)
        assert b.get("/api/band-states").json() == []


def test_edge_snapshot_at_first_slot(client):
    recs = client.post("/api/band-states/snapshot", json=dict(suite="S2", seed=5, slot=1)).json()
    assert len(recs) == 32


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_regression_non_finite_body_is_422(client, token):
    """BUGLOG 2026-10-02: a non-finite number must give 422, not a 500 from the error encoder."""
    import json
    body = json.dumps(dict(FRESH, band=4)).replace('"staleness": 0.125', f'"staleness": {token}')
    r = client.post("/api/band-states", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "staleness" in r.text


def _preflight(c):
    return c.options("/api/band-states", headers={"Origin": "http://localhost:3000",
                                                  "Access-Control-Request-Method": "POST"})


def test_case18_cors_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVESHAK_CORS_ORIGINS", "http://localhost:3000")
    with TestClient(create_app(db_path=str(tmp_path / "c.db"))) as c:
        r = _preflight(c)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_case19_cors_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ANVESHAK_CORS_ORIGINS", raising=False)
    with TestClient(create_app(db_path=str(tmp_path / "c.db"))) as c:
        assert "access-control-allow-origin" not in _preflight(c).headers


def test_case20_import_has_no_side_effects(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    db = root / "artifacts" / "scan_app.db"
    existed = db.exists()
    subprocess.run([sys.executable, "-c", "import backend.app"], cwd=root, check=True)
    assert db.exists() == existed
