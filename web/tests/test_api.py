import json
from pathlib import Path

from fastapi.testclient import TestClient

from web.api.main import app

_ROOT = Path(__file__).resolve().parents[2]
_P01 = _ROOT / "task" / "data" / "P01_intro.json"


def test_p01_lists_ids_and_moves_orbits():
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    loaded = client.post("/api/scenario", json=data)
    assert loaded.status_code == 200
    body = loaded.json()
    assert set(body["satellites"]) == {sat["id"] for sat in data["satellites"]}
    assert set(body["jobs"]) == {job["id"] for job in data["jobs"]}
    assert body["scenario_id"] == data["meta"]["id"]

    at_0 = client.get("/api/orbits", params={"step": 0})
    at_10 = client.get("/api/orbits", params={"step": 10})
    assert at_0.status_code == 200
    assert at_10.status_code == 200
    sats0 = at_0.json()["satellites"]
    sats10 = at_10.json()["satellites"]
    assert len(sats0) == len(data["satellites"])
    assert len(sats10) == len(data["satellites"])
    assert [row["id"] for row in sats0] == [sat["id"] for sat in data["satellites"]]
    assert (sats0[0]["lat"], sats0[0]["lon"]) != (sats10[0]["lat"], sats10[0]["lon"])
    assert len(sats0[0]["orbit"]) == 64
    assert len(at_0.json()["sun_ecef"]) == 3


def test_tracks_stream_matches_steps():
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    assert client.post("/api/scenario", json=data).status_code == 200
    raw = client.get("/api/tracks")
    assert raw.status_code == 200
    rows = [json.loads(line) for line in raw.text.splitlines() if line.strip()]
    frames = [row for row in rows if row["type"] == "frame"]
    assert [row["step"] for row in frames] == list(range(data["time"]["steps"] + 1))
    assert len(frames[0]["pos"]) == len(data["satellites"])
    assert frames[0]["pos"][0][:2] != frames[10]["pos"][0][:2]
    orbits = next(row for row in rows if row["type"] == "orbits")
    assert len(orbits["path"]) == len(data["satellites"])
    assert len(orbits["path"][0]) == 64

    at_0 = client.get("/api/orbits", params={"step": 0}).json()["satellites"][0]
    at_10 = client.get("/api/orbits", params={"step": 10}).json()["satellites"][0]
    assert abs(frames[0]["pos"][0][0] - at_0["lat"]) < 0.02
    assert abs(frames[0]["pos"][0][1] - at_0["lon"]) < 0.02
    assert abs(frames[10]["pos"][0][0] - at_10["lat"]) < 0.02
    assert abs(frames[10]["pos"][0][1] - at_10["lon"]) < 0.02
