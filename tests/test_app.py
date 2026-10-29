import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import runner, server

SPEC = {"home": "CHI", "away": "GB", "out_home": [], "out_away": [], "seed": 5, "team_scale": 1.5, "player_scale": 1.5}


def finish(games, workers):
    run = runner.MillionRun(SPEC, games, workers)
    run.start()
    while run.snapshot()["running"]:
        time.sleep(0.2)
    return run.snapshot()


def test_wilson_band_shrinks_and_contains_the_estimate():
    lo1, hi1 = runner.wilson_band(0.55, 100)
    lo2, hi2 = runner.wilson_band(0.55, 10000)
    assert lo1 < 0.55 < hi1 and lo2 < 0.55 < hi2
    assert (hi2 - lo2) < (hi1 - lo1) / 5
    assert runner.wilson_band(0.5, 0) == (0.0, 1.0)


def test_thin_keeps_the_last_point():
    rows = list(range(1000))
    out = runner.thin(rows, 100)
    assert len(out) <= 101 and out[-1] == 999 and out[0] == 0


def test_a_run_gives_the_same_answer_with_any_number_of_workers():
    a, b = finish(200, 1), finish(200, 2)
    assert a["games_done"] == b["games_done"] == 200
    for key in ("home_win_prob", "mean_margin", "margin_sd"):
        assert a[key] == b[key]
    assert a["teams"] == b["teams"]


def test_a_run_reports_players_and_team_totals():
    s = finish(100, 2)
    assert 0 < s["home_win_prob"] < 1 and s["ci_low"] < s["home_win_prob"] < s["ci_high"]
    assert 10 < s["teams"][0]["points"] < 40
    assert len(s["boxes"]) == 2 and s["boxes"][0]["passing"] and s["boxes"][0]["receiving"]
    assert not s["running"]
    assert sum(c[2] for c in s["scores"]) == 100 and all(c[0] >= 0 and c[1] >= 0 for c in s["scores"])
    assert sum(c[2] for c in s["scores"] if c[0] > c[1]) + 0.5 * sum(c[2] for c in s["scores"] if c[0] == c[1]) == s["home_win_prob"] * 100


def test_the_typical_game_is_the_finished_game_closest_to_the_average():
    run = runner.MillionRun(SPEC, 300, 2)
    run.start()
    while run.snapshot()["running"]:
        time.sleep(0.2)
    index = run.typical_index()
    assert 0 <= index < 300 and index == run.typical_index()
    rows = np.vstack([r for _, r in sorted(run.prints, key=lambda x: x[0])])
    spread = rows.std(axis=0)
    distance = (((rows - rows.mean(axis=0)) / spread) ** 2).sum(axis=1)
    assert index == int(np.argmin(distance))
    assert distance[index] < np.median(distance)


def test_a_stopped_run_stops():
    run = runner.MillionRun(SPEC, 100000, 2)
    run.start()
    time.sleep(2)
    run.stop()
    n = run.snapshot()["games_done"]
    time.sleep(1)
    assert run.snapshot()["games_done"] == n and n < 100000


def test_the_server_answers():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        return json.loads(urllib.request.urlopen(base + path).read())

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    page = urllib.request.urlopen(base + "/").read()
    assert b"<title>Football Sim</title>" in page and b"bigMoments" in page and b'id="tip"' in page
    assert len(get("/api/teams")) == 32
    lines = get("/api/lines")
    assert len(lines) == 32 and lines[0]["rank"] == 1 and lines[0]["score"] >= lines[-1]["score"]
    lineup = get("/api/lineup?team=CHI")
    assert lineup["active"][0]["position"] == "QB"
    assert all("out" in p and "role" in p for p in lineup["active"]) and lineup["as_of"]
    post("/api/run", {"home": "CHI", "away": "GB", "games": 200, "seed": 7})
    while get("/api/run/status")["running"]:
        time.sleep(0.2)
    typical = get("/api/run/typical")
    assert typical["final"] == [typical["events"][-1]["home"], typical["events"][-1]["away"]]
    assert typical["of"] == 200 and 0 <= typical["index"] < 200 and typical["boxes"][0]["passing"]
    game = post("/api/game", {"home": "CHI", "away": "GB", "seed": 3})
    assert game["events"][0]["text"] == "Kickoff" and game["events"][-1]["p"] in (0.0, 0.5, 1.0)
    assert [r["name"] for r in game["boxes"][0]["passing"]][0] == lineup["active"][0]["name"]
    assert game["final"] == [game["events"][-1]["home"], game["events"][-1]["away"]]
    assert all(0 <= e["p"] <= 1 for e in game["events"])
    assert "seasons" in get("/api/validation")
    httpd.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All app checks passed.")
