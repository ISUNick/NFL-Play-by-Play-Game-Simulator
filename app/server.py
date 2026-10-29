import argparse
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import update
from app import runner
from app.box import build_box
from engine import winprob
from engine.game import Game
from engine.tables import Tables
from engine.teams import ALL_TEAMS, TeamModel
from players.lineup import Ratings, UNITS, matchup, unit_grades

STATIC = Path(__file__).resolve().parent / "static"
BACKTEST_DIR = ROOT / "data" / "backtest"
TEAM_SCALE = PLAYER_SCALE = 1.5
state = {"run": None, "tables": None, "wp": None}
lock = threading.Lock()


def shared():
    with lock:
        if state["tables"] is None:
            state["tables"], state["wp"] = Tables(), winprob.WinProb()
            state["ratings"] = Ratings.load().scaled(TEAM_SCALE, PLAYER_SCALE)
            state["lineup"] = runner.load_lineup()
            state["model"] = TeamModel()
    return state


def spec_from(body: dict) -> dict:
    return {"home": body["home"], "away": body["away"], "out_home": body.get("out_home", []),
            "out_away": body.get("out_away", []), "play_home": body.get("play_home", []),
            "play_away": body.get("play_away", []), "seed": int(body.get("seed", 1)),
            "swap_home": swaps(body.get("swap_home")), "swap_away": swaps(body.get("swap_away")),
            "team_scale": TEAM_SCALE, "player_scale": PLAYER_SCALE}


def swaps(chosen) -> dict:
    return {unit: team for unit, team in (chosen or {}).items() if unit in UNITS and team}


def teams() -> list:
    return [{"abbr": t.abbr, "name": t.name} for t in sorted(ALL_TEAMS, key=lambda t: t.name)]


def as_of() -> str:
    # the week the ratings and injury report were built from. Saved with the ratings so a fresh clone has it too
    priors = ROOT / "data" / "players" / "priors.json"
    if priors.exists():
        saved = json.loads(priors.read_text()).get("as_of")
        if saved:
            return saved
    files = sorted((ROOT / "data" / "raw").glob("injuries_*.parquet"))
    if not files:
        return ""
    d = pd.read_parquet(files[-1], columns=["season", "week"])
    return f"{int(d.season.max())} week {int(d.week.max())}"


def lineup(team: str) -> dict:
    # every skill player on the roster who can take snaps. Players the injury report lists out come pre checked
    s = shared()
    t = s["lineup"][s["lineup"].team == team]
    skill = t[t.position.isin(["QB", "RB", "WR", "TE"]) & ((t.snap_share > 0.03) | t.out)]
    skill = skill.assign(order=skill.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3}))
    skill = skill.sort_values(["order", "out", "snap_share"], ascending=[True, True, False])
    rows = []
    for r in skill.itertuples():
        reason = ""
        if r.out:
            reason = r.injury_status if isinstance(r.injury_status, str) else {"RES": "injured reserve", "INA": "inactive"}.get(r.status, "out")
        rows.append({"name": r.full_name, "position": r.position, "out": bool(r.out), "reason": reason,
                     "role": f"{100 * r.snap_share:.0f}% of snaps"})
    unmodeled = t[~t.position.isin(["QB", "RB", "WR", "TE"]) & t.out & t.position.isin(["OL", "DL", "LB", "DB"])]
    return {"active": rows, "unmodeled_out": [f"{r.full_name} ({r.position})" for r in unmodeled.itertuples()],
            "as_of": as_of()}


def game_events(s: dict, game, pregame: float, label, rollouts: int = 0, exact_every: int = 0) -> list:
    events = [{"text": "Kickoff", "home": 0, "away": 0, "left": 3600, "p": s["wp"].home(game, pregame)}]
    i = 0
    while not (game.over and not game._queue):
        ev = game.step()
        i += 1
        done = game.over and not game._queue
        p = (1.0 if game.scores[0] > game.scores[1] else 0.5 if game.scores[0] == game.scores[1] else 0.0) if done \
            else s["wp"].home(game, pregame)
        row = {"text": ev.text, "home": game.scores[0], "away": game.scores[1], "left": winprob.seconds_left(game),
               "p": p, "quarter": game.quarter, "clock": game.clock}
        if rollouts and exact_every and i % exact_every == 0 and not done:
            row["exact"] = winprob.rollout(game, rollouts, f"{label}|{i}")
        events.append(row)
    return events


def packaged(game, units, events: list) -> dict:
    return {"events": events, "final": list(game.scores),
            "boxes": [build_box({n: dict(c) for n, c in game.players[side].items()}, units[side].roster) for side in (0, 1)]}


def matchup_for(s: dict, spec: dict) -> tuple:
    return matchup(spec["home"], spec["away"], s["lineup"], s["ratings"], s["model"], spec["out_home"], spec["out_away"],
                   play_home=spec["play_home"], play_away=spec["play_away"],
                   swap_home=spec.get("swap_home"), swap_away=spec.get("swap_away"))


def play_live_game(body: dict) -> dict:
    s = shared()
    spec = spec_from(body)
    edges, units = matchup_for(s, spec)
    game = Game(s["tables"], random.Random(f"live|{spec['seed']}"), names=(spec["home"], spec["away"]),
                edges=edges, home_field=True, units=units)
    events = game_events(s, game, float(body.get("pregame", 0.5)), spec["seed"], int(body.get("rollouts", 0)), int(body.get("every", 0)))
    return packaged(game, units, events)


def typical_game() -> dict:
    # replays the finished game that sat closest to the average of the whole run. Same seed, so it plays out identically
    run = state["run"]
    if run is None:
        raise ValueError("start a run first")
    s = shared()
    index = run.typical_index()
    spec = run.spec
    edges, units = matchup_for(s, spec)
    replay = Game(s["tables"], random.Random(f"{spec['seed']}|{index}"), names=(spec["home"], spec["away"]),
                  edges=edges, home_field=True, units=units)
    snap = run.snapshot()
    pregame = min(0.97, max(0.03, snap.get("home_win_prob", 0.5)))
    events = game_events(s, replay, pregame, index)
    out = packaged(replay, units, events)
    out.update({"index": index, "of": snap["games_done"], "home": spec["home"], "away": spec["away"], "pregame": pregame,
                "average_points": [snap["teams"][0]["points"], snap["teams"][1]["points"]]})
    return out


def pregame_probability(body: dict, sims: int = 200) -> float:
    s = shared()
    spec = spec_from(body)
    edges, units = matchup_for(s, spec)
    wins = 0.0
    for i in range(sims):
        g = runner.play_one(s["tables"], spec, edges, units, i)
        wins += 1.0 if g.scores[0] > g.scores[1] else 0.5 if g.scores[0] == g.scores[1] else 0.0
    return wins / sims


def validation() -> dict:
    out = {"seasons": [], "model": None}
    for path in sorted(BACKTEST_DIR.glob("summary_*_players_t1.5_p1.5.json")):
        d = json.loads(path.read_text())
        out["seasons"].append({"season": path.name.split("_")[1], **d})
    wp = ROOT / "data" / "tables" / "winprob.json"
    if wp.exists():
        out["winprob"] = json.loads(wp.read_text())["report"]
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, payload, status=200, kind="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            pass

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send((STATIC / "index.html").read_bytes(), kind="text/html; charset=utf-8")
            elif url.path == "/api/teams":
                self.send(teams())
            elif url.path == "/api/units":
                self.send(unit_grades(shared()["ratings"]))
            elif url.path == "/api/lineup":
                self.send(lineup(q["team"]))
            elif url.path == "/api/run/status":
                run = state["run"]
                self.send(run.snapshot() if run else {"idle": True})
            elif url.path == "/api/run/typical":
                self.send(typical_game())
            elif url.path == "/api/validation":
                self.send(validation())
            else:
                self.send({"error": "not found"}, 404)
        except Exception as e:
            self.send({"error": str(e)}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self.body()
            if url.path == "/api/run":
                if state["run"] is not None:
                    state["run"].stop()
                run = runner.MillionRun(spec_from(body), int(body.get("games", 10000)), int(body.get("workers", runner.default_workers())))
                state["run"] = run
                run.start()
                self.send({"started": True})
            elif url.path == "/api/run/stop":
                if state["run"] is not None:
                    state["run"].stop()
                self.send({"stopped": True})
            elif url.path == "/api/game":
                self.send(play_live_game(body))
            elif url.path == "/api/pregame":
                self.send({"p": pregame_probability(body, int(body.get("sims", 200)))})
            else:
                self.send({"error": "not found"}, 404)
        except Exception as e:
            self.send({"error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8000)
    ap.add_argument("--no-update", action="store_true", help="skip downloading new games and rebuilding the ratings")
    args = ap.parse_args()
    if not args.no_update:
        print("Checking for new games, injuries and rosters...")
        if update.refresh():
            print("Ratings are up to date.")
        else:
            print("Could not update (no internet?). Using the ratings already saved.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if state["run"] is not None:
            state["run"].stop()


if __name__ == "__main__":
    main()
