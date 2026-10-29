import math
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.game import Game
from engine.tables import Tables
from engine.teams import TeamModel
from app.box import build_box
from players.lineup import Ratings, matchup

BATCH = 100
FINGERPRINT = ["points", "pass_yards", "rush_yards", "td", "interceptions_thrown", "sack", "plays", "first_downs"]
TEAM_KEYS = ["pass_yards", "rush_yards", "td", "interceptions_thrown", "sack", "plays", "first_downs", "pass_att", "pass_comp"]
_worker = {}


def load_lineup() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "players" / "current_roster.csv")


def default_workers() -> int:
    # every core, but leave one free for the browser and the server once there are more than two
    cores = os.cpu_count() or 2
    return cores if cores <= 2 else cores - 1


def lineup_args(spec: dict) -> dict:
    return {"out_home": spec["out_home"], "out_away": spec["out_away"],
            "play_home": spec.get("play_home", []), "play_away": spec.get("play_away", []),
            "swap_home": spec.get("swap_home"), "swap_away": spec.get("swap_away")}


def rosters_for(spec: dict) -> list:
    ratings = Ratings.load().scaled(spec["team_scale"], spec["player_scale"])
    _, units = matchup(spec["home"], spec["away"], load_lineup(), ratings, TeamModel(), **lineup_args(spec))
    return [u.roster for u in units]


def _init(spec):
    _worker["spec"] = spec
    _worker["tables"] = Tables()
    ratings = Ratings.load().scaled(spec["team_scale"], spec["player_scale"])
    _worker["edges"], _worker["units"] = matchup(spec["home"], spec["away"], load_lineup(), ratings, TeamModel(),
                                                 **lineup_args(spec))


def play_one(tables, spec, edges, units, index) -> Game:
    g = Game(tables, random.Random(f"{spec['seed']}|{index}"), names=(spec["home"], spec["away"]),
             edges=edges, home_field=True, units=units)
    while not (g.over and not g._queue):
        g.step()
    return g


def run_batch(job) -> tuple:
    start, count = job
    spec, tables = _worker["spec"], _worker["tables"]
    total, prints = Counter(), []
    for i in range(start, start + count):
        g = play_one(tables, spec, _worker["edges"], _worker["units"], i)
        h, a = g.scores
        prints.append([g.scores[side] if k == "points" else g.stats[side][k] for side in (0, 1) for k in FINGERPRINT])
        total["games"] += 1
        total["home_wins"] += 1.0 if h > a else 0.5 if h == a else 0.0
        total["ties"] += h == a
        total[f"score|{h}|{a}"] += 1
        total["ot"] += g.went_to_ot
        total["margin"] += h - a
        total["margin_sq"] += (h - a) ** 2
        for side in (0, 1):
            total[f"points{side}"] += g.scores[side]
            for k in TEAM_KEYS:
                total[f"{k}{side}"] += g.stats[side][k]
            for name, line in g.players[side].items():
                for stat, v in line.items():
                    if stat.startswith("long_"):
                        continue
                    total[f"p|{side}|{name}|{stat}"] += v
    return total, start, np.array(prints, dtype=np.float32)


class MillionRun:
    def __init__(self, spec: dict, games: int, workers: int):
        self.spec, self.games, self.workers = spec, games, workers
        self.totals = Counter()
        self.history = []
        self.lock = threading.Lock()
        self.started = self.finished = None
        self.stopped = False
        self._pool = None
        self._recent = []
        self.prints = []
        self.error = None
        self.rosters = rosters_for(spec)

    def start(self):
        self.started = time.time()
        self._pool = mp.Pool(self.workers, initializer=_init, initargs=(self.spec,))
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self):
        jobs = [(s, min(BATCH, self.games - s)) for s in range(0, self.games, BATCH)]
        try:
            for batch, start, prints in self._pool.imap_unordered(run_batch, jobs):
                with self.lock:
                    self.totals.update(batch)
                    self.prints.append((start, prints))
                    n = self.totals["games"]
                    p = self.totals["home_wins"] / n
                    self.history.append((int(n), p, *wilson_band(p, n)))
                    self._recent.append((time.time(), n))
                    self._recent = self._recent[-12:]
        except Exception as e:
            self.error = str(e)
        finally:
            self.finished = time.time()
            self._pool.close()

    def stop(self):
        self.stopped = True
        if self._pool is not None:
            self._pool.terminate()
        self.finished = self.finished or time.time()

    def typical_index(self) -> int:
        # the finished game whose team stats and score sit closest to the average of all games (each stat in standard deviations)
        with self.lock:
            if not self.prints:
                raise ValueError("no games finished yet")
            index = np.concatenate([np.arange(start, start + len(rows)) for start, rows in self.prints])
            rows = np.vstack([rows for _, rows in self.prints])
        spread = rows.std(axis=0)
        spread[spread == 0] = 1.0
        distance = (((rows - rows.mean(axis=0)) / spread) ** 2).sum(axis=1)
        return int(index[int(np.argmin(distance))])

    def snapshot(self) -> dict:
        with self.lock:
            t, n = self.totals, int(self.totals["games"])
            now = self.finished or time.time()
            elapsed = max(now - (self.started or now), 1e-9)
            rate = (n / elapsed) if len(self._recent) < 2 or self.finished else \
                (self._recent[-1][1] - self._recent[0][1]) / max(self._recent[-1][0] - self._recent[0][0], 1e-9)
            out = {"games_done": n, "games_total": self.games, "elapsed": elapsed, "games_per_second": rate,
                   "running": self.finished is None, "stopped": self.stopped, "error": self.error, "home": self.spec["home"],
                   "away": self.spec["away"], "history": thin(self.history, 300)}
            if n:
                p = t["home_wins"] / n
                lo, hi = wilson_band(p, n)
                mean_margin = t["margin"] / n
                out.update({"home_win_prob": p, "ci_low": lo, "ci_high": hi, "mean_margin": mean_margin,
                            "margin_sd": math.sqrt(max(t["margin_sq"] / n - mean_margin ** 2, 0.0)),
                            "tie_rate": t["ties"] / n, "ot_rate": t["ot"] / n,
                            "teams": [{"points": t[f"points{s}"] / n, **{k: t[f"{k}{s}"] / n for k in TEAM_KEYS}} for s in (0, 1)],
                            "boxes": [build_box(player_lines(t, n, side), self.rosters[side]) for side in (0, 1)],
                            "scores": score_counts(t)})
            return out


def wilson_band(p: float, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 1.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def thin(rows: list, keep: int) -> list:
    if len(rows) <= keep:
        return rows
    step = len(rows) / keep
    return [rows[int(i * step)] for i in range(keep)] + [rows[-1]]


def score_counts(t: Counter) -> list:
    # how many games ended with each final score: [[home points, away points, games], ...]
    rows = []
    for key, v in t.items():
        if key.startswith("score|"):
            _, h, a = key.split("|")
            rows.append([int(h), int(a), int(v)])
    return rows


def player_lines(t: Counter, n: int, side: int) -> dict:
    # average per game for every player of one side: {name: {stat: value}}
    lines = {}
    prefix = f"p|{side}|"
    for key, v in t.items():
        if key.startswith(prefix):
            _, _, name, stat = key.split("|", 3)
            lines.setdefault(name, {})[stat] = v / n
    return lines
