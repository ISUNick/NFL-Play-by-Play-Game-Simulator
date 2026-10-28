import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import winprob
from engine.game import Game
from engine.tables import Tables
from engine.teams import TeamModel
from players.lineup import Ratings, matchup

_worker = {}


def _init():
    _worker["tables"], _worker["model"] = Tables(), TeamModel()
    _worker["ratings"] = Ratings.load()
    _worker["lineup"] = pd.read_csv(ROOT / "data" / "players" / "current_roster.csv")


def games_for(job):
    # one matchup, many games. Returns, for every game, the features and seconds left at every event, and the result
    home, away, games, seed = job
    edges, units = matchup(home, away, _worker["lineup"], _worker["ratings"], _worker["model"])
    out = []
    for i in range(games):
        g = Game(_worker["tables"], random.Random(f"{seed}|{i}"), names=(home, away), edges=edges, home_field=True, units=units)
        rows, left = [], []
        while True:
            snap = winprob.snapshot(g)
            rows.append(winprob.features(snap))
            left.append(snap["left"])
            if g.over and not g._queue:
                break
            g.step()
        a, b = g.scores
        out.append((np.array(rows), np.array(left), 1.0 if a > b else 0.5 if a == b else 0.0))
    return out


def build_jobs(matchups: int, games: int, seed: int) -> list:
    teams = sorted(Ratings.load().team.index)
    rng = random.Random(seed)
    return [(*rng.sample(teams, 2), games, f"wp{seed}|{i}") for i in range(matchups)]


def assemble(done: list) -> tuple:
    # the pregame chance of a matchup is the share of its own games the home team won
    x, y, left, pre, group = [], [], [], [], []
    for k, games in enumerate(done):
        p = min(0.97, max(0.03, float(np.mean([g[2] for g in games]))))
        for rows, secs, result in games:
            x.append(rows)
            left.append(secs)
            y.append(np.full(len(rows), result))
            pre.append(np.full(len(rows), math.log(p / (1 - p))))
            group.append(np.full(len(rows), k))
    return np.vstack(x), np.concatenate(y), np.concatenate(left), np.concatenate(pre), np.concatenate(group)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchups", type=int, default=300)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    with mp.Pool(args.workers, initializer=_init) as pool:
        done = pool.map(games_for, build_jobs(args.matchups, args.games, args.seed), chunksize=4)
    x, y, left, pre, group = assemble(done)
    prior = pre * np.sqrt(np.maximum(left, 0) / 3600.0)
    train = group % 5 != 0
    beta = winprob.fit(np.column_stack([x, prior])[train], y[train])
    z = beta[0] + np.column_stack([x, prior]) @ beta[1:]
    p = 1 / (1 + np.exp(-z))
    test = ~train
    bins = np.digitize(p[test], [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    calibration = [{"bin": int(b), "n": int((bins == b).sum()), "predicted": float(p[test][bins == b].mean()),
                    "actual": float(y[test][bins == b].mean())} for b in np.unique(bins)]
    report = {"games": args.matchups * args.games, "rows": int(len(y)), "test_logloss": logloss(y[test], p[test]),
              "coin_flip_logloss": math.log(2), "calibration": calibration}
    winprob.MODEL_PATH.write_text(json.dumps({"features": winprob.FEATURES, "beta": list(beta[:-1]),
                                              "prior_weight": float(beta[-1]), "report": report}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
