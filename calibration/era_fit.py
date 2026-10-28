import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.game import Game
from engine.tables import Tables

STATS = ["plays", "pass_att", "comp", "pass_yards", "rush_att", "rush_yards", "sacks", "ints", "first_downs", "points"]
PRIORS = ROOT / "data" / "players" / "priors.json"
_worker = {}


def real_targets(seasons: list) -> dict:
    # real average per team per game, counted the way the engine counts
    cols = ["game_id", "season", "season_type", "home_score", "away_score", "play_type", "penalty", "two_point_attempt",
            "pass_attempt", "sack", "complete_pass", "interception", "rush_attempt", "yards_gained", "first_down"]
    frames = [pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet", columns=cols) for s in seasons]
    d = pd.concat(frames)
    d = d[d.season_type == "REG"]
    games = d.game_id.nunique()
    team_games = 2 * games
    points = d.drop_duplicates("game_id")[["home_score", "away_score"]].sum().sum()
    s = d[d.play_type.isin(["pass", "run"]) & (d.penalty.fillna(0) != 1) & (d.two_point_attempt.fillna(0) != 1)]
    out = {"plays": len(s), "pass_att": ((s.pass_attempt == 1) & (s.sack != 1)).sum(), "comp": (s.complete_pass == 1).sum(),
           "pass_yards": s[s.complete_pass == 1].yards_gained.sum(), "rush_att": (s.rush_attempt == 1).sum(),
           "rush_yards": s[s.rush_attempt == 1].yards_gained.sum(), "sacks": (s.sack == 1).sum(),
           "ints": (s.interception == 1).sum(), "first_downs": (s.first_down == 1).sum(), "points": points}
    return {k: float(v) / team_games for k, v in out.items()}


def _init(table_dir):
    _worker["tables"] = Tables(table_dir)


def _play(seed):
    g = Game(_worker["tables"], random.Random(seed), names=("A", "B"))
    while not (g.over and not g._queue):
        g.step()
    rows = []
    for side in (0, 1):
        st = g.stats[side]
        rows.append([st["plays"], st["pass_att"], st["pass_comp"], st["pass_yards"], st["rush_att"], st["rush_yards"],
                     st["sack"], st["interceptions_thrown"], st["first_downs"], g.scores[side]])
    return rows


def sim_stats(table_dir, games: int, workers: int = 2) -> dict:
    with mp.Pool(workers, initializer=_init, initargs=(str(table_dir),)) as pool:
        rows = [r for game in pool.map(_play, range(games), chunksize=25) for r in game]
    return dict(zip(STATS, np.mean(rows, axis=0)))


def logit(p):
    return math.log(p / (1 - p))


def update(dials: dict, real: dict, sim: dict, variances: dict, damping: float = 0.8) -> dict:
    new = dict(dials)
    new["pace"] = dials["pace"] * (1 + damping * (sim["plays"] / real["plays"] - 1))
    new["completion"] += damping * (logit(real["comp"] / real["pass_att"]) - logit(sim["comp"] / sim["pass_att"]))
    new["interception"] += damping * (logit(real["ints"] / real["pass_att"]) - logit(sim["ints"] / sim["pass_att"]))
    real_sack = real["sacks"] / (real["pass_att"] + real["sacks"])
    sim_sack = sim["sacks"] / (sim["pass_att"] + sim["sacks"])
    new["sack"] += damping * (logit(real_sack) - logit(sim_sack))
    new["completion_tilt"] += damping * (real["pass_yards"] / real["comp"] - sim["pass_yards"] / sim["comp"]) / variances["pass"]
    new["run_tilt"] += damping * (real["rush_yards"] / real["rush_att"] - sim["rush_yards"] / sim["rush_att"]) / variances["run"]
    return new


def fit(table_dir: Path, target_seasons: list, games: int, rounds: int, workers: int) -> dict:
    priors = json.loads(PRIORS.read_text())
    variances = {"pass": priors["completion_yards"]["yards_variance"], "run": priors["run_yards"]["yards_variance"]}
    real = real_targets(target_seasons)
    dials = {"pace": 1.0, "completion": 0.0, "interception": 0.0, "sack": 0.0, "completion_tilt": 0.0, "run_tilt": 0.0}
    era_path = Path(table_dir) / "era.json"
    for i in range(rounds):
        era_path.write_text(json.dumps(dials))
        sim = sim_stats(table_dir, games, workers)
        print(f"round {i}: " + "  ".join(f"{k} {sim[k]:.2f}/{real[k]:.2f}" for k in STATS), flush=True)
        dials = update(dials, real, sim, variances)
    era_path.write_text(json.dumps({**dials, "target_seasons": target_seasons, "real": real}, indent=1))
    final = sim_stats(table_dir, games, workers)
    print("final: " + "  ".join(f"{k} {final[k]:.2f}/{real[k]:.2f}" for k in STATS))
    return dials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    ap.add_argument("--target", default="2025", help="real seasons to match, like 2025 or 2024-2025")
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    first, _, last = args.target.partition("-")
    seasons = list(range(int(first), int(last or first) + 1))
    fit(Path(args.tables), seasons, args.games, args.rounds, args.workers)


if __name__ == "__main__":
    main()
