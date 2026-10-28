import argparse
import json
import multiprocessing as mp
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
import build_players as bp
from calibration.backtest import _init, simulate
from engine.teams import TeamModel
from players.lineup import Ratings, matchup

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT_DIR = ROOT / "data" / "predictions"


def schedule(season: int) -> pd.DataFrame:
    path = bp.RAW_DIR / "schedule.csv"
    urllib.request.urlretrieve(SCHEDULE_URL, path)
    df = pd.read_csv(path)
    df = df[(df.season == season) & (df.game_type == "REG")].copy()
    df["home_team"], df["away_team"] = df.home_team.replace(bp.TEAM_FIX), df.away_team.replace(bp.TEAM_FIX)
    return df


def predict(season: int, sims: int, workers: int, team_scale: float, player_scale: float) -> pd.DataFrame:
    games = schedule(season)
    todo = games[games.result.isna()]
    ratings = Ratings.load().scaled(team_scale, player_scale)
    lineup = pd.read_csv(ROOT / "data" / "players" / "current_roster.csv")
    model = TeamModel()
    jobs = []
    for g in todo.itertuples():
        edges, units = matchup(g.home_team, g.away_team, lineup, ratings, model)
        jobs.append((g.game_id, g.home_team, g.away_team, edges, units, sims))
    rows = []
    with mp.Pool(workers, initializer=_init, initargs=("players", str(ROOT / "data" / "tables"))) as pool:
        for game_id, out in pool.imap_unordered(simulate, jobs, chunksize=2):
            h = np.array([o[0] for o in out], float)
            a = np.array([o[1] for o in out], float)
            g = todo[todo.game_id == game_id].iloc[0]
            rows.append({"game_id": game_id, "week": int(g.week), "home_team": g.home_team, "away_team": g.away_team,
                         "home_win": float((h > a).mean() + 0.5 * (h == a).mean()), "margin": float((h - a).mean()),
                         "total": float((h + a).mean()), "spread_line": g.spread_line, "total_line": g.total_line})
    return pd.DataFrame(rows).sort_values(["week", "game_id"]), games


def standings(games: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    done = games[games.result.notna()]
    wins, losses, ties = {}, {}, {}
    for g in done.itertuples():
        for team, mine in ((g.home_team, g.result), (g.away_team, -g.result)):
            wins[team] = wins.get(team, 0) + (mine > 0)
            losses[team] = losses.get(team, 0) + (mine < 0)
            ties[team] = ties.get(team, 0) + (mine == 0)
    rest, margin = {}, {}
    for r in pred.itertuples():
        rest[r.home_team] = rest.get(r.home_team, 0) + r.home_win
        rest[r.away_team] = rest.get(r.away_team, 0) + 1 - r.home_win
        margin.setdefault(r.home_team, []).append(r.margin)
        margin.setdefault(r.away_team, []).append(-r.margin)
    teams = sorted(set(games.home_team) | set(games.away_team))
    rows = [{"team": t, "w": int(wins.get(t, 0)), "l": int(losses.get(t, 0)), "t": int(ties.get(t, 0)),
             "proj_wins": wins.get(t, 0) + 0.5 * ties.get(t, 0) + rest.get(t, 0),
             "rest_of_season_margin": float(np.mean(margin.get(t, [0])))} for t in teams]
    return pd.DataFrame(rows).sort_values("proj_wins", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--team-scale", type=float, default=1.5)
    ap.add_argument("--player-scale", type=float, default=1.5)
    args = ap.parse_args()
    pred, games = predict(args.season, args.sims, args.workers, args.team_scale, args.player_scale)
    table = standings(games, pred)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_DIR / f"games_{args.season}.csv", index=False)
    table.to_csv(OUT_DIR / f"teams_{args.season}.csv", index=False)
    print(table.round(2).to_string(index=False))
    print(pred[pred.week == pred.week.min()].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
