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
sys.path.insert(0, str(ROOT / "data_prep"))
import build_players as bp
from engine.game import Game
from engine.tables import Tables
from engine.teams import TeamModel
from players.lineup import Ratings, matchup

OUT_DIR = ROOT / "data" / "backtest"
SPREAD_SD = 13.86
STAT_KEYS = ["pass_yards", "rush_yards", "sack", "interceptions_thrown", "plays", "first_downs"]
_worker = {}


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def real_games(seasons: list) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "season_type", "home_team", "away_team", "result", "total", "spread_line", "total_line"]
    df = pd.concat([pd.read_parquet(bp.RAW_DIR / f"pbp_{s}.parquet", columns=cols) for s in seasons])
    df = df[df.season_type == "REG"].drop_duplicates("game_id").dropna(subset=["result"])
    df["home_team"], df["away_team"] = df.home_team.replace(bp.TEAM_FIX), df.away_team.replace(bp.TEAM_FIX)
    return df.reset_index(drop=True)


def real_team_stats(seasons: list) -> pd.DataFrame:
    cols = ["game_id", "posteam", "play_type", "qb_dropback", "qb_scramble", "sack", "interception", "complete_pass",
            "rush_attempt", "yards_gained", "first_down", "penalty"]
    df = pd.concat([pd.read_parquet(bp.RAW_DIR / f"pbp_{s}.parquet", columns=cols) for s in seasons])
    df = df[df.play_type.isin(["pass", "run"]) & (df.penalty.fillna(0) != 1)]
    df["posteam"] = df.posteam.replace(bp.TEAM_FIX)
    df["pass_yards"] = np.where(df.complete_pass == 1, df.yards_gained, 0)
    df["rush_yards"] = np.where((df.rush_attempt == 1) | (df.qb_scramble == 1), df.yards_gained, 0)
    g = df.groupby(["game_id", "posteam"])
    return pd.DataFrame({"pass_yards": g.pass_yards.sum(), "rush_yards": g.rush_yards.sum(), "sack": g.sack.sum(),
                         "interceptions_thrown": g.interception.sum(), "plays": g.size(),
                         "first_downs": g.first_down.sum()}).reset_index()


def _init(variant, table_dir):
    _worker["tables"], _worker["variant"] = Tables(table_dir), variant


def simulate(job):
    game_id, home, away, edges, units, n = job
    tables = _worker["tables"]
    use_units = units if _worker["variant"] == "players" else None
    rows = []
    for i in range(n):
        g = Game(tables, random.Random(f"bt|{game_id}|{i}"), names=(home, away), edges=edges, home_field=True, units=use_units)
        while not (g.over and not g._queue):
            g.step()
        rows.append((g.scores[0], g.scores[1], g.went_to_ot, {k: [g.stats[0][k], g.stats[1][k]] for k in STAT_KEYS}))
    return game_id, rows


def week_jobs(games: pd.DataFrame, season: int, week: int, roster, snaps_all, injuries_all, n: int,
              team_scale: float = 1.0, player_scale: float = 1.0, table_dir=None, qb_scale=None) -> list:
    built = bp.build_ratings(list(range(season - bp.SEASONS_BACK, season + 1)), before=(season, week))
    ratings = Ratings.from_built(built).scaled(team_scale, player_scale, qb_scale)
    lineup = bp.build_lineup_inputs(roster, snaps_all, injuries_all, season, week)
    model = TeamModel()
    jobs = []
    for g in games[(games.season == season) & (games.week == week)].itertuples():
        try:
            edges, units = matchup(g.home_team, g.away_team, lineup, ratings, model)
        except (ValueError, KeyError) as e:
            print(f"skipped {g.game_id}: {e}")
            continue
        jobs.append((g.game_id, g.home_team, g.away_team, edges, units, n))
    return jobs


def score(df: pd.DataFrame) -> dict:
    # df needs: win_prob (home wins in the sim), sim_margin, sim_total, result, total, spread_line, total_line
    won = (df.result > 0).astype(float).to_numpy()
    p = df.win_prob.clip(0.001, 0.999).to_numpy()
    vegas_p = np.array([normal_cdf(s / SPREAD_SD) for s in df.spread_line])
    ties = (df.result == 0).to_numpy()
    keep = ~ties
    out = {
        "games": int(len(df)),
        "straight_up_accuracy": float(np.mean((p[keep] > 0.5) == (won[keep] == 1))),
        "vegas_straight_up_accuracy": float(np.mean((vegas_p[keep] > 0.5) == (won[keep] == 1))),
        "brier": float(np.mean((p[keep] - won[keep]) ** 2)),
        "vegas_brier": float(np.mean((vegas_p[keep] - won[keep]) ** 2)),
        "logloss": float(-np.mean(won[keep] * np.log(p[keep]) + (1 - won[keep]) * np.log(1 - p[keep]))),
        "vegas_logloss": float(-np.mean(won[keep] * np.log(vegas_p[keep]) + (1 - won[keep]) * np.log(1 - vegas_p[keep]))),
        "margin_mae": float(np.mean(np.abs(df.sim_margin - df.result))),
        "vegas_margin_mae": float(np.mean(np.abs(df.spread_line - df.result))),
        "margin_corr_with_result": float(np.corrcoef(df.sim_margin, df.result)[0, 1]),
        "vegas_margin_corr_with_result": float(np.corrcoef(df.spread_line, df.result)[0, 1]),
        "total_mae": float(np.mean(np.abs(df.sim_total - (df.total)))),
        "vegas_total_mae": float(np.mean(np.abs(df.total_line - df.total))),
        "mean_sim_total": float(df.sim_total.mean()), "mean_real_total": float(df.total.mean()),
        "sim_vs_vegas_margin_corr": float(np.corrcoef(df.sim_margin, df.spread_line)[0, 1]),
    }
    return out


def calibration_table(df: pd.DataFrame, bins=(0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)) -> list:
    cut = pd.cut(df.win_prob, list(bins), include_lowest=True)
    rows = []
    for b, g in df.groupby(cut, observed=True):
        rows.append({"bin": str(b), "games": len(g), "predicted": float(g.win_prob.mean()),
                     "actual": float((g.result > 0).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--first-week", type=int, default=5)
    ap.add_argument("--last-week", type=int, default=18)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--variant", choices=["players", "teams"], default="players")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--team-scale", type=float, default=1.0)
    ap.add_argument("--player-scale", type=float, default=1.0)
    ap.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--qb-scale", type=float, default=None)
    args = ap.parse_args()

    games = real_games([args.season])
    roster = bp.latest_roster()
    seasons = list(range(args.season - bp.SEASONS_BACK, args.season + 1))
    snaps_all, injuries_all = bp.load_raw("snaps", seasons), bp.load_raw("injuries", seasons)
    stats = real_team_stats([args.season])
    rows, stat_rows = [], []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.variant, args.tables)) as pool:
        for week in range(args.first_week, args.last_week + 1):
            jobs = week_jobs(games, args.season, week, roster, snaps_all, injuries_all, args.sims,
                             args.team_scale, args.player_scale, args.tables, args.qb_scale)
            for game_id, sims in pool.imap_unordered(simulate, jobs):
                g = games[games.game_id == game_id].iloc[0]
                h = np.array([s[0] for s in sims], float)
                a = np.array([s[1] for s in sims], float)
                wins = (h > a).mean() + 0.5 * (h == a).mean()
                rows.append({**g.to_dict(), "win_prob": wins, "sim_margin": float((h - a).mean()),
                             "sim_total": float((h + a).mean()), "sim_margin_sd": float((h - a).std())})
                for side, team in ((0, g.home_team), (1, g.away_team)):
                    sim_mean = {k: float(np.mean([s[3][k][side] for s in sims])) for k in STAT_KEYS}
                    real = stats[(stats.game_id == game_id) & (stats.posteam == team)]
                    if len(real):
                        stat_rows.append({"game_id": game_id, "team": team, **{f"sim_{k}": v for k, v in sim_mean.items()},
                                          **{f"real_{k}": float(real[k].iloc[0]) for k in STAT_KEYS}})
            print(f"week {week} done, {len(rows)} games", flush=True)
    df = pd.DataFrame(rows).sort_values(["week", "game_id"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.season}_{args.variant}_t{args.team_scale}_p{args.player_scale}{args.tag}"
    df.to_csv(OUT_DIR / f"games_{tag}.csv", index=False)
    st = pd.DataFrame(stat_rows)
    st.to_csv(OUT_DIR / f"team_stats_{tag}.csv", index=False)
    result = score(df)
    result["calibration"] = calibration_table(df)
    result["team_stat_means"] = {k: {"sim": float(st[f"sim_{k}"].mean()), "real": float(st[f"real_{k}"].mean())} for k in STAT_KEYS}
    result["sim_margin_sd"] = float(df.sim_margin_sd.mean())
    result["real_margin_sd"] = float(df.result.std())
    (OUT_DIR / f"summary_{tag}.json").write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        if k != "calibration":
            print(k, v)


if __name__ == "__main__":
    main()
