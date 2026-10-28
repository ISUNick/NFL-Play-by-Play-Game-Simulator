import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "data" / "tables" / "team_model.json"
EFFECTS_PATH = ROOT / "data" / "tables" / "team_effects.csv"
NET_SD = 5.17
STEP = {"completion": 0.25, "interception": 0.40, "sack": 0.40, "completion_yards": 0.010, "run_yards": 0.020}
FIELD = {"completion": "completion", "interception": "interception", "sack": "sack",
         "completion_yards": "completion_tilt", "run_yards": "run_tilt"}


_T = None


def _init():
    global _T
    from engine.tables import Tables
    _T = Tables()


def play(seed, edges):
    from engine.game import Game
    g = Game(_T, random.Random(seed), edges=edges)
    while not g.over or g._queue:
        g.step()
    return g.scores[0], g.scores[1]


def _chunk(args):
    start, n, edge0, edge1 = args
    from engine.teams import Edge
    e = (Edge(**edge0), Edge(**edge1))
    return [play(s, e) for s in range(start, start + n)]


def run_games(n, edge0, edge1, base_seed, pool):
    jobs = [(base_seed + i, min(500, n - i), edge0, edge1) for i in range(0, n, 500)]
    rows = [r for part in pool.imap(_chunk, jobs) for r in part]
    return np.array(rows, dtype=float)


def sens(args):
    eff = pd.read_csv(EFFECTS_PATH)
    channels = {}
    with mp.Pool(args.workers, initializer=_init) as pool:
        for ch, step in STEP.items():
            t0 = time.time()
            up = run_games(args.games, {FIELD[ch]: step}, {}, 1000, pool)
            down = run_games(args.games, {FIELD[ch]: -step}, {}, 1000, pool)
            m_up, m_down = up[:, 0] - up[:, 1], down[:, 0] - down[:, 1]
            slope = (m_up.mean() - m_down.mean()) / (2 * step)
            err = np.sqrt(m_up.var() / len(m_up) + m_down.var() / len(m_down)) / (2 * step)
            own = (up[:, 0].mean() - down[:, 0].mean()) / (2 * step)
            other = (up[:, 1].mean() - down[:, 1].mean()) / (2 * step)
            row = {"sensitivity": float(slope), "sensitivity_error": float(err), "own_points": float(own),
                   "other_points": float(other)}
            for side in ("offense", "defense"):
                e = eff[(eff.channel == ch) & (eff.side == side)].iloc[0]
                row[f"tau_{side}"] = float(e.true_sd / e["var"])
            channels[ch] = row
            print(f"{ch:18} margin per game per unit {slope:9.3f} +/- {err:.3f}   (own {own:+.3f}, other team {other:+.3f})   "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model = {"channels": channels}
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print("wrote", MODEL_PATH)
    rescale()


def rating_scale(extra: float = 1.0) -> float:
    from engine.teams import ALL_TEAMS
    total = np.array([t.offense + t.defense for t in ALL_TEAMS], float)
    return NET_SD / total.std() * extra


def rescale(extra: float = None):
    model = json.loads(MODEL_PATH.read_text())
    if extra is None:
        extra = model.get("league_correction", 1.0)
    k = rating_scale(extra)
    for side in ("offense", "defense"):
        sd = {ch: model["channels"][ch][f"tau_{side}"] * abs(model["channels"][ch]["sensitivity"]) for ch in model["channels"]}
        total = sum(sd.values())
        for ch in sd:
            model["channels"][ch][f"points_sd_{side}"] = sd[ch]
            model["channels"][ch][f"share_{side}"] = sd[ch] / total
        print(f"{side}: dial spreads add up to {total:.2f} points per game (real total {3.56 if side == 'offense' else 2.33})")
    model.pop("league_centering", None)
    model.update({"net_sd_target": NET_SD, "league_correction": extra,
                  "points_per_rating_point_offense": k, "points_per_rating_point_defense": k})
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(f"points per rating point: {k:.4f} (league correction {extra:.4f})")


def _solve(f, lo=-1.0, hi=1.0):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def center():
    import math
    from engine.tables import Tables
    from engine.teams import ALL_TEAMS, TeamModel, logit_shift
    model = TeamModel()
    model.params["channel_centering"] = {}
    raw = [model.raw_shifts(o, d) for o in ALL_TEAMS for d in ALL_TEAMS if o is not d]
    tables = Tables()
    rng = random.Random(1)
    pools = {"completion_yards": [y for v in tables.completion.values() for (y, _, _) in v],
             "run_yards": [y for v in tables.run.values() for (y, _, _) in v]}
    pools = {k: rng.sample(v, min(60000, len(v))) for k, v in pools.items()}
    bases = {"completion": 0.641, "interception": 0.0225, "sack": 0.0628}

    def tilted_mean(ys, t):
        top = max(ys) if t > 0 else min(ys)
        w = [math.exp(t * (y - top)) for y in ys]
        return sum(y * x for y, x in zip(ys, w)) / sum(w)

    out = {}
    for ch in CHANNELS_ALL:
        s_list = [r[ch] for r in raw]
        if ch in bases:
            p0 = bases[ch]
            out[ch] = _solve(lambda c: sum(logit_shift(p0, s + c) for s in s_list) / len(s_list) - p0)
        else:
            ys = pools[ch]
            m0 = sum(ys) / len(ys)
            sub = s_list[::7]
            out[ch] = _solve(lambda c: sum(tilted_mean(ys, s + c) for s in sub) / len(sub) - m0, -0.05, 0.05)
        print(f"{ch:18} centering offset {out[ch]:+.5f}")
    saved = json.loads(MODEL_PATH.read_text())
    saved["channel_centering"] = out
    MODEL_PATH.write_text(json.dumps(saved, indent=2))
    print("wrote", MODEL_PATH)


CHANNELS_ALL = ("completion", "interception", "sack", "completion_yards", "run_yards")


def _league_chunk(args):
    start, pairs = args
    from engine.teams import ALL_TEAMS, TeamModel
    model = TeamModel()
    out = []
    for k, (i, j) in enumerate(pairs):
        e = model.edge_pair(ALL_TEAMS[i], ALL_TEAMS[j])
        a, b = play(start + k, e)
        out.append((i, j, a, b))
    return out


def real_league(from_season: int):
    path = ROOT / "calibration" / "games_cache.pkl"
    if path.exists():
        g = pd.read_pickle(path)
    else:
        import glob
        cols = ["game_id", "season", "week", "season_type", "home_team", "away_team", "home_score", "away_score",
                "spread_line", "result"]
        frames = []
        for f in sorted(glob.glob(str(ROOT / "data/raw/pbp_*.parquet"))):
            d = pd.read_parquet(f, columns=cols)
            frames.append(d.drop_duplicates("game_id"))
        g = pd.concat(frames, ignore_index=True)
        g = g[g.season_type == "REG"].dropna(subset=["home_score", "spread_line"]).copy()
        g.to_pickle(path)
    g = g[g.season >= from_season].copy()
    g["margin"] = g.home_score - g.away_score
    return g


def league(args):
    from engine.teams import ALL_TEAMS, TeamModel
    model = TeamModel()
    rng = random.Random(2024)
    n = args.games
    pairs = []
    for _ in range(n):
        i, j = rng.sample(range(32), 2)
        pairs.append((i, j))
    jobs = [(10_000_000 + k, pairs[k:k + 500]) for k in range(0, n, 500)]
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init) as pool:
        rows = [r for part in pool.imap(_league_chunk, jobs) for r in part]
    print(f"{n:,} games in {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows, columns=["i", "j", "a", "b"])
    df["margin"] = df.a - df.b
    edges = [model.points_edge(t, t) for t in ALL_TEAMS]
    net = np.array([x + y for x, y in edges])
    df["pred"] = net[df.i] - net[df.j]
    report(df, net, args)


def real_stats(g, spread_margin):
    return {"margin sd": g.margin.std(), "mean |margin|": g.margin.abs().mean(),
            "margins of 21+": (g.margin.abs() >= 21).mean() * 100, "one score games (8 or less)": (g.margin.abs() <= 8).mean() * 100,
            "margins of 3 or less": (g.margin.abs() <= 3).mean() * 100, "ties": (g.margin == 0).mean() * 100}


def report(df, net, args):
    from engine.teams import ALL_TEAMS
    g = real_league(2018)
    X = np.c_[np.ones(len(df)), df.pred]
    icpt, slope = np.linalg.lstsq(X, df.margin.to_numpy(), rcond=None)[0]
    resid = df.margin - (icpt + slope * df.pred)
    print(f"\nSIM margin = {icpt:+.2f} + {slope:.3f} x predicted margin   (1.000 means the dials are exactly right)")
    print(f"suggested correction for the rating scale: x {1 / slope:.4f}")
    rx = np.c_[np.ones(len(g)), g.spread_line]
    ri, rs = np.linalg.lstsq(rx, g.margin.to_numpy(), rcond=None)[0]
    print(f"REAL margin = {ri:+.2f} + {rs:.3f} x market spread (the market is a good yardstick when this is near 0 + 1)")
    rres = g.margin - (ri + rs * g.spread_line)
    print(f"margin sd around the expected margin:  sim {resid.std():.2f}   real {rres.std():.2f}")

    print("\n{:<34}{:>9}{:>9}".format("whole games", "real", "sim"))
    sim = df.assign(m=df.margin)
    neutral_sd = (g.spread_line - g.margin.mean()).std()
    h = g.margin.mean()
    rl = g.assign(m=g.margin - h)
    rows = [("margin sd", rl.m.std(), sim.m.std()), ("mean |margin|", rl.m.abs().mean(), sim.m.abs().mean()),
            ("games decided by 21+ (%)", (rl.m.abs() >= 21).mean() * 100, (sim.m.abs() >= 21).mean() * 100),
            ("games decided by 9+ (%)", (rl.m.abs() >= 9).mean() * 100, (sim.m.abs() >= 9).mean() * 100),
            ("games decided by 3 or less (%)", (rl.m.abs() <= 3).mean() * 100, (sim.m.abs() <= 3).mean() * 100),
            ("sd of the expected gap between teams", neutral_sd, df.pred.std())]
    for name, a, b in rows:
        print(f"{name:<34}{a:9.2f}{b:9.2f}")

    print("\nbetter team wins, by size of the expected gap (market spread minus home field for real games)")
    print("{:<14}{:>10}{:>10}{:>10}{:>10}".format("gap (points)", "real n", "real win%", "sim n", "sim win%"))
    neutral = g.spread_line - h
    rg = g.assign(gap=neutral.abs(), fav_won=np.where(neutral >= 0, g.margin - h > 0, g.margin - h < 0) + 0.5 * (g.margin - h == 0))
    sg = df.assign(gap=df.pred.abs(), fav_won=np.where(df.pred >= 0, df.margin > 0, df.margin < 0) + 0.5 * (df.margin == 0))
    for lo, hi in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 99)]:
        a, b = rg[(rg.gap >= lo) & (rg.gap < hi)], sg[(sg.gap >= lo) & (sg.gap < hi)]
        print(f"{lo:>3} to {hi if hi < 99 else '+':<7}{len(a):10d}{a.fav_won.mean() * 100:10.1f}{len(b):10d}{b.fav_won.mean() * 100:10.1f}")
    print(f"{'all games':<14}{len(rg):10d}{rg.fav_won.mean() * 100:10.1f}{len(sg):10d}{sg.fav_won.mean() * 100:10.1f}")

    rows = []
    for t, team in enumerate(ALL_TEAMS):
        a = df[df.i == t]
        b = df[df.j == t]
        wins = (a.margin > 0).sum() + 0.5 * (a.margin == 0).sum() + (b.margin < 0).sum() + 0.5 * (b.margin == 0).sum()
        n = len(a) + len(b)
        pf = (a.a.sum() + b.b.sum()) / n
        pa = (a.b.sum() + b.a.sum()) / n
        rows.append((team.name, team.offense, team.defense, team.overall, net[t], wins / n, pf, pa))
    tab = pd.DataFrame(rows, columns=["team", "off", "def", "ovr", "pred net", "win%", "pts for", "pts against"])
    p = tab["win%"].to_numpy()
    obs_var = p.var() * 17 ** 2 + (p * (1 - p)).mean() * 17
    rg2 = g.copy()
    wins = pd.concat([rg2.assign(team=rg2.home_team, w=(rg2.margin > 0) + 0.5 * (rg2.margin == 0)),
                      rg2.assign(team=rg2.away_team, w=(rg2.margin < 0) + 0.5 * (rg2.margin == 0))])
    per = wins.groupby(["season", "team"]).agg(w=("w", "sum"), n=("w", "size"))
    per["w17"] = per.w / per.n * 17
    print(f"\nSD of 17 game win totals across teams:  real {per.w17.std():.2f}   sim {obs_var ** 0.5:.2f}")
    print(f"best and worst team's true win% (sim): {p.max() * 100:.1f}% and {p.min() * 100:.1f}%   "
          f"real 2018 to 2025 best/worst seasons: {per.w17.max():.1f} and {per.w17.min():.1f} wins of 17")
    print("\n" + tab.sort_values("pred net", ascending=False).round(2).to_string(index=False))
    summary = {"slope": float(slope), "intercept": float(icpt), "games": len(df)}
    (ROOT / "calibration" / "league_result.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sens")
    a.add_argument("--games", type=int, default=40000)
    a.add_argument("--workers", type=int, default=2)
    rs = sub.add_parser("rescale")
    rs.add_argument("--correction", type=float, default=None, help="multiplier on the rating scale")
    sub.add_parser("center")
    lg = sub.add_parser("league")
    lg.add_argument("--games", type=int, default=200000)
    lg.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "sens":
        sens(args)
    elif args.cmd == "rescale":
        rescale(args.correction)
    elif args.cmd == "center":
        center()
    elif args.cmd == "league":
        league(args)


if __name__ == "__main__":
    main()
