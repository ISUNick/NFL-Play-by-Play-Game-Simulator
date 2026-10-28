import glob
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
OUT = ROOT / "data" / "tables" / "home_field.json"
FIRST, LAST, SKIP = 2010, 2025, 2020
MIN_FOUL = 150


def real_margin() -> dict:
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "pbp_*.parquet"))):
        d = pd.read_parquet(f, columns=["game_id", "season", "location", "home_score", "away_score"])
        rows.append(d.groupby("game_id").agg(season=("season", "first"), loc=("location", "first"),
                                             hs=("home_score", "max"), a=("away_score", "max")))
    g = pd.concat(rows)
    g = g[(g["loc"] == "Home") & g.season.between(FIRST, LAST) & (g.season != SKIP)]
    m = g.hs - g.a
    return {"games": int(len(g)), "home_margin": float(m.mean()), "home_margin_se": float(m.std() / len(g) ** 0.5),
            "home_win_pct": float(((m > 0).mean() + 0.5 * (m == 0).mean()))}


def real_penalties() -> dict:
    import numpy as np
    import pandas as pd
    import build_penalty as bp
    cols = bp.COLS + ["posteam_type", "location"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "pbp_*.parquet")))],
                   ignore_index=True).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g = df.groupby("game_id")
    for c in ["down", "ydstogo", "yardline_100", "posteam", "qtr", "game_seconds_remaining"]:
        df["next_" + c] = g[c].shift(-1)
    f = lambda c: df[c].fillna(0) == 1
    base = (df["down"].isin([1, 2, 3, 4]) & df["season"].between(FIRST, LAST) & (df["season"] != SKIP) & ~f("timeout")
            & ~f("two_point_attempt") & df["yardline_100"].notna() & df["ydstogo"].notna() & df["score_differential"].notna()
            & df["game_seconds_remaining"].notna() & (df["location"] == "Home"))
    plays = base & df["play_type"].isin(["run", "pass", "punt", "field_goal"])
    flagged = base & (df["play_type"] == "no_play") & f("penalty") & df["penalty_team"].notna()
    kept = df["next_posteam"] == df["posteam"]
    first = (df["next_down"] == 1) & (df["down"] > 1)
    same = df["next_down"] == df["down"]
    kick = df["penalty_type"].fillna("").str.contains(bp.KICK_FOUL_TEXT, case=False, regex=True)
    ok = flagged & ~kick & kept & df["next_yardline_100"].notna() & (first | same)
    w = df[plays | ok].copy()
    w["is_pen"] = ok.loc[w.index]
    w["kind"] = np.where(w["is_pen"], bp.penalty_kind(w["desc"]), "x")
    w["foul"] = w["penalty_type"].fillna("")
    w["by_off"] = w.is_pen & (w.penalty_team == w.posteam)
    w["by_def"] = w.is_pen & (w.penalty_team != w.posteam)
    w["pre"] = w.is_pen & (w.kind == "pre")
    w["off_home"] = w.posteam_type == "home"

    out = {}
    for phase, pre in (("pre", True), ("post", False)):
        for side, col in (("offense", "by_off"), ("defense", "by_def")):
            hit = w[col] & (w.pre == pre) & w.is_pen
            at_home = (w.off_home) if side == "offense" else ~w.off_home
            rate = {}
            for loc, sel in (("home", at_home), ("away", ~at_home)):
                rate[loc] = float((hit & sel).sum() / sel.sum())
            mean = (rate["home"] + rate["away"]) / 2
            entry = {"rate": rate, "multiplier": {loc: rate[loc] / mean for loc in rate}}
            if pre:
                fouls = w[hit].foul
                pooled = fouls.value_counts()
                big = [x for x in pooled.index if pooled[x] >= MIN_FOUL]
                weights = {}
                total = {loc: float((hit & sel).sum()) for loc, sel in (("home", at_home), ("away", ~at_home))}
                for name in big + ["other"]:
                    share = {}
                    for loc, sel in (("home", at_home), ("away", ~at_home)):
                        m = hit & sel
                        k = (w[m].foul == name).sum() if name != "other" else (~w[m].foul.isin(big)).sum()
                        share[loc] = k / total[loc]
                    pool = sum(share[l] * total[l] for l in share) / sum(total.values())
                    weights[name] = {loc: float(share[loc] / pool) if pool else 1.0 for loc in share}
                entry["foul_weights"] = weights
            out[f"{phase}_{side}"] = entry
    return out


def measure() -> None:
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    data = {"sample": f"{FIRST} to {LAST} without {SKIP}, games at a home field", "real": real_margin(),
            "penalties": real_penalties(), "rating_points": old.get("rating_points", 0.0)}
    OUT.write_text(json.dumps(data, indent=1))
    r = data["real"]
    print(f"real home margin {r['home_margin']:.2f} (se {r['home_margin_se']:.2f}), home win rate {r['home_win_pct']:.3f}, {r['games']} games")
    for key, e in data["penalties"].items():
        print(f"{key:14s} rate home {e['rate']['home']:.4f} away {e['rate']['away']:.4f}  multiplier home {e['multiplier']['home']:.3f} away {e['multiplier']['away']:.3f}")
        for name, wt in e.get("foul_weights", {}).items():
            print(f"      {name:34s} home {wt['home']:.2f} away {wt['away']:.2f}")


_W = {}


def _init():
    from engine.tables import Tables
    from engine.teams import TeamModel
    _W["tables"], _W["model"] = Tables(), TeamModel()


def _play(job):
    from engine.game import Game
    seed, bonus = job
    model = _W["model"]
    model.home_points = bonus
    from engine.teams import Team
    avg = Team("Avg", "AVG", round(model._stats["offense"][0]), round(model._stats["defense"][0]), 80)
    edges = model.edge_pair(avg, avg, home_field=True)
    g = Game(_W["tables"], random.Random(seed), names=("H", "A"), edges=edges, home_field=True, playoff=False)
    while not (g.over and not g._queue):
        g.step()
    flags = [g.stats[i].get("penalty", 0) for i in (0, 1)]
    return g.scores[0] - g.scores[1], flags[0], flags[1]


def simulate(bonus: float, games: int, workers: int = 2):
    with mp.Pool(workers, initializer=_init) as pool:
        res = pool.map(_play, [(f"hf|{bonus}|{i}", bonus) for i in range(games)], chunksize=200)
    n = len(res)
    margin = sum(r[0] for r in res) / n
    sd = (sum((r[0] - margin) ** 2 for r in res) / n) ** 0.5
    win = sum(r[0] > 0 for r in res) / n + 0.5 * sum(r[0] == 0 for r in res) / n
    return margin, sd / n ** 0.5, win, sum(r[1] for r in res) / n, sum(r[2] for r in res) / n


def calibrate(games: int = 20000) -> None:
    data = json.loads(OUT.read_text())
    target = data["real"]["home_margin"]
    m0 = simulate(0.0, games)
    m3 = simulate(3.0, games)
    slope = (m3[0] - m0[0]) / 3.0
    bonus = (target - m0[0]) / slope
    print(f"bonus 0: margin {m0[0]:.2f} (se {m0[1]:.2f}); bonus 3: margin {m3[0]:.2f}; slope {slope:.2f} points per rating point")
    data["rating_points"] = round(bonus, 2)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"target {target:.2f} -> rating_points {data['rating_points']}")


def check(games: int = 20000, bonus: float = None) -> None:
    data = json.loads(OUT.read_text())
    m, se, win, fh, fa = simulate(data["rating_points"] if bonus is None else bonus, games)
    r = data["real"]
    print(f"sim home margin {m:.2f} (se {se:.2f}), real {r['home_margin']:.2f}; home win rate {win:.3f}, real {r['home_win_pct']:.3f}")
    print(f"penalties per game: home team {fh:.2f}, away team {fa:.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    args = [float(a) if "." in a else int(a) for a in sys.argv[2:]]
    {"measure": measure, "calibrate": calibrate, "check": check}[cmd](*args)
