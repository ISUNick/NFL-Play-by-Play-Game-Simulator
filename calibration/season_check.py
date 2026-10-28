import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))
import team_calibration as tc


def real_wins() -> np.ndarray:
    g = tc.real_league(2010)
    both = pd.concat([g.assign(team=g.home_team, w=(g.margin > 0) + 0.5 * (g.margin == 0)),
                      g.assign(team=g.away_team, w=(g.margin < 0) + 0.5 * (g.margin == 0))])
    per = both.groupby(["season", "team"]).agg(w=("w", "sum"), n=("w", "size"))
    return (per.w / per.n * 17).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", type=int, default=300)
    ap.add_argument("--seasons", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    from engine.teams import ALL_TEAMS

    pairs = [(i, j) for i in range(32) for j in range(i + 1, 32)]
    allp = [p for p in pairs for _ in range(args.per_pair)]
    jobs = [(40_000_000 + k, allp[k:k + 500]) for k in range(0, len(allp), 500)]
    t0 = time.time()
    with mp.Pool(args.workers, initializer=tc._init) as pool:
        rows = [r for part in pool.imap(tc._league_chunk, jobs) for r in part]
    print(f"{len(rows):,} games in {time.time() - t0:.0f}s")
    pool_ = {}
    for i, j, a, b in rows:
        pool_.setdefault((i, j), []).append(1.0 if a > b else 0.5 if a == b else 0.0)
    pool_ = {k: np.array(v) for k, v in pool_.items()}

    rng = np.random.default_rng(7)
    wins = np.zeros((args.seasons, 32))
    for s in range(args.seasons):
        for _ in range(17):
            order = rng.permutation(32)
            for k in range(0, 32, 2):
                a, b = int(order[k]), int(order[k + 1])
                lo, hi = (a, b) if a < b else (b, a)
                r = rng.choice(pool_[(lo, hi)])
                wins[s, lo] += r
                wins[s, hi] += 1 - r
    sim = wins.ravel()
    real = real_wins()

    def share(x, lo, hi):
        return ((x >= lo) & (x < hi)).mean() * 100

    print(f"\nreal: {len(real)} team seasons (2010 to 2025, scaled to 17 games)   sim: {len(sim):,} team seasons\n")
    print(f"{'':<34}{'real':>8}{'sim':>8}")
    rows2 = [("average wins", real.mean(), sim.mean()), ("spread of wins (SD)", real.std(), sim.std()),
             ("12 or more wins (%)", share(real, 11.75, 99), share(sim, 11.75, 99)),
             ("13 or more wins (%)", share(real, 12.75, 99), share(sim, 12.75, 99)),
             ("14 or more wins (%)", share(real, 13.75, 99), share(sim, 13.75, 99)),
             ("15 or more wins (%)", share(real, 14.75, 99), share(sim, 14.75, 99)),
             ("8 or 9 wins, near .500 (%)", share(real, 7.75, 9.75), share(sim, 7.75, 9.75)),
             ("5 or fewer wins (%)", share(real, -1, 5.75), share(sim, -1, 5.75)),
             ("4 or fewer wins (%)", share(real, -1, 4.75), share(sim, -1, 4.75)),
             ("3 or fewer wins (%)", share(real, -1, 3.75), share(sim, -1, 3.75))]
    for name, a, b in rows2:
        print(f"{name:<34}{a:8.2f}{b:8.2f}")
    print("\nbest and worst team of each season (average over seasons)")
    g = tc.real_league(2010)
    both = pd.concat([g.assign(team=g.home_team, w=(g.margin > 0) + 0.5 * (g.margin == 0)),
                      g.assign(team=g.away_team, w=(g.margin < 0) + 0.5 * (g.margin == 0))])
    per = both.groupby(["season", "team"]).agg(w=("w", "sum"), n=("w", "size"))
    per["w17"] = per.w / per.n * 17
    top = per.groupby("season").w17.max()
    bot = per.groupby("season").w17.min()
    print(f"  best team   real {top.mean():5.1f} (range {top.min():.1f} to {top.max():.1f})   sim {wins.max(1).mean():5.1f} "
          f"(5th to 95th percentile {np.percentile(wins.max(1), 5):.1f} to {np.percentile(wins.max(1), 95):.1f})")
    print(f"  worst team  real {bot.mean():5.1f} (range {bot.min():.1f} to {bot.max():.1f})   sim {wins.min(1).mean():5.1f} "
          f"(5th to 95th percentile {np.percentile(wins.min(1), 5):.1f} to {np.percentile(wins.min(1), 95):.1f})")
    print("\nrecords (wins, rounded down for ties):  share of all team seasons")
    print(f"{'wins':>5}{'real %':>9}{'sim %':>8}")
    for w in range(0, 18):
        print(f"{w:>5}{((np.floor(real + 0.25) == w).mean() * 100):9.1f}{((np.floor(sim + 0.25) == w).mean() * 100):8.1f}")
    avg = wins.mean(0)
    print("\nthe 32 teams: average wins over all simulated seasons, and the best and worst season any team had")
    tab = pd.DataFrame({"team": [t.name for t in ALL_TEAMS], "ovr": [t.overall for t in ALL_TEAMS], "avg wins": avg,
                        "lowest": wins.min(0), "highest": wins.max(0),
                        "10+ win seasons %": (wins >= 9.75).mean(0) * 100})
    print(tab.sort_values("avg wins", ascending=False).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
