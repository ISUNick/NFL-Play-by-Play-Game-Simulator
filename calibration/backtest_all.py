import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "walk_forward"


def run(cmd, env=None):
    print(" ".join(str(c) for c in cmd), flush=True)
    if subprocess.run([str(c) for c in cmd], cwd=ROOT, env={**os.environ, **(env or {})}).returncode != 0:
        sys.exit("step failed")


def prepare(season: int, games: int, workers: int) -> Path:
    # tables for a test season only see the four seasons before it, and the era dials match the season right before it
    table_dir = WORK / f"tables_{season}"
    if not (table_dir / "era.json").exists():
        shutil.rmtree(table_dir, ignore_errors=True)
        table_dir.mkdir(parents=True)
        env = {"SIM_TABLE_DIR": str(table_dir), "SIM_TABLE_SEASONS": f"{season - 4}-{season - 1}"}
        run([sys.executable, "data_prep/build_all.py"], env)
        for name in ("winprob.json",):
            if (ROOT / "data" / "tables" / name).exists() and not (table_dir / name).exists():
                shutil.copy(ROOT / "data" / "tables" / name, table_dir / name)
        run([sys.executable, "calibration/era_fit.py", "--tables", table_dir, "--target", str(season - 1),
             "--games", games, "--workers", workers])
    return table_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2022-2025")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--era-games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--team-scale", type=float, default=1.5)
    ap.add_argument("--player-scale", type=float, default=1.5)
    ap.add_argument("--first-week", type=int, default=5)
    args = ap.parse_args()
    first, _, last = args.seasons.partition("-")
    for season in range(int(first), int(last or first) + 1):
        table_dir = prepare(season, args.era_games, args.workers)
        run([sys.executable, "calibration/backtest.py", "--season", season, "--first-week", args.first_week,
             "--sims", args.sims, "--workers", args.workers, "--team-scale", args.team_scale,
             "--player-scale", args.player_scale, "--tables", table_dir, "--tag", "_wf"])


if __name__ == "__main__":
    main()
