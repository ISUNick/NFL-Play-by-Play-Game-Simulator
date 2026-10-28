import os
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
FULL_SEASON_GAMES = 250
SEASONS_USED = 4


def table_dir() -> Path:
    # tables are written here, and the engine reads them from here (SIM_TABLE_DIR changes it for a backtest)
    return Path(os.environ.get("SIM_TABLE_DIR", "data/tables"))


def table_seasons() -> list:
    # SIM_TABLE_SEASONS="2021-2024" picks the years. Otherwise the last four complete seasons
    wanted = os.environ.get("SIM_TABLE_SEASONS")
    if wanted:
        first, last = (int(x) for x in wanted.split("-"))
        return list(range(first, last + 1))
    complete = []
    for f in sorted(RAW_DIR.glob("pbp_*.parquet")):
        games = pd.read_parquet(f, columns=["game_id", "season_type"]).query("season_type == 'REG'").game_id.nunique()
        if games >= FULL_SEASON_GAMES:
            complete.append(int(f.stem.split("_")[1]))
    return complete[-SEASONS_USED:]


def pbp_files(raw_dir=RAW_DIR) -> list:
    return [Path(raw_dir) / f"pbp_{s}.parquet" for s in table_seasons()]
