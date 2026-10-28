import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import half_window_bucket, half_zone_bucket, late_score_class, late_window2
from clock_context import FAST_PLAY_SECONDS, load_rows
from table_tools import build_levels_table, update_specs
from paths import table_dir

TABLE_DIR = table_dir()
FROM_SEASON = 2018
MAX_SECONDS = 240
MIN_PLAYS = 30

LATE_TIMEOUT_LEVELS = [
    ("L1", ["lwin", "lzone", "lsd", "ldown"]),
    ("L2", ["lwin", "lsd", "ldown"]),
    ("L3", ["lwin", "lsd"]),
    ("L4", ["lwin"]),
    ("L5", ["lwin2"]),
]
SHRINK = 30


def late_frame(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["season"] >= FROM_SEASON]
    stayed_in = ((d["play_type"] == "run") & (d["out_of_bounds"] != 1)) | (
        (d["play_type"] == "pass") & ((d["complete_pass"] == 1) | (d["sack"] == 1)) & (d["out_of_bounds"] != 1))
    q = d[d["is_snap"] & (d["qtr"] == 4) & stayed_in & (d["penalty"] != 1) & (d["touchdown"] != 1)
          & (d["next_off"] == d["posteam"]) & d["next_down"].notna() & d["next_yl"].notna()
          & d["posteam_timeouts_remaining"].notna() & d["defteam_timeouts_remaining"].notna()
          & d["half_seconds_remaining"].notna() & d["score_differential"].notna()].copy()
    q["off_to"] = q["next_is_to"] & (q["next_to_team"] == q["posteam"])
    q["def_to"] = q["next_is_to"] & (q["next_to_team"] == q["defteam"])
    q["clock_end"] = np.where(q["next_is_to"], q["next_clock"], q["half_seconds_remaining"] - FAST_PLAY_SECONDS)
    q = q[q["clock_end"].between(1, MAX_SECONDS)].copy()
    q["lwin"] = half_window_bucket(q["clock_end"])
    q["lzone"] = half_zone_bucket(q["next_yl"])
    q["lsd"] = late_score_class(q["score_differential"])
    q["lwin2"] = late_window2(q["clock_end"])
    q["ldown"] = q["next_down"].clip(upper=4).astype(int)
    return q


def build_all(d: pd.DataFrame):
    q = late_frame(d)
    off = q[(q["posteam_timeouts_remaining"] > 0) & (q["lsd"] != "lead")].copy()
    off["p_timeout"] = off["off_to"].astype(float)
    t_off = build_levels_table(off, LATE_TIMEOUT_LEVELS, ["p_timeout"], min_n=MIN_PLAYS, shrink=SHRINK)

    dfn = q[(q["defteam_timeouts_remaining"] > 0) & ~q["off_to"] & (q["lsd"] != "lead")].copy()
    dfn["p_timeout"] = dfn["def_to"].astype(float)
    t_def = build_levels_table(dfn, LATE_TIMEOUT_LEVELS, ["p_timeout"], min_n=MIN_PLAYS, shrink=SHRINK)
    return q, off, t_off, dfn, t_def


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    q, off, t_off, dfn, t_def = build_all(load_rows())
    for name, table in (("late_timeout_off", t_off), ("late_timeout_def", t_def)):
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        update_specs(TABLE_DIR, name, f"{name}.csv", LATE_TIMEOUT_LEVELS, ["p_timeout"])
        print(f"  {name}.csv: " + ", ".join(f"{lv} {int((table['level'] == lv).sum())}" for lv, _ in LATE_TIMEOUT_LEVELS)
              + " situations kept")
    print(f"\n4th quarter, last 4 minutes, since {FROM_SEASON}: {len(off):,} plays where the offense could call a timeout, "
          f"{len(dfn):,} where a tied or leading defense could")
    print("\nChance of a timeout right after an in bounds play, by clock reading (L5), offense tied or behind:")
    print("  clock left    offense   defense")
    for w in ["w10", "w20", "w30", "w45", "w60", "w90", "w120", "w180", "w240"]:
        row = []
        for t in (t_off, t_def):
            v = t[(t["level"] == "L5") & (t["lwin2"] == w)]["p_timeout"]
            row.append(f"{v.iloc[0]:7.1%}" if len(v) else "    n/a")
        print(f"  {w:>10}   {row[0]}   {row[1]}")
