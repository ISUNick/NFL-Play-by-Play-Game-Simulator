import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_special import flag, load_raw
from buckets import late_distance_bucket, late_score_bucket, late_zone_bucket
from table_tools import build_levels_table, update_specs
from paths import table_dir

TABLE_DIR = table_dir()
OT_FROM_SEASON = 2012
OT_MIN_PLAYS = 20
OT_LEVELS = [("L1", ["late_zone", "late_dist", "late_score"])]


def build_fourth_down_ot(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(df["qtr"] == 5) & (df["down"] == 4) & ~flag(df, "penalty")
           & df["play_type"].isin(["pass", "run", "punt", "field_goal"])
           & ~flag(df, "qb_kneel") & ~flag(df, "qb_spike") & (df["season"] >= OT_FROM_SEASON)].copy()
    f = f.dropna(subset=["ydstogo", "yardline_100", "score_differential"])
    f["late_zone"] = late_zone_bucket(f["yardline_100"])
    f["late_dist"] = late_distance_bucket(f["ydstogo"])
    f["late_score"] = late_score_bucket(f["score_differential"])
    f["p_go"] = f["play_type"].isin(["pass", "run"]).astype(float)
    f["p_punt"] = (f["play_type"] == "punt").astype(float)
    f["p_fg"] = (f["play_type"] == "field_goal").astype(float)
    return f, build_levels_table(f, OT_LEVELS, ["p_go", "p_punt", "p_fg"], min_n=OT_MIN_PLAYS)


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plays, table = build_fourth_down_ot(load_raw())
    table.to_csv(TABLE_DIR / "fourth_down_ot.csv", index=False)
    update_specs(TABLE_DIR, "fourth_down_ot", "fourth_down_ot.csv", OT_LEVELS, ["p_go", "p_punt", "p_fg"])
    print("fourth_down_ot.csv: " + ", ".join(f"{lv} {int((table['level'] == lv).sum())}" for lv, _ in OT_LEVELS)
          + f" situations kept, from {len(plays)} real overtime 4th downs since {OT_FROM_SEASON}")
    print("\nOvertime 4th down choices (real):")
    print("  score         zone   distance     plays     go    punt    kick")
    t = table[table["level"] == "L1"].sort_values(["late_score", "late_zone", "late_dist"])
    for r in t.itertuples():
        print(f"  {r.late_score:<12s}  {r.late_zone:<5s}  {r.late_dist:<8s}  {int(r.n):6d}  {r.p_go:5.0%}  {r.p_punt:6.0%}  {r.p_fg:6.0%}")
