import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import half_window_bucket, half_zone_bucket
from clock_context import FAST_PLAY_SECONDS, load_rows
from table_tools import build_levels_table, update_specs
from paths import table_dir

TABLE_DIR = table_dir()
SNAP_MAX_SECONDS = 120
TIMEOUT_MAX_SECONDS = 240
SNAP_MIN_PLAYS = 60
TIMEOUT_MIN_PLAYS = 40

HALF_SNAP_LEVELS = [
    ("L1", ["hwin", "rs", "hzone", "dto", "has_to", "hdown"]),
    ("L2", ["hwin", "rs", "hzone", "dto", "has_to"]),
    ("L3", ["hwin", "rs", "hzone", "has_to"]),
    ("L4", ["hwin", "rs", "hzone"]),
    ("L5", ["hwin", "hzone"]),
    ("L6", ["hwin"]),
]
HALF_TIMEOUT_LEVELS = [("L1", ["hwin", "hzone", "hdown"]), ("L2", ["hwin", "hzone"]), ("L3", ["hwin"])]
HALF_SPIKE_LEVELS = [("L1", ["hwin", "hzone", "hdown"]), ("L2", ["hwin", "hdown"]), ("L3", ["hwin"])]
SPIKE_MIN_PLAYS = 40


def snap_frame(d: pd.DataFrame) -> pd.DataFrame:
    s = d[d["is_snap"] & (d["qtr"] == 2) & d["half_seconds_remaining"].between(1, SNAP_MAX_SECONDS)
          & d["down"].isin([1, 2, 3]) & d["yardline_100"].notna()
          & d["posteam_timeouts_remaining"].notna() & d["defteam_timeouts_remaining"].notna()].copy()
    s["hwin"] = half_window_bucket(s["half_seconds_remaining"])
    s["rs"] = np.where(s["running"].isna(), "newposs", np.where(s["running"] == 1, "running", "stopped"))
    s["hzone"] = half_zone_bucket(s["yardline_100"])
    s["dto"] = s["defteam_timeouts_remaining"].clip(upper=2).astype(int)
    s["has_to"] = (s["posteam_timeouts_remaining"] > 0).astype(int)
    s["hdown"] = s["down"].astype(int)
    s["p_kneel"] = (s["play_type"] == "qb_kneel").astype(float)
    s["p_spike"] = (s["play_type"] == "qb_spike").astype(float)
    s["p_fg"] = (s["play_type"] == "field_goal").astype(float)
    return s


def timeout_frame(d: pd.DataFrame) -> pd.DataFrame:
    stayed_in = ((d["play_type"] == "run") & (d["out_of_bounds"] != 1)) | (
        (d["play_type"] == "pass") & ((d["complete_pass"] == 1) | (d["sack"] == 1)) & (d["out_of_bounds"] != 1))
    q = d[d["is_snap"] & (d["qtr"] == 2) & stayed_in & (d["penalty"] != 1) & (d["touchdown"] != 1)
          & (d["next_off"] == d["posteam"]) & d["next_down"].notna() & d["next_yl"].notna()
          & d["posteam_timeouts_remaining"].notna() & d["defteam_timeouts_remaining"].notna()
          & d["half_seconds_remaining"].notna()].copy()
    q["off_to"] = q["next_is_to"] & (q["next_to_team"] == q["posteam"])
    q["def_to"] = q["next_is_to"] & (q["next_to_team"] == q["defteam"])
    q["clock_end"] = np.where(q["next_is_to"], q["next_clock"], q["half_seconds_remaining"] - FAST_PLAY_SECONDS)
    q = q[q["clock_end"].between(1, TIMEOUT_MAX_SECONDS)].copy()
    q["hwin"] = half_window_bucket(q["clock_end"])
    q["hzone"] = half_zone_bucket(q["next_yl"])
    q["hdown"] = q["next_down"].clip(upper=4).astype(int)
    return q


def build_all(d: pd.DataFrame):
    snap = snap_frame(d)
    half_snap = build_levels_table(snap, HALF_SNAP_LEVELS, ["p_kneel", "p_spike", "p_fg"], min_n=SNAP_MIN_PLAYS)

    q = timeout_frame(d)
    off = q[q["posteam_timeouts_remaining"] > 0].copy()
    off["p_timeout"] = off["off_to"].astype(float)
    half_off = build_levels_table(off, HALF_TIMEOUT_LEVELS, ["p_timeout"], min_n=TIMEOUT_MIN_PLAYS)

    dfn = q[(q["defteam_timeouts_remaining"] > 0) & ~q["off_to"]].copy()
    dfn["p_timeout"] = dfn["def_to"].astype(float)
    half_def = build_levels_table(dfn, HALF_TIMEOUT_LEVELS, ["p_timeout"], min_n=TIMEOUT_MIN_PLAYS)
    nt = q[(q["posteam_timeouts_remaining"] == 0) & ~q["next_is_to"] & (q["hdown"] <= 3)].copy()
    nt["p_spike"] = (nt["next_type"] == "qb_spike").astype(float)
    half_spike = build_levels_table(nt, HALF_SPIKE_LEVELS, ["p_spike"], min_n=SPIKE_MIN_PLAYS)
    sp = nt[nt["p_spike"] == 1].copy()
    sp["runoff"] = (sp["half_seconds_remaining"] - FAST_PLAY_SECONDS - sp["next_clock"]).clip(lower=0)
    spike_runoff = sp[sp["runoff"].between(0, 40)][["runoff"]].astype(int).reset_index(drop=True)
    return snap, half_snap, off, half_off, dfn, half_def, nt, half_spike, spike_runoff


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    snap, half_snap, off, half_off, dfn, half_def, nt, half_spike, spike_runoff = build_all(rows)
    for name, table, levels, outcomes in [
        ("half_snap", half_snap, HALF_SNAP_LEVELS, ["p_kneel", "p_spike", "p_fg"]),
        ("half_timeout_off", half_off, HALF_TIMEOUT_LEVELS, ["p_timeout"]),
        ("half_timeout_def", half_def, HALF_TIMEOUT_LEVELS, ["p_timeout"]),
        ("half_spike", half_spike, HALF_SPIKE_LEVELS, ["p_spike"]),
    ]:
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        update_specs(TABLE_DIR, name, f"{name}.csv", levels, outcomes)
        print(f"  {name}.csv: " + ", ".join(f"{lv} {int((table['level'] == lv).sum())}" for lv, _ in levels)
              + " situations kept")
    spike_runoff.to_csv(TABLE_DIR / "bank_spike_runoff.csv", index=False)
    print(f"  bank_spike_runoff.csv: {len(spike_runoff):,} real spikes, median {spike_runoff['runoff'].median():.0f} s of clock "
          f"between the end of the play and the spike")
    print(f"\nEnd of half: {len(snap):,} real snaps, {len(off):,} plays where the offense could call a timeout, "
          f"{len(dfn):,} where the defense could")
    print(f"  per game: kneels {snap['p_kneel'].sum() / rows['game_id'].nunique():.3f}, "
          f"spikes {snap['p_spike'].sum() / rows['game_id'].nunique():.3f}")
    print("\nChance of each timeout by seconds left in the half (all positions):")
    print("  seconds left   offense timeout   defense timeout")
    for w in ["s14", "s30", "s60", "s120", "s240"]:
        o = half_off[(half_off["level"] == "L3") & (half_off["hwin"] == w)]["p_timeout"]
        f = half_def[(half_def["level"] == "L3") & (half_def["hwin"] == w)]["p_timeout"]
        print(f"  {w:>12}   {o.iloc[0] if len(o) else float('nan'):15.1%}   {f.iloc[0] if len(f) else float('nan'):15.1%}")
    print("\nChance the next snap is a spike when the offense has no timeout (L3, all positions):")
    for w in ["s14", "s20", "s30", "s45", "s60", "s90"]:
        v = half_spike[(half_spike["level"] == "L3") & (half_spike["hwin"] == w)]["p_spike"]
        print(f"  {w:>12}   {v.iloc[0] if len(v) else float('nan'):6.1%}")
