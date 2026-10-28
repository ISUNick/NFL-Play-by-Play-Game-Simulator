from pathlib import Path

import numpy as np
import pandas as pd
from paths import pbp_files

RAW_DIR = Path("data/raw")

COLS = ["game_id", "play_id", "season", "season_type", "qtr", "drive", "posteam", "defteam", "down", "ydstogo",
        "yardline_100", "score_differential", "game_seconds_remaining", "half_seconds_remaining",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining", "play_type", "complete_pass",
        "out_of_bounds", "sack", "qb_scramble", "penalty", "timeout", "timeout_team", "qb_kneel", "qb_spike",
        "touchdown", "home_team", "away_team", "home_opening_kickoff"]


SNAP_TYPES = ["pass", "run", "qb_kneel", "qb_spike", "field_goal", "punt"]
FAST_PLAY_SECONDS = 6


def load_rows(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = pbp_files(raw_dir)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df[df["season_type"] == "REG"].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    df["is_to"] = (df["timeout"] == 1) & (df["play_type"] == "no_play")
    df["is_snap"] = df["play_type"].isin(SNAP_TYPES)
    d = df[df["is_to"] | df["is_snap"]].copy().reset_index(drop=True)
    d["opener"] = d["home_team"].where(d["home_opening_kickoff"] == 1, d["away_team"])

    d["off"] = d["posteam"].where(d["is_snap"])
    d["off"] = d.groupby("game_id")["off"].bfill()
    d["snap_down"] = d["down"].where(d["is_snap"])
    d["nd"] = d.groupby("game_id")["snap_down"].bfill()
    d["snap_yl"] = d["yardline_100"].where(d["is_snap"])
    d["nyl"] = d.groupby("game_id")["snap_yl"].bfill()

    g = d.groupby("game_id")
    for c in ["is_to", "timeout_team", "play_type", "drive", "posteam", "complete_pass",
              "out_of_bounds", "sack", "penalty", "qb_kneel", "qtr", "qb_scramble"]:
        d["prev_" + c] = g[c].shift(1)
    d["next_is_to"] = g["is_to"].shift(-1).fillna(False).astype(bool)
    d["next_to_team"] = g["timeout_team"].shift(-1)
    d["next_clock"] = g["half_seconds_remaining"].shift(-1)
    d["next_type"] = g["play_type"].shift(-1)
    d["next_off"] = g["off"].shift(-1)
    d["next_down"] = g["nd"].shift(-1)
    d["next_yl"] = g["nyl"].shift(-1)

    d["prev_is_to"] = d["prev_is_to"].fillna(False).astype(bool)
    same_drive = ((d["prev_drive"] == d["drive"]) & (d["prev_posteam"] == d["posteam"])
                  & (d["prev_qtr"] == d["qtr"]))
    prev_ran = (
        ((d["prev_play_type"] == "pass") & ((d["prev_complete_pass"] == 1) | (d["prev_sack"] == 1))
         & (d["prev_out_of_bounds"] != 1))
        | ((d["prev_play_type"] == "run") & (d["prev_out_of_bounds"] != 1))
        | (d["prev_qb_kneel"] == 1)
    ) & (d["prev_penalty"] != 1)
    d["running"] = np.where(d["prev_is_to"], 0.0, np.where(same_drive, prev_ran.astype(float), np.nan))
    return d
