import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import score_bucket, tempo_bucket
from table_tools import update_specs
from paths import pbp_files, table_dir

RAW_DIR = Path("data/raw")
TABLE_DIR = table_dir()
MIN_CELL = 100
MAX_SAMPLES = 1000
SEED = 42
MAX_INCOMPLETE = 12
MAX_INTERVAL = 120
LEVELS = [("L1", ["kind", "tempo", "score_b"]), ("L2", ["kind", "tempo"]), ("L3", ["kind"])]

COLS = ["game_id", "play_id", "qtr", "posteam", "play_type", "complete_pass", "incomplete_pass",
        "out_of_bounds", "sack", "interception", "fumble_lost", "penalty", "timeout",
        "touchdown", "qb_scramble", "two_point_attempt", "game_seconds_remaining",
        "half_seconds_remaining", "score_differential"]


def load_clock_plays() -> pd.DataFrame:
    files = pbp_files(RAW_DIR)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g = df.groupby("game_id")
    for c in ["timeout", "play_type", "half_seconds_remaining"]:
        df["next_" + c] = g[c].shift(-1)
    warning_next = df["next_play_type"].isna() & (df["next_half_seconds_remaining"] == 120)

    for c, name in (("game_seconds_remaining", "next_real_seconds"), ("qtr", "next_real_qtr")):
        df[name] = df.groupby("game_id")[c].shift(-1)
    df["seconds"] = df["game_seconds_remaining"] - df["next_real_seconds"]
    df["next_qtr"] = df["next_real_qtr"]

    f = lambda c: df[c].fillna(0) == 1
    pt = df["play_type"]
    two_pt = f("two_point_attempt")
    is_pass = (pt == "pass") & ~two_pt
    is_run = (pt == "run") & ~two_pt
    scramble, oob = f("qb_scramble"), f("out_of_bounds")
    df["kind"] = np.select(
        [is_pass & f("sack"), is_pass & f("interception"), is_pass & f("touchdown"),
         is_pass & f("complete_pass") & oob, is_pass & f("complete_pass"), is_pass & f("incomplete_pass"),
         is_run & f("touchdown"), is_run & f("fumble_lost"),
         is_run & scramble & oob, is_run & scramble, is_run & oob, is_run,
         pt == "punt", pt == "field_goal", pt == "kickoff", pt == "extra_point"],
        ["sack", "interception", "pass_td", "complete_oob", "complete_in", "incomplete",
         "run_td", "run_fumble", "scramble_oob", "scramble_in", "run_oob", "run_in",
         "punt", "field_goal", "kickoff", "extra_point"],
        default="other")

    keep = ((df["kind"] != "other") & (df["qtr"] <= 4) & (df["next_qtr"] == df["qtr"])
            & ~f("timeout") & (df["next_timeout"].fillna(0) != 1) & ~warning_next
            & df["seconds"].between(0, MAX_INTERVAL)
            & ~((df["kind"] == "incomplete") & (df["seconds"] > MAX_INCOMPLETE))
            & df["half_seconds_remaining"].notna() & df["score_differential"].notna())
    df = df[keep].copy()
    df["tempo"] = tempo_bucket(df["qtr"], df["half_seconds_remaining"])
    df["score_b"] = score_bucket(df["score_differential"])
    df["seconds"] = df["seconds"].astype(int)
    return df


def build_time_bank(df: pd.DataFrame, rng) -> pd.DataFrame:
    keys_all = ["kind", "tempo", "score_b"]
    pieces = []
    for level, keys in LEVELS:
        for values, sub in df.groupby(keys):
            if len(sub) < MIN_CELL:
                continue
            values = values if isinstance(values, tuple) else (values,)
            secs = sub["seconds"].to_numpy()
            if len(secs) > MAX_SAMPLES:
                secs = rng.choice(secs, MAX_SAMPLES, replace=False)
            row = dict(zip(keys, values))
            pieces.append(pd.DataFrame({"level": level,
                                        **{k: row.get(k, "ALL") for k in keys_all},
                                        "seconds": secs}))
    return pd.concat(pieces, ignore_index=True)


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plays = load_clock_plays()
    print(f"Plays used for timing: {len(plays):,}")
    bank = build_time_bank(plays, np.random.default_rng(SEED))
    bank.to_csv(TABLE_DIR / "bank_time.csv", index=False)
    update_specs(TABLE_DIR, "bank_time", "bank_time.csv", LEVELS, ["seconds"])
    print(f"bank_time.csv: {len(bank):,} rows")

    print("\nSeconds off the game clock, by kind (normal tempo, all scores):")
    n = plays[plays["tempo"] == "normal"]
    print(n.groupby("kind")["seconds"].agg(["count", "median", "mean"]).round(1).to_string())

    print("\nIn bounds run, median seconds by tempo and score:")
    r = plays[plays["kind"] == "run_in"]
    print(r.pivot_table(index="tempo", columns="score_b", values="seconds", aggfunc="median").round(0).to_string())
