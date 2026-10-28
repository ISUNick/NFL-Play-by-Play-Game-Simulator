import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import tempo_bucket
from build_clock import RAW_DIR
from paths import pbp_files, table_dir

TABLE_DIR = table_dir()
FROM_SEASON = 2018
MAX_D2 = 12
COLS = ["game_id", "play_id", "season", "qtr", "game_seconds_remaining", "half_seconds_remaining", "play_type",
        "timeout", "timeout_team", "posteam", "defteam", "out_of_bounds", "complete_pass", "sack",
        "incomplete_pass", "penalty", "touchdown", "two_point_attempt"]


def load() -> pd.DataFrame:
    files = pbp_files(RAW_DIR)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df[df["season"] >= FROM_SEASON].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g = df.groupby("game_id")
    for c in ["game_seconds_remaining", "qtr", "timeout", "timeout_team", "play_type"]:
        df["next_" + c] = g[c].shift(-1)
    df["after_gs"] = g["game_seconds_remaining"].shift(-2)
    df["after_qtr"] = g["qtr"].shift(-2)
    df["after_timeout"] = g["timeout"].shift(-2)
    return df


def build(df: pd.DataFrame):
    f = lambda c: df[c].fillna(0) == 1
    running = (((df["play_type"] == "run") & ~f("out_of_bounds"))
               | ((df["play_type"] == "pass") & (f("complete_pass") | f("sack")) & ~f("out_of_bounds")))
    ok = (running & ~f("penalty") & ~f("touchdown") & ~f("two_point_attempt") & (df["qtr"] <= 4)
          & df["half_seconds_remaining"].notna() & df["posteam"].notna())
    df = df.assign(tempo=tempo_bucket(df["qtr"], df["half_seconds_remaining"]))
    plays = df[ok].copy()
    called = (plays["next_timeout"] == 1) & (plays["next_qtr"] == plays["qtr"])
    plays["p_off"] = (called & (plays["next_timeout_team"] == plays["posteam"])).astype(float)
    plays["p_def"] = (called & (plays["next_timeout_team"] == plays["defteam"])).astype(float)

    early = plays[plays["tempo"] == "normal"]
    rate = pd.DataFrame([{"p_off": early["p_off"].mean(), "p_def": early["p_def"].mean(), "n": len(early)}])

    t = plays[called & (plays["after_qtr"] == plays["qtr"]) & (plays["after_timeout"].fillna(0) != 1)].copy()
    t["d1"] = t["game_seconds_remaining"] - t["next_game_seconds_remaining"]
    t["d2"] = t["next_game_seconds_remaining"] - t["after_gs"]
    t = t[t["d1"].between(0, 60) & t["d2"].between(0, MAX_D2)]
    bank = t[["tempo", "d1", "d2"]].astype({"d1": int, "d2": int}).reset_index(drop=True)
    return rate, bank, plays


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rate, bank, plays = build(load())
    rate.to_csv(TABLE_DIR / "timeout_early.csv", index=False)
    bank.to_csv(TABLE_DIR / "bank_timeout.csv", index=False)
    games = plays["game_id"].nunique()
    print(f"timeout_early.csv: outside the last 5 minutes of a half, after an in bounds running clock play "
          f"the offense calls a timeout {rate.p_off[0]:.2%} and the defense {rate.p_def[0]:.2%} ({int(rate.n[0]):,} plays)")
    print(f"bank_timeout.csv: {len(bank):,} real timeouts after running clock plays")
    print("\nSeconds of clock before the timeout (d1, from the snap) and from the timeout to the next snap (d2):")
    print(bank.groupby("tempo")[["d1", "d2"]].agg(["count", "mean"]).round(1).to_string())
