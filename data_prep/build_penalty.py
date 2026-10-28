import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import add_buckets
from build_clock import RAW_DIR
from table_tools import build_levels_table, update_specs
from paths import pbp_files, table_dir

TABLE_DIR = table_dir()
PENALTY_FROM_SEASON = 2018
RATE_MIN_PLAYS = 150
RATE_SHRINK = 30
PRE_LEVELS = [("L1", ["down", "dist_b", "zone_b"]), ("L2", ["down", "dist_b"]), ("L3", ["down"])]
POST_LEVELS = [("L1", ["kind", "zone_b", "down"]), ("L2", ["kind", "zone_b"]), ("L3", ["kind"])]
MAX_SECONDS = 120
RUN_TEXT = r" (?:left|right) (?:end|tackle|guard)|up the middle"

COLS = ["game_id", "play_id", "season", "qtr", "down", "ydstogo", "yardline_100", "posteam", "play_type",
        "penalty", "penalty_team", "penalty_type", "timeout", "two_point_attempt", "sack", "qb_scramble",
        "pass_length", "desc", "game_seconds_remaining", "score_differential"]


KICK_FOUL_TEXT = "Kick|Punt|Fair Catch"


def load_snaps() -> pd.DataFrame:
    files = pbp_files(RAW_DIR)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g = df.groupby("game_id")
    for c in ["down", "ydstogo", "yardline_100", "posteam", "qtr", "game_seconds_remaining"]:
        df["next_" + c] = g[c].shift(-1)
    return df


def penalty_kind(desc: pd.Series) -> pd.Series:
    d = desc.fillna("")
    is_pass = d.str.contains(" pass ", regex=False)
    return pd.Series(np.select(
        [d.str.contains("sacked", regex=False), d.str.contains("scrambles", regex=False),
         is_pass & d.str.contains(" deep", regex=False), is_pass, d.str.contains(RUN_TEXT, regex=True)],
        ["sack", "scramble", "pass_deep", "pass_short", "run"], default="pre"), index=desc.index)


def play_kind(df: pd.DataFrame) -> pd.Series:
    f = lambda c: df[c].fillna(0) == 1
    return pd.Series(np.select(
        [df["play_type"].isin(["punt", "field_goal"]), f("sack"), f("qb_scramble"), df["play_type"] == "run",
         (df["play_type"] == "pass") & (df["pass_length"] == "deep")],
        ["kick", "sack", "scramble", "run", "pass_deep"], default="pass_short"), index=df.index)


def build_all(df: pd.DataFrame):
    f = lambda c: df[c].fillna(0) == 1
    base = (df["down"].isin([1, 2, 3, 4]) & (df["season"] >= PENALTY_FROM_SEASON) & ~f("timeout")
            & ~f("two_point_attempt") & df["yardline_100"].notna() & df["ydstogo"].notna()
            & df["score_differential"].notna() & df["game_seconds_remaining"].notna())
    plays = base & df["play_type"].isin(["run", "pass", "punt", "field_goal"])

    flagged = base & (df["play_type"] == "no_play") & f("penalty") & df["penalty_team"].notna()
    kept_team = df["next_posteam"] == df["posteam"]
    first = (df["next_down"] == 1) & (df["down"] > 1)
    same = df["next_down"] == df["down"]
    kick_foul = df["penalty_type"].fillna("").str.contains(KICK_FOUL_TEXT, case=False, regex=True)
    ok = flagged & ~kick_foul & kept_team & df["next_yardline_100"].notna() & (first | same)

    work = df[plays | ok].copy()
    work = work.join(add_buckets(work)[["dist_b", "zone_b"]])
    work["down"] = work["down"].astype(int)
    work["is_pen"] = ok.loc[work.index]
    work["kind"] = np.where(work["is_pen"], penalty_kind(work["desc"]), play_kind(work))
    work["off"] = (work["is_pen"] & (work["penalty_team"] == work["posteam"])).astype(float)
    work["deff"] = (work["is_pen"] & (work["penalty_team"] != work["posteam"])).astype(float)
    work["p_off"], work["p_def"] = work["off"], work["deff"]

    pre_pen = work["is_pen"] & (work["kind"] == "pre")
    pre_rows = work.copy()
    pre_rows["p_off"] = pre_rows["off"].where(pre_pen, 0.0)
    pre_rows["p_def"] = pre_rows["deff"].where(pre_pen, 0.0)
    pre = build_levels_table(pre_rows, PRE_LEVELS, ["p_off", "p_def"], min_n=RATE_MIN_PLAYS, shrink=RATE_SHRINK)

    post_rows = work[~pre_pen & (work["kind"] != "kick")].copy()
    post_rows["p_off"] = post_rows["off"].where(post_rows["is_pen"], 0.0)
    post_rows["p_def"] = post_rows["deff"].where(post_rows["is_pen"], 0.0)
    post = build_levels_table(post_rows, POST_LEVELS, ["p_off", "p_def"], min_n=RATE_MIN_PLAYS, shrink=RATE_SHRINK)

    p = df[ok].copy()
    p = p.join(add_buckets(p)[["zone_b"]])
    p["kind"] = penalty_kind(p["desc"])
    secs = p["game_seconds_remaining"] - p["next_game_seconds_remaining"]
    p["seconds"] = np.where((p["next_qtr"] == p["qtr"]) & secs.between(0, MAX_SECONDS), secs, np.nan)
    p["side"] = np.where(p["penalty_team"] == p["posteam"], "off", "def")
    p["dy"] = (p["next_yardline_100"] - p["yardline_100"]).astype(int)
    p["down"] = p["down"].astype(int)
    distance_reset = p["next_ydstogo"] >= np.minimum(10, p["next_yardline_100"])
    p["rel"] = np.where((p["next_down"] == 1) & ((p["down"] > 1) | ((p["side"] == "def") & distance_reset)),
                        "first", "same")
    p["foul"] = p["penalty_type"].fillna("Penalty").str.replace(",", "", regex=False)
    p = p[~((p["side"] == "off") & (p["rel"] == "first"))]
    p = p.dropna(subset=["seconds"])
    p["seconds"] = p["seconds"].astype(int)
    bank = p[["side", "kind", "down", "zone_b", "rel", "dy", "seconds", "foul"]].reset_index(drop=True)
    return work, pre, post, bank


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    work, pre, post, bank = build_all(load_snaps())
    pre.to_csv(TABLE_DIR / "penalty_pre.csv", index=False)
    post.to_csv(TABLE_DIR / "penalty_post.csv", index=False)
    bank.to_csv(TABLE_DIR / "bank_penalty.csv", index=False)
    update_specs(TABLE_DIR, "penalty_pre", "penalty_pre.csv", PRE_LEVELS, ["p_off", "p_def"])
    update_specs(TABLE_DIR, "penalty_post", "penalty_post.csv", POST_LEVELS, ["p_off", "p_def"])
    games = work["game_id"].nunique()
    print("penalty_pre.csv: " + ", ".join(f"{lv} {int((pre['level'] == lv).sum())}" for lv, _ in PRE_LEVELS)
          + " situations;  penalty_post.csv: " + ", ".join(f"{lv} {int((post['level'] == lv).sum())}" for lv, _ in POST_LEVELS))
    print(f"bank_penalty.csv: {len(bank):,} real penalties since {PENALTY_FROM_SEASON}")
    print(f"\nPer game: {len(bank) / games:.2f} penalties kept (offense {(bank.side == 'off').sum() / games:.2f}, "
          f"defense {(bank.side == 'def').sum() / games:.2f}), {(bank.kind == 'pre').mean():.0%} before the snap, "
          f"mean {bank.seconds.mean():.1f} seconds each; {(bank.rel == 'first').mean():.1%} were automatic first downs")
    print("\nPenalty chance per attempt, by kind of play (L3), wiped out AFTER the snap:")
    print("  kind          offense   defense      plays")
    for r in post[post["level"] == "L3"].sort_values("kind").itertuples():
        print(f"  {r.kind:<12s}  {r.p_off:6.2%}   {r.p_def:6.2%}   {int(r.n):9,}")
    print("\nDefensive penalties after the snap by field zone (L2), pass_deep vs run:")
    for kind in ("pass_deep", "pass_short", "run"):
        t = post[(post["level"] == "L2") & (post["kind"] == kind)].sort_values("zone_b")
        print(f"  {kind:<10s} " + "  ".join(f"{r.zone_b} {r.p_def:.1%}" for r in t.itertuples()))
    print("\nBefore the snap, by down (L3):")
    for r in pre[pre["level"] == "L3"].sort_values("down").itertuples():
        print(f"  down {r.down}: offense {r.p_off:.2%}, defense {r.p_def:.2%}")
    print("\nWhat the penalties did, by play kind (yards the ball moved, positive is backwards):")
    print(bank.groupby(["kind", "side"])["dy"].agg(["count", "mean", "median"]).round(1).to_string())
    print("\nMost common foul by kind:")
    for kind, t in bank.groupby("kind"):
        print(f"  {kind:<10s} " + ", ".join(f"{k} {v / len(t):.0%}" for k, v in t["foul"].value_counts().head(3).items()))
