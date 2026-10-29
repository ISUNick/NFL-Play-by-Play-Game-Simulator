import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paths import pbp_files, table_dir

TIME_EDGES = [0, 5, 10, 20, 30, 45, 60]
ZONE_EDGES = [0, 10, 30, 38, 50]


def final_minute_calls() -> pd.DataFrame:
    # how often a trailing offense drops back in the last minute. Late in the game it is nearly always
    cols = ["season_type", "qtr", "game_seconds_remaining", "score_differential", "down", "play_type", "qb_dropback", "penalty"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in pbp_files()])
    d = d[(d.season_type == "REG") & (d.qtr == 4) & d.play_type.isin(["pass", "run"]) & (d.penalty.fillna(0) != 1)
          & d.down.isin([1, 2, 3, 4]) & (d.score_differential < 0) & d.game_seconds_remaining.between(1, 60)]
    d["time_f"] = pd.cut(d.game_seconds_remaining, TIME_EDGES, labels=False)
    d["trail_f"] = (d.score_differential < -8).astype(int)
    out = d.groupby(["time_f", "trail_f"]).qb_dropback.agg(n="size", p_dropback="mean").reset_index()
    return out


def half_end_fourth_downs() -> pd.DataFrame:
    # fourth downs in the last seconds of the second quarter: nearly everyone in range kicks, whatever the score
    cols = ["season_type", "qtr", "half_seconds_remaining", "down", "yardline_100", "play_type", "penalty"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in pbp_files()])
    d = d[(d.season_type == "REG") & (d.qtr == 2) & (d.down == 4) & d.half_seconds_remaining.between(1, 12)
          & (d.yardline_100 <= 50) & (d.penalty.fillna(0) != 1) & d.play_type.isin(["pass", "run", "field_goal", "punt"])]
    d["near"] = (d.half_seconds_remaining <= 5).astype(int)
    d["zone"] = pd.cut(d.yardline_100, ZONE_EDGES, labels=False)
    d["fg"], d["go"] = d.play_type == "field_goal", d.play_type.isin(["pass", "run"])
    return d.groupby(["near", "zone"]).agg(n=("fg", "size"), p_fg=("fg", "mean"), p_go=("go", "mean")).reset_index()


HURRY_KINDS = ("run", "complete", "sack")
EVENT_TYPES = ["pass", "run", "no_play", "field_goal", "punt", "qb_spike", "qb_kneel"]


def two_minute_hurry() -> pd.DataFrame:
    # seconds from snap to snap when a team down 1 to 8 runs its two minute drill and the clock keeps running.
    # The normal timing bank mixes in teams that are not hurrying, so this spot gets its own
    cols = ["game_id", "play_id", "season_type", "qtr", "half_seconds_remaining", "score_differential", "posteam",
            "play_type", "out_of_bounds", "complete_pass", "sack", "penalty", "touchdown", "interception",
            "fumble_lost", "timeout", "posteam_timeouts_remaining"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in pbp_files()])
    d = d[(d.season_type == "REG") & d.play_type.isin(EVENT_TYPES)].sort_values(["game_id", "play_id"])
    nxt = d.groupby("game_id").shift(-1)
    in_bounds = (d.out_of_bounds.fillna(0) != 1) & ((d.play_type == "run") | (d.complete_pass == 1) | (d.sack == 1))
    clean = (d.penalty.fillna(0) != 1) & (d.touchdown.fillna(0) != 1) & (d.interception.fillna(0) != 1) & (d.fumble_lost.fillna(0) != 1)
    keep = ((d.qtr == 4) & d.play_type.isin(["pass", "run"]) & in_bounds & clean
            & d.score_differential.between(-8, -1) & d.half_seconds_remaining.between(1, 120)
            & (nxt.qtr == 4) & (nxt.timeout.fillna(0) != 1) & (nxt.posteam == d.posteam) & (nxt.half_seconds_remaining > 0))
    q = d[keep].assign(seconds=(d.half_seconds_remaining - nxt.half_seconds_remaining)[keep])
    q = q[q.seconds.between(1, 60)]
    q["kind"] = np.where(q.sack == 1, "sack", np.where(q.play_type == "run", "run", "complete"))
    q["has_to"] = (q.posteam_timeouts_remaining > 0).astype(int)
    return q[["kind", "has_to", "seconds"]].astype({"seconds": int})


if __name__ == "__main__":
    table = final_minute_calls()
    table.to_csv(table_dir() / "play_call_final.csv", index=False)
    print(table.to_string(index=False))
    half = half_end_fourth_downs()
    half.to_csv(table_dir() / "fourth_half_end.csv", index=False)
    print(half.to_string(index=False))
    hurry = two_minute_hurry()
    hurry.to_csv(table_dir() / "bank_hurry.csv", index=False)
    print(hurry.groupby(["kind", "has_to"]).seconds.describe()[["count", "25%", "50%", "75%"]].to_string())
