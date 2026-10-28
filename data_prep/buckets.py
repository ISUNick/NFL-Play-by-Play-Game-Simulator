
import numpy as np
import pandas as pd
from paths import pbp_files


def distance_bucket(ydstogo: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [ydstogo <= 2, ydstogo <= 6, ydstogo <= 10],
            ["short", "medium", "long"],
            default="very_long",
        ),
        index=ydstogo.index,
    )


def zone_bucket(yardline_100: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [yardline_100 >= 90, yardline_100 >= 80, yardline_100 >= 50,
             yardline_100 > 20, yardline_100 > 5],
            ["own_10", "backed_up", "own_half", "opp_half", "red_zone"],
            default="goal_line",
        ),
        index=yardline_100.index,
    )


def score_bucket(score_differential: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [score_differential <= -9, score_differential <= -1,
             score_differential == 0, score_differential <= 8],
            ["trailing_big", "trailing_close", "tied", "leading_close"],
            default="leading_big",
        ),
        index=score_differential.index,
    )


def time_bucket(game_seconds_remaining: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(game_seconds_remaining <= 300, "late", "normal"),
        index=game_seconds_remaining.index,
    )


def fourth_distance_bucket(ydstogo: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([ydstogo <= 1, ydstogo <= 2, ydstogo <= 4, ydstogo <= 7, ydstogo <= 10],
                  ["1", "2", "3_4", "5_7", "8_10"], default="11_plus"),
        index=ydstogo.index,
    )


def fourth_zone_bucket(yardline_100: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([yardline_100 <= 5, yardline_100 <= 20, yardline_100 <= 30, yardline_100 <= 36,
                   yardline_100 <= 40, yardline_100 <= 50, yardline_100 <= 65, yardline_100 <= 80],
                  ["z1_5", "z6_20", "z21_30", "z31_36", "z37_40", "z41_50", "z51_65", "z66_80"],
                  default="z81_99"),
        index=yardline_100.index,
    )


def tempo_bucket(qtr: pd.Series, half_seconds_remaining: pd.Series) -> pd.Series:
    end_of_half = qtr.isin([2, 4, 5])
    return pd.Series(
        np.select([end_of_half & (half_seconds_remaining <= 120),
                   end_of_half & (half_seconds_remaining <= 300)],
                  ["last2", "last5"], default="normal"),
        index=qtr.index,
    )


def late_time_bucket(qtr: pd.Series, game_seconds_remaining: pd.Series) -> pd.Series:
    q4, q2 = (qtr == 4), (qtr == 2)
    s = np.where(q2, game_seconds_remaining - 1800, game_seconds_remaining)
    s = pd.Series(s, index=qtr.index)
    in_window = (q4 | q2) & (s > 0) & (s <= 420)
    last2, last5 = in_window & (s <= 120), in_window & (s <= 300)
    return pd.Series(
        np.select([q4 & last2, q4 & last5, q4 & in_window, q2 & last2, q2 & last5, q2 & in_window],
                  ["q4_last2", "q4_last5", "q4_last7", "q2_last2", "q2_last5", "q2_last7"], default="none"),
        index=qtr.index,
    )


def late_score_bucket(score_differential: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([score_differential <= -17, score_differential <= -9, score_differential <= -4,
                   score_differential <= -1, score_differential == 0, score_differential <= 8],
                  ["trail_17p", "trail_9_16", "trail_4_8", "trail_1_3", "tied", "lead_1_8"], default="lead_9p"),
        index=score_differential.index,
    )


def late_zone_bucket(yardline_100: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([yardline_100 <= 10, yardline_100 <= 35, yardline_100 <= 45, yardline_100 <= 65],
                  ["goal", "fg", "edge", "mid"], default="own"),
        index=yardline_100.index,
    )


def late_zone2_bucket(yardline_100: pd.Series) -> pd.Series:
    return pd.Series(np.where(yardline_100 <= 45, "near", "far"), index=yardline_100.index)


def late_distance_bucket(ydstogo: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([ydstogo <= 2, ydstogo <= 6], ["short", "medium"], default="long"),
        index=ydstogo.index,
    )


def fourth_late_distance_bucket(ydstogo: pd.Series) -> pd.Series:
    return pd.Series(np.select([ydstogo <= 1, ydstogo <= 2, ydstogo <= 6], ["1", "2", "3_6"], default="7p"),
                     index=ydstogo.index)


def late_quarter_bucket(late_time: pd.Series) -> pd.Series:
    return late_time.str[:2]


HALF_WINDOW_LIMITS = [3, 6, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240]


def half_window_bucket(seconds_left: pd.Series) -> pd.Series:
    conditions = [(seconds_left > 0) & (seconds_left <= lim) for lim in HALF_WINDOW_LIMITS]
    return pd.Series(np.select(conditions, [f"s{lim}" for lim in HALF_WINDOW_LIMITS], default="none"),
                     index=seconds_left.index)


def half_zone_bucket(yardline_100: pd.Series) -> pd.Series:
    return pd.Series(np.select([yardline_100 <= 37, yardline_100 <= 55, yardline_100 <= 75],
                               ["h_fg", "h_mid", "h_own"], default="h_deep"), index=yardline_100.index)


def late_score_class(score_diff: pd.Series) -> pd.Series:
    return pd.Series(np.select([score_diff == 0, score_diff >= 1, score_diff >= -3, score_diff >= -8, score_diff >= -16],
                               ["tied", "lead", "t1_3", "t4_8", "t9_16"], default="t17p"), index=score_diff.index)


LATE_WINDOW2_LIMITS = [10, 20, 30, 45, 60, 90, 120, 180, 240]


def late_window2(seconds_left: pd.Series) -> pd.Series:
    conditions = [(seconds_left > 0) & (seconds_left <= lim) for lim in LATE_WINDOW2_LIMITS]
    return pd.Series(np.select(conditions, [f"w{lim}" for lim in LATE_WINDOW2_LIMITS], default="none"),
                     index=seconds_left.index)


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dist_b"] = distance_bucket(out["ydstogo"])
    out["zone_b"] = zone_bucket(out["yardline_100"])
    out["score_b"] = score_bucket(out["score_differential"])
    out["time_b"] = time_bucket(out["game_seconds_remaining"])
    return out


if __name__ == "__main__":
    cols = ["down", "ydstogo", "yardline_100", "score_differential",
            "game_seconds_remaining", "play_type"]
    files = pbp_files()
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    df = df[df["play_type"].isin(["pass", "run"])].dropna(subset=cols)
    df = add_buckets(df)
    for col in ["down", "dist_b", "zone_b", "score_b", "time_b"]:
        print(df[col].value_counts().sort_index(), "\n")
    combos = df.groupby(["down", "dist_b", "zone_b", "score_b", "time_b"]).size()
    print("Total situation combos:", len(combos))
    print("Smallest combo size:", combos.min())
    print("Combos under 100 plays:", (combos < 100).sum())
