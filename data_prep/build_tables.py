import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import add_buckets, fourth_distance_bucket, late_score_bucket, late_time_bucket
from table_tools import build_levels_table, lookup_row, update_specs, with_key
from paths import pbp_files, table_dir

RAW_DIR = Path("data/raw")
TABLE_DIR = table_dir()

SHORT_MAX_AIR_YARDS = 9
MEDIUM_MAX_AIR_YARDS = 19

SITUATION = ["down", "dist_b", "zone_b", "score_b", "time_b"]

LEVELS_FULL = [("L1", SITUATION), ("L2", ["down", "dist_b", "zone_b"]),
               ("L3", ["down", "dist_b"]), ("L4", [])]


LEVELS_LATE = [("L1", ["down", "dist_b", "late_score", "late_time"]), ("L2", ["down", "late_score", "late_time"]),
               ("L3", ["late_score", "late_time"])]


LEVELS_MID = [("L1", ["down", "dist_b", "zone_b"]), ("L2", ["down", "dist_b"]),
              ("L3", ["down"]), ("L4", [])]


LEVELS_RESULT = [("L1", ["down", "dist_b", "zone_b"]), ("L2", ["down", "zone_b"]),
                 ("L3", ["zone_b"]), ("L4", [])]


LEVELS_FOURTH_CALL = [("L1", ["fdist", "zone_b", "score_b"]), ("L2", ["fdist", "zone_b"]), ("L3", ["fdist"]), ("L4", [])]
FOURTH_CALL_MIN_PLAYS = 20
FOURTH_CALL_SHRINK = 15

LEVELS_FOURTH_DEPTH = [("L1", ["fdist", "zone_b"]), ("L2", ["fdist"]), ("L3", [])]
LEVELS_FOURTH_RESULT = [("L1", ["depth", "fdist", "zone_b"]), ("L2", ["depth", "fdist"]), ("L3", ["depth"]), ("L4", [])]
FOURTH_THROW_MIN_PLAYS = 20
FOURTH_THROW_SHRINK = 15

PLAYS_FROM_SEASON = 2018
LOAD_COLS = ["season", "qtr", "down", "ydstogo", "yardline_100", "score_differential",
             "game_seconds_remaining", "play_type", "qb_dropback", "qb_scramble",
             "rush_attempt", "sack", "interception", "complete_pass", "air_yards",
             "penalty", "fumble_lost", "yards_gained", "out_of_bounds",
             "return_yards", "return_touchdown", "pass_touchdown", "rush_touchdown"]


NEEDED = ["down", "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"]


def load_plays() -> pd.DataFrame:
    files = pbp_files(RAW_DIR)
    df = pd.concat([pd.read_parquet(f, columns=LOAD_COLS) for f in files], ignore_index=True)
    df = df[df["play_type"].isin(["pass", "run"]) & (df["penalty"].fillna(0) != 1) & (df["season"] >= PLAYS_FROM_SEASON)]
    df = df.dropna(subset=NEEDED).copy()
    df["down"] = df["down"].astype(int)
    df = add_buckets(df)
    df["late_score"] = late_score_bucket(df["score_differential"])
    df["late_time"] = late_time_bucket(df["qtr"], df["game_seconds_remaining"])

    df["is_dropback"] = df["qb_dropback"] == 1
    designed_run = (df["rush_attempt"] == 1) & (df["qb_scramble"] != 1) & ~df["is_dropback"]
    df = df[df["is_dropback"] | designed_run].copy()

    df["p_dropback"] = df["is_dropback"].astype(int)
    df["p_sack"] = (df["sack"] == 1).astype(int)
    df["p_scramble"] = ((df["qb_scramble"] == 1) & (df["sack"] != 1)).astype(int)
    df["is_throw"] = df["is_dropback"] & (df["p_sack"] == 0) & (df["p_scramble"] == 0)
    return df


def depth_bucket(air_yards: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([air_yards <= SHORT_MAX_AIR_YARDS, air_yards <= MEDIUM_MAX_AIR_YARDS],
                  ["short", "medium"], default="deep"),
        index=air_yards.index)


def build_throw_frame(plays: pd.DataFrame) -> pd.DataFrame:
    t = plays[plays["is_throw"]].dropna(subset=["air_yards"]).copy()
    t["depth"] = depth_bucket(t["air_yards"])
    for d in ["short", "medium", "deep"]:
        t[f"p_{d}"] = (t["depth"] == d).astype(int)
    t["p_complete"] = (t["complete_pass"] == 1).astype(int)
    t["p_int"] = (t["interception"] == 1).astype(int)
    return t


def build_fumble_table(plays: pd.DataFrame, throws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kinds = {
        "sack": plays[plays["p_sack"] == 1],
        "scramble": plays[plays["p_scramble"] == 1],
        "completion": throws[throws["p_complete"] == 1],
        "run": plays[~plays["is_dropback"]],
    }
    for kind, g in kinds.items():
        rows.append({"kind": kind, "n": len(g), "p_fumble_lost": (g["fumble_lost"] == 1).mean()})
    return pd.DataFrame(rows)


def build_field_goal_table() -> pd.DataFrame:
    import statsmodels.api as sm

    files = pbp_files(RAW_DIR)
    cols = ["play_type", "field_goal_result", "kick_distance", "yardline_100",
            "half_seconds_remaining"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    fg = df[df["play_type"] == "field_goal"].dropna(
        subset=["kick_distance", "field_goal_result", "half_seconds_remaining"]).copy()
    fg["made"] = (fg["field_goal_result"] == "made").astype(int)
    fg["dist"] = fg["kick_distance"].round().astype(int)
    fg["rushed"] = (fg["half_seconds_remaining"] <= 30).astype(float)

    offset = fg["kick_distance"] - fg["yardline_100"]
    print(f"\nKick distance minus yard line: exactly 18 for {(offset == 18).mean():.1%} "
          f"of kicks, 19 for {(offset == 19).mean():.1%}, 17 for {(offset == 17).mean():.1%}")

    X = sm.add_constant(fg[["dist", "rushed"]].astype(float))
    model = sm.Logit(fg["made"], X).fit(disp=0)
    a, b, r = model.params["const"], model.params["dist"], model.params["rushed"]
    print(f"Field goal fit on {len(fg):,} real kicks: "
          f"logit(make) = {a:.4f} + {b:.5f} * distance + {r:.4f} * rushed "
          f"(rushed p-value {model.pvalues['rushed']:.4f})")

    real_n = fg[fg["rushed"] == 0].groupby("dist")["made"].agg(n_real="size", real_rate="mean")
    real_r = fg[fg["rushed"] == 1].groupby("dist")["made"].agg(n_rushed="size", rushed_rate="mean")
    distances = np.arange(18, 76)
    out = pd.DataFrame({"kick_distance": distances})
    out["make_prob"] = 1 / (1 + np.exp(-(a + b * out["kick_distance"])))
    out["make_prob_rushed"] = 1 / (1 + np.exp(-(a + r + b * out["kick_distance"])))
    out = out.merge(real_n, left_on="kick_distance", right_index=True, how="left")
    out = out.merge(real_r, left_on="kick_distance", right_index=True, how="left")
    out[["n_real", "n_rushed"]] = out[["n_real", "n_rushed"]].fillna(0).astype(int)
    return out


def save(table: pd.DataFrame, name: str, levels, outcomes):
    table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    update_specs(TABLE_DIR, name, f"{name}.csv", levels, outcomes)
    print(f"  {name}.csv: " + ", ".join(f"{lvl} {int((table['level'] == lvl).sum())}"
                                        for lvl, _ in levels) + " situations kept")


def show(table, levels, label, **situation):
    lvl, row = lookup_row(table, levels, situation)
    cols = [c for c in row.index if c.startswith("p_")]
    probs = "  ".join(f"{c[2:]} {row[c]:.1%}" for c in cols)
    print(f"  {label}: {lvl}, {int(row['n']):,} plays | {probs}")


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "table_specs.json").unlink(missing_ok=True)
    plays = load_plays()
    throws = build_throw_frame(plays)
    print(f"Plays used: {len(plays):,} ({int(plays['is_dropback'].sum()):,} dropbacks, "
          f"{int((~plays['is_dropback']).sum()):,} designed runs), {len(throws):,} throws")

    fg_table = build_field_goal_table()
    fg_table.to_csv(TABLE_DIR / "field_goal.csv", index=False)
    print("\nField goal make probability, fitted curve (real rate, number of real kicks):")
    for d in [25, 35, 45, 50, 55, 60]:
        r = fg_table[fg_table["kick_distance"] == d].iloc[0]

        def fmt(fit, real, n):
            return f"{fit:.1%} ({real:.1%}, {int(n)} kicks)" if n else f"{fit:.1%} (no real kicks)"
        print(f"  {d} yards  normal: {fmt(r['make_prob'], r['real_rate'], r['n_real'])}"
              f"   rushed: {fmt(r['make_prob_rushed'], r['rushed_rate'], r['n_rushed'])}")

    print("\nTables written:")
    play_call = build_levels_table(plays, LEVELS_FULL, ["p_dropback"])
    save(play_call, "play_call", LEVELS_FULL, ["p_dropback"])

    fourth = plays[plays["down"] == 4].copy()
    fourth["fdist"] = fourth_distance_bucket(fourth["ydstogo"])
    play_call_fourth = build_levels_table(fourth, LEVELS_FOURTH_CALL, ["p_dropback"], min_n=FOURTH_CALL_MIN_PLAYS,
                                          shrink=FOURTH_CALL_SHRINK)
    save(play_call_fourth, "play_call_fourth", LEVELS_FOURTH_CALL, ["p_dropback"])

    late_plays = plays[plays["late_time"] != "none"]
    play_call_late = build_levels_table(late_plays, LEVELS_LATE, ["p_dropback"])
    save(play_call_late, "play_call_late", LEVELS_LATE, ["p_dropback"])

    split = build_levels_table(plays[plays["is_dropback"]], LEVELS_MID, ["p_sack", "p_scramble"])
    save(split, "dropback_split", LEVELS_MID, ["p_sack", "p_scramble"])

    depth = build_levels_table(throws, LEVELS_FULL, ["p_short", "p_medium", "p_deep"])
    save(depth, "pass_depth", LEVELS_FULL, ["p_short", "p_medium", "p_deep"])

    result_levels = with_key(LEVELS_RESULT, "depth")
    result = build_levels_table(throws, result_levels, ["p_complete", "p_int"])
    save(result, "pass_result", result_levels, ["p_complete", "p_int"])

    throws4 = throws[throws["down"] == 4].copy()
    throws4["fdist"] = fourth_distance_bucket(throws4["ydstogo"])
    depth4 = build_levels_table(throws4, LEVELS_FOURTH_DEPTH, ["p_short", "p_medium", "p_deep"],
                                min_n=FOURTH_THROW_MIN_PLAYS, shrink=FOURTH_THROW_SHRINK)
    save(depth4, "pass_depth_fourth", LEVELS_FOURTH_DEPTH, ["p_short", "p_medium", "p_deep"])
    result4 = build_levels_table(throws4, LEVELS_FOURTH_RESULT, ["p_complete", "p_int"],
                                 min_n=FOURTH_THROW_MIN_PLAYS, shrink=FOURTH_THROW_SHRINK)
    save(result4, "pass_result_fourth", LEVELS_FOURTH_RESULT, ["p_complete", "p_int"])

    throws3 = throws[throws["down"] == 3].copy()
    throws3["fdist"] = fourth_distance_bucket(throws3["ydstogo"])
    depth3 = build_levels_table(throws3, LEVELS_FOURTH_DEPTH, ["p_short", "p_medium", "p_deep"],
                                min_n=FOURTH_THROW_MIN_PLAYS, shrink=FOURTH_THROW_SHRINK)
    save(depth3, "pass_depth_third", LEVELS_FOURTH_DEPTH, ["p_short", "p_medium", "p_deep"])
    result3 = build_levels_table(throws3, LEVELS_FOURTH_RESULT, ["p_complete", "p_int"],
                                 min_n=FOURTH_THROW_MIN_PLAYS, shrink=FOURTH_THROW_SHRINK)
    save(result3, "pass_result_third", LEVELS_FOURTH_RESULT, ["p_complete", "p_int"])

    fumbles = build_fumble_table(plays, throws)
    fumbles.to_csv(TABLE_DIR / "fumble_lost.csv", index=False)
    print("  fumble_lost.csv: " + ", ".join(f"{r.kind} {r.p_fumble_lost:.2%}" for r in fumbles.itertuples()))

    print("\nSanity checks (level used, plays behind it, probabilities):")
    base = dict(down=1, dist_b="long", zone_b="own_half", score_b="tied", time_b="normal")
    show(play_call, LEVELS_FULL, "1st and 10, own half, tied: dropback share", **base)
    third = dict(down=3, dist_b="long", zone_b="own_half", score_b="tied", time_b="normal")
    show(play_call, LEVELS_FULL, "3rd and 7 to 10, own half, tied: dropback share", **third)
    show(split, LEVELS_MID, "1st and 10 dropback", **base)
    show(split, LEVELS_MID, "3rd and 7 to 10 dropback (sacks go up)", **third)
    show(depth, LEVELS_FULL, "1st and 10 throw depth", **base)
    late = dict(down=2, dist_b="long", zone_b="own_half", score_b="trailing_big", time_b="late")
    show(depth, LEVELS_FULL, "2nd and long, trailing big, late: throw depth", **late)
    for d in ["short", "medium", "deep"]:
        show(result, result_levels, f"1st and 10, {d} throw", depth=d, **base)
    for zone in ["own_half", "red_zone", "goal_line"]:
        sit = dict(base, zone_b=zone)
        show(result, result_levels, f"1st and 10, short throw, {zone}", depth="short", **sit)
