import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from buckets import (fourth_distance_bucket, fourth_zone_bucket,
                     late_distance_bucket, late_score_bucket, late_time_bucket, late_zone_bucket, late_zone2_bucket,
                     fourth_late_distance_bucket, late_quarter_bucket,
                     score_bucket, time_bucket)


from table_tools import build_levels_table, lookup_row, update_specs
from paths import pbp_files, table_dir

RAW_DIR = Path("data/raw")
TABLE_DIR = table_dir()
SEED = 42

KICKOFF_FROM_SEASON = 2025
ONSIDE_FROM_SEASON = 2018
FREE_KICK_FROM_SEASON = 2010
ONSIDE_MAX_DISTANCE = 25
TRY_FROM_SEASON = 2015
FOURTH_FROM_SEASON = 2018

PUNT_TARGET = 500
PUNT_MAX_WINDOW = 12
PUNT_MIN_CELL = 100
PUNT_MAX_SAMPLES = 1000

SCRIMMAGE = ["pass", "run", "qb_kneel", "qb_spike"]

TRY_LEVELS = [("L1", ["sd_key", "phase"]), ("L2", ["sd_key"]), ("L3", ["score_b", "phase"]), ("L4", [])]
FOURTH_LEVELS = [("L1", ["fdist", "fzone", "score_b", "time_b"]), ("L2", ["fdist", "fzone", "score_b"]),
                 ("L3", ["fdist", "fzone"]), ("L4", ["fzone"]), ("L5", [])]


FOURTH_LATE_LEVELS = [("L1", ["late_zone", "late_fdist", "late_score", "late_time", "late_q"]),
                      ("L2", ["late_zone", "late_fdist", "late_score", "late_q"]),
                      ("L3", ["late_zone", "late_fdist", "late_score"]),
                      ("L4", ["late_zone", "late_fdist"]),
                      ("L5", ["late_fdist"])]


FOURTH_LATE_MIN_PLAYS = 5
FOURTH_LATE_SHRINK = 8
FOURTH_MIN_PLAYS = 30
FOURTH_SHRINK = 25

ONSIDE_CHOICE_LEVELS = [("L1", ["late_score", "kwin"]), ("L2", ["late_score"])]
ONSIDE_CHOICE_MIN_PLAYS = 25

COLS = ["game_id", "play_id", "season", "qtr", "posteam", "defteam", "down", "ydstogo", "yardline_100",
        "play_type", "penalty", "kick_distance", "touchback", "punt_blocked", "punt_out_of_bounds",
        "punt_fair_catch", "punt_downed", "return_yards", "return_touchdown", "fumble_lost",
        "kickoff_out_of_bounds", "kickoff_fair_catch", "extra_point_result", "two_point_conv_result",
        "two_point_attempt", "score_differential", "game_seconds_remaining", "qb_kneel", "qb_spike", "safety"]


def load_raw() -> pd.DataFrame:
    files = pbp_files(RAW_DIR)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g = df.groupby("game_id")
    for c in ["posteam", "yardline_100", "play_type"]:
        df["next_" + c] = g[c].shift(-1)
    df["prev_safety"] = g["safety"].shift(1).fillna(0) == 1
    return df


def flag(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(0) == 1


def pooled_indices(yl: np.ndarray, cum: np.ndarray, yard: int, rng) -> np.ndarray:
    for w in range(PUNT_MAX_WINDOW + 1):
        lo, hi = max(1, yard - w), min(99, yard + w)
        if cum[hi] - cum[lo - 1] >= PUNT_TARGET:
            break
    idx = np.flatnonzero((yl >= lo) & (yl <= hi))
    if len(idx) < PUNT_MIN_CELL:
        idx = np.arange(len(yl))
    if len(idx) > PUNT_MAX_SAMPLES:
        idx = rng.choice(idx, PUNT_MAX_SAMPLES, replace=False)
    return idx


def prepare_punts(df: pd.DataFrame) -> pd.DataFrame:
    p = df[(df["play_type"] == "punt") & ~flag(df, "penalty")].copy()
    p["blocked"] = flag(p, "punt_blocked")
    p = p[p["blocked"] | p["kick_distance"].notna()]
    p = p.dropna(subset=["yardline_100"])
    p["yardline_100"] = p["yardline_100"].clip(1, 99).astype(int)
    p["kick_distance"] = p["kick_distance"].fillna(0).clip(lower=0).astype(int)
    p["return_yards"] = p["return_yards"].fillna(0).astype(int)
    p["result"] = np.select(
        [p["blocked"], flag(p, "touchback"), flag(p, "punt_out_of_bounds"), flag(p, "punt_fair_catch"),
         flag(p, "punt_downed"), flag(p, "return_touchdown")],
        ["blocked", "touchback", "oob", "fair_catch", "downed", "return_td"], default="returned")
    p["return_td"] = (p["result"] == "return_td").astype(int)
    p["fumble_lost"] = flag(p, "fumble_lost").astype(int)
    seen = p["blocked"] & p["next_play_type"].isin(SCRIMMAGE) & p["next_yardline_100"].notna()
    receiver_has_it = p["next_posteam"] != p["posteam"]
    spot = np.where(receiver_has_it, 100 - p["next_yardline_100"], p["next_yardline_100"])
    p["blocked_poss"] = np.where(seen, receiver_has_it.astype(int), 1)
    p["blocked_delta"] = np.where(seen, spot - p["yardline_100"], 0).astype(int)
    return p


def build_punt_bank(p: pd.DataFrame, rng) -> pd.DataFrame:
    yl = p["yardline_100"].to_numpy()
    cum = np.cumsum(np.bincount(yl, minlength=101))
    keep = ["kick_distance", "result", "return_yards", "return_td", "fumble_lost", "blocked_poss", "blocked_delta"]
    pieces = []
    for yard in range(1, 100):
        chosen = p.iloc[pooled_indices(yl, cum, yard, rng)][keep].copy()
        chosen.insert(0, "yardline_100", yard)
        pieces.append(chosen)
    return pd.concat(pieces, ignore_index=True)


def check_punt_geometry(p: pd.DataFrame):
    ok = p["next_play_type"].isin(SCRIMMAGE) & p["next_yardline_100"].notna() & (p["next_posteam"] != p["posteam"])
    land = p["yardline_100"] - p["kick_distance"]
    fc = ok & (p["result"] == "fair_catch")
    match_fc = ((100 - land)[fc] == p["next_yardline_100"][fc]).mean()
    rt = ok & (p["result"] == "returned") & (p["fumble_lost"] == 0)
    match_rt = ((100 - land - p["return_yards"])[rt] == p["next_yardline_100"][rt]).mean()
    print(f"\nPunt geometry check against the real next play: fair catches {match_fc:.1%} exact "
          f"({int(fc.sum()):,} punts), returns {match_rt:.1%} exact ({int(rt.sum()):,} punts)")
    assert match_fc > 0.98, "the punt landing rule does not match the real data"
    assert match_rt > 0.85, "the punt return rule does not match the real data"


def build_kickoff_banks(df: pd.DataFrame):
    k = df[(df["play_type"] == "kickoff") & ~flag(df, "penalty") & (df["season"] >= ONSIDE_FROM_SEASON)].copy()
    k["is_onside"] = k["kick_distance"].notna() & (k["kick_distance"] <= ONSIDE_MAX_DISTANCE)
    seen = k["next_play_type"].isin(SCRIMMAGE) & k["next_yardline_100"].notna()
    k["receiver_has_it"] = k["next_posteam"] == k["posteam"]

    onside = k[k["is_onside"] & seen].copy()
    bank_onside = pd.DataFrame({"recovered": (~onside["receiver_has_it"]).astype(int),
                                "start": onside["next_yardline_100"].astype(int)})

    first_season = min(KICKOFF_FROM_SEASON, int(df["season"].max()))
    n = k[(~k["is_onside"]) & (k["season"] >= first_season)].copy()
    td = flag(n, "return_touchdown")
    seen_n = n["next_play_type"].isin(SCRIMMAGE) & n["next_yardline_100"].notna() & n["receiver_has_it"]
    n = n[td | seen_n].copy()
    td = flag(n, "return_touchdown")
    n["kind"] = np.select([td, flag(n, "touchback"), flag(n, "kickoff_out_of_bounds"), flag(n, "kickoff_fair_catch")],
                          ["return_td", "touchback", "oob", "fair_catch"], default="return")
    bank_kick = pd.DataFrame({"kind": n["kind"].to_numpy(),
                              "start": np.where(td, 0, n["next_yardline_100"]).astype(int),
                              "td": td.astype(int).to_numpy()})
    return bank_kick, bank_onside


def build_free_kick_bank(df: pd.DataFrame) -> pd.DataFrame:
    k = df[(df["play_type"] == "kickoff") & df["prev_safety"] & ~flag(df, "penalty")
           & (df["season"] >= FREE_KICK_FROM_SEASON)].copy()
    td = flag(k, "return_touchdown")
    seen = k["next_play_type"].isin(SCRIMMAGE) & k["next_yardline_100"].notna() & (k["next_posteam"] == k["posteam"])
    k = k[td | seen].copy()
    td = flag(k, "return_touchdown")
    k["kind"] = np.select([td, flag(k, "touchback"), flag(k, "kickoff_out_of_bounds"), flag(k, "kickoff_fair_catch")],
                          ["return_td", "touchback", "oob", "fair_catch"], default="return")
    return pd.DataFrame({"kind": k["kind"].to_numpy(),
                         "start": np.where(td, 0, k["next_yardline_100"]).astype(int),
                         "td": td.astype(int).to_numpy()})


def build_onside_choice(df: pd.DataFrame) -> pd.DataFrame:
    k = df[(df["play_type"] == "kickoff") & ~flag(df, "penalty") & (df["season"] >= ONSIDE_FROM_SEASON)
           & (df["qtr"] == 4) & df["score_differential"].notna() & df["kick_distance"].notna()
           & df["game_seconds_remaining"].notna()].copy()
    k["kicker_lead"] = -k["score_differential"]
    k = k[k["kicker_lead"] < 0].copy()
    k["late_score"] = late_score_bucket(k["kicker_lead"])
    k["kwin"] = np.select([k["game_seconds_remaining"] <= 120, k["game_seconds_remaining"] <= 300,
                           k["game_seconds_remaining"] <= 600], ["k120", "k300", "k600"], default="k900")
    k["p_onside"] = (k["kick_distance"] <= ONSIDE_MAX_DISTANCE).astype(float)
    return build_levels_table(k, ONSIDE_CHOICE_LEVELS, ["p_onside"], min_n=ONSIDE_CHOICE_MIN_PLAYS)


def build_try_tables(df: pd.DataFrame):
    x = df[df["extra_point_result"].notna() & (df["season"] >= TRY_FROM_SEASON)].copy()
    xp = pd.DataFrame({"n": [len(x)],
                       "p_good": [(x["extra_point_result"] == "good").mean()],
                       "p_failed": [(x["extra_point_result"] == "failed").mean()],
                       "p_blocked": [(x["extra_point_result"] == "blocked").mean()]})
    t = df[flag(df, "two_point_attempt") & df["two_point_conv_result"].notna() & (df["season"] >= TRY_FROM_SEASON)].copy()
    two = pd.DataFrame({"n": [len(t)], "p_success": [(t["two_point_conv_result"] == "success").mean()]})

    tries = pd.concat([x.assign(p_two=0), t.assign(p_two=1)], ignore_index=True)
    tries = tries.dropna(subset=["score_differential", "qtr"])
    tries["sd_key"] = tries["score_differential"].clip(-20, 20).astype(int)
    tries["phase"] = np.where(tries["qtr"] >= 4, "late", "early")
    tries["score_b"] = score_bucket(tries["score_differential"])
    choice = build_levels_table(tries, TRY_LEVELS, ["p_two"])
    return xp, two, choice


def build_fourth_down(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(df["down"] == 4) & ~flag(df, "penalty") & df["play_type"].isin(["pass", "run", "punt", "field_goal"])
           & ~flag(df, "qb_kneel") & ~flag(df, "qb_spike") & (df["season"] >= FOURTH_FROM_SEASON)].copy()
    f = f.dropna(subset=["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"])
    f["fdist"] = fourth_distance_bucket(f["ydstogo"])
    f["fzone"] = fourth_zone_bucket(f["yardline_100"])
    f["score_b"] = score_bucket(f["score_differential"])
    f["time_b"] = time_bucket(f["game_seconds_remaining"])
    f["p_go"] = f["play_type"].isin(["pass", "run"]).astype(float)
    f["p_punt"] = (f["play_type"] == "punt").astype(float)
    f["p_fg"] = (f["play_type"] == "field_goal").astype(float)
    return build_levels_table(f, FOURTH_LEVELS, ["p_go", "p_punt", "p_fg"], min_n=FOURTH_MIN_PLAYS, shrink=FOURTH_SHRINK)


def build_fourth_down_late(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(df["down"] == 4) & ~flag(df, "penalty") & df["play_type"].isin(["pass", "run", "punt", "field_goal"])
           & ~flag(df, "qb_kneel") & ~flag(df, "qb_spike") & (df["season"] >= FOURTH_FROM_SEASON)].copy()
    f = f.dropna(subset=["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"])
    f["late_time"] = late_time_bucket(f["qtr"], f["game_seconds_remaining"])
    f = f[f["late_time"] != "none"].copy()
    f["late_zone"] = late_zone_bucket(f["yardline_100"])
    f["late_zone2"] = late_zone2_bucket(f["yardline_100"])
    f["late_dist"] = late_distance_bucket(f["ydstogo"])
    f["late_fdist"] = fourth_late_distance_bucket(f["ydstogo"])
    f["late_q"] = late_quarter_bucket(f["late_time"])
    f["late_score"] = late_score_bucket(f["score_differential"])
    f["p_go"] = f["play_type"].isin(["pass", "run"]).astype(float)
    f["p_punt"] = (f["play_type"] == "punt").astype(float)
    f["p_fg"] = (f["play_type"] == "field_goal").astype(float)
    return build_levels_table(f, FOURTH_LATE_LEVELS, ["p_go", "p_punt", "p_fg"], min_n=FOURTH_LATE_MIN_PLAYS,
                              shrink=FOURTH_LATE_SHRINK)


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raw = load_raw()

    punts = prepare_punts(raw)
    bank_punt = build_punt_bank(punts, rng)
    bank_punt.to_csv(TABLE_DIR / "bank_punt.csv", index=False)
    print(f"bank_punt.csv: {len(bank_punt):,} rows from {len(punts):,} real punts")
    check_punt_geometry(punts)
    print("\nPunt results by yard line (yardline_100 of the punting team):")
    print("  yard line   avg distance   touchback   returned   fair catch   blocked")
    for yard in [40, 55, 65, 75, 85, 95]:
        b = bank_punt[bank_punt["yardline_100"] == yard]
        land = yard - b["kick_distance"]
        tb = ((land <= 0) | (b["result"] == "touchback")).mean()
        print(f"  {yard:9d}   {b[b['result'] != 'blocked']['kick_distance'].mean():12.1f}   {tb:9.1%}   "
              f"{(b['result'] == 'returned').mean():8.1%}   {(b['result'] == 'fair_catch').mean():10.1%}   "
              f"{(b['result'] == 'blocked').mean():7.1%}")

    bank_kick, bank_onside = build_kickoff_banks(raw)
    bank_kick.to_csv(TABLE_DIR / "bank_kickoff.csv", index=False)
    bank_onside.to_csv(TABLE_DIR / "bank_onside.csv", index=False)
    print(f"\nbank_kickoff.csv: {len(bank_kick):,} real {KICKOFF_FROM_SEASON} kickoffs")
    print(bank_kick.groupby("kind")["start"].agg(["count", "mean"]).assign(
        share=lambda d: d["count"] / d["count"].sum()).round(3).to_string())
    print(f"  average starting yardline_100 for the receiving team: {bank_kick[bank_kick['td'] == 0]['start'].mean():.1f}")
    print(f"bank_onside.csv: {len(bank_onside)} real onside kicks since {ONSIDE_FROM_SEASON}, "
          f"recovered by the kicking team {bank_onside['recovered'].mean():.1%}")

    bank_free = build_free_kick_bank(raw)
    bank_free.to_csv(TABLE_DIR / "bank_free_kick.csv", index=False)
    print(f"\nbank_free_kick.csv: {len(bank_free)} real free kicks after a safety since {FREE_KICK_FROM_SEASON}; "
          f"the receiving team starts at yardline_100 {bank_free[bank_free['td'] == 0]['start'].mean():.1f} on average "
          f"(a normal kickoff: {bank_kick[bank_kick['td'] == 0]['start'].mean():.1f})")
    print(bank_free.groupby("kind")["start"].agg(["count", "mean"]).round(1).to_string())

    xp, two, choice = build_try_tables(raw)
    xp.to_csv(TABLE_DIR / "extra_point.csv", index=False)
    two.to_csv(TABLE_DIR / "two_point.csv", index=False)
    choice.to_csv(TABLE_DIR / "try_choice.csv", index=False)
    update_specs(TABLE_DIR, "try_choice", "try_choice.csv", TRY_LEVELS, ["p_two"])
    print(f"\nExtra point: good {xp['p_good'][0]:.1%}, failed {xp['p_failed'][0]:.1%}, blocked {xp['p_blocked'][0]:.1%} "
          f"({int(xp['n'][0]):,} kicks)")
    print(f"Two point try: success {two['p_success'][0]:.1%} ({int(two['n'][0]):,} tries)")
    print("Chance a team goes for two (score is after the touchdown, so -5 means trailing by 5):")
    for sd, phase in [(6, "early"), (0, "late"), (-1, "late"), (-2, "late"), (-5, "late"), (-8, "late"),
                      (-5, "early"), (5, "late"), (12, "late"), (14, "early")]:
        sit = {"sd_key": sd, "phase": phase, "score_b": score_bucket(pd.Series([sd]))[0]}
        level, row = lookup_row(choice, TRY_LEVELS, sit)
        print(f"  score {sd:+3d}, {phase:5s}: {row['p_two']:.1%}  ({level}, {int(row['n']):,} tries)")

    onside_choice = build_onside_choice(raw)
    onside_choice.to_csv(TABLE_DIR / "onside_choice.csv", index=False)
    update_specs(TABLE_DIR, "onside_choice", "onside_choice.csv", ONSIDE_CHOICE_LEVELS, ["p_onside"])
    print("\nOnside kick choice (kicking team just scored but is still behind, 4th quarter):")
    print("  behind by      seconds left   tries onside   real kickoffs")
    for lv in ["L1"]:
        for r in onside_choice[onside_choice["level"] == lv].sort_values(["late_score", "kwin"]).itertuples():
            print(f"  {r.late_score:<11s}    {r.kwin:<12s}   {r.p_onside:11.1%}   {int(r.n):13d}")

    fourth = build_fourth_down(raw)
    fourth.to_csv(TABLE_DIR / "fourth_down.csv", index=False)
    update_specs(TABLE_DIR, "fourth_down", "fourth_down.csv", FOURTH_LEVELS, ["p_go", "p_punt", "p_fg"])
    fourth_late = build_fourth_down_late(raw)
    fourth_late.to_csv(TABLE_DIR / "fourth_down_late.csv", index=False)
    update_specs(TABLE_DIR, "fourth_down_late", "fourth_down_late.csv", FOURTH_LATE_LEVELS, ["p_go", "p_punt", "p_fg"])
    print("Fourth down late game: " + ", ".join(f"{lv} {int((fourth_late['level'] == lv).sum())}"
                                              for lv, _ in FOURTH_LATE_LEVELS) + " situations kept")
    print("\nFourth down: " + ", ".join(f"{lv} {int((fourth['level'] == lv).sum())}" for lv, _ in FOURTH_LEVELS)
          + " situations kept")
    print("  situation                                   level  plays    go    punt   kick")
    for label, sit in [
        ("4th and 1, own 30, tied, early", dict(fdist="1", fzone="z66_80", score_b="tied", time_b="normal")),
        ("4th and 10, own 30, tied, early", dict(fdist="8_10", fzone="z66_80", score_b="tied", time_b="normal")),
        ("4th and 1, opp 40, tied, early", dict(fdist="1", fzone="z37_40", score_b="tied", time_b="normal")),
        ("4th and 8, opp 40, tied, early", dict(fdist="8_10", fzone="z37_40", score_b="tied", time_b="normal")),
        ("4th and 3, opp 25, tied, early", dict(fdist="3_4", fzone="z21_30", score_b="tied", time_b="normal")),
        ("4th and goal at the 2, tied, early", dict(fdist="2", fzone="z1_5", score_b="tied", time_b="normal")),
        ("4th and 5, opp 45, trailing big, late", dict(fdist="5_7", fzone="z41_50", score_b="trailing_big", time_b="late")),
        ("4th and 5, opp 45, leading big, late", dict(fdist="5_7", fzone="z41_50", score_b="leading_big", time_b="late")),
    ]:
        level, r = lookup_row(fourth, FOURTH_LEVELS, sit)
        print(f"  {label:42s} {level}  {int(r['n']):6,}  {r['p_go']:5.0%}  {r['p_punt']:5.0%}  {r['p_fg']:5.0%}")
