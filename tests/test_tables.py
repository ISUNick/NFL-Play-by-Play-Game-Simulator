import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_prep"))
from table_tools import MIN_PLAYS

TABLES = ROOT / "data" / "tables"
DOWNS = [1, 2, 3, 4]
DISTS = ["short", "medium", "long", "very_long"]
ZONES = ["own_10", "backed_up", "own_half", "opp_half", "red_zone", "goal_line"]
SCORES = ["trailing_big", "trailing_close", "tied", "leading_close", "leading_big"]
TIMES = ["normal", "late"]
DEPTHS = ["short", "medium", "deep"]
MIN_YARD_FOR_DEPTH = {"short": 1, "medium": 10, "deep": 20}


def load_specs():
    return json.loads((TABLES / "table_specs.json").read_text())


def index_table(table: pd.DataFrame, levels: list) -> dict:
    index = {}
    for row in table.to_dict("records"):
        keys = dict(levels)[row["level"]]
        index[(row["level"], tuple(str(row[k]) for k in keys))] = row
    return index


def lookup(index: dict, levels: list, situation: dict):
    for level, keys in levels:
        row = index.get((level, tuple(str(situation[k]) for k in keys)))
        if row is not None:
            return level, row
    raise AssertionError(f"no row found for {situation}")


def test_probability_tables_are_valid():
    specs = load_specs()
    for name in ["play_call", "dropback_split", "pass_depth", "pass_result"]:
        spec = specs[name]
        t = pd.read_csv(TABLES / spec["file"])
        outcomes = spec["outcomes"]
        assert not t[outcomes].isna().any().any(), f"{name} has missing probabilities"
        assert ((t[outcomes] >= 0) & (t[outcomes] <= 1)).all().all(), f"{name} has a probability outside 0 to 1"
        assert (t["n"] >= MIN_PLAYS).all(), f"{name} trusts a situation with too few plays"
        last_level = spec["levels"][-1][0]
        assert (t["level"] == last_level).any(), f"{name} is missing its catch all level"
    split = pd.read_csv(TABLES / "dropback_split.csv")
    assert ((split["p_sack"] + split["p_scramble"]) <= 1 + 1e-9).all()
    depth = pd.read_csv(TABLES / "pass_depth.csv")
    assert ((depth["p_short"] + depth["p_medium"] + depth["p_deep"] - 1).abs() < 1e-6).all()
    result = pd.read_csv(TABLES / "pass_result.csv")
    assert ((result["p_complete"] + result["p_int"]) <= 1 + 1e-9).all()


def test_every_situation_has_an_answer():
    specs = load_specs()
    indexes = {n: index_table(pd.read_csv(TABLES / specs[n]["file"]), specs[n]["levels"])
               for n in ["play_call", "dropback_split", "pass_depth", "pass_result"]}
    checked = 0
    for down, dist, zone, score, time in product(DOWNS, DISTS, ZONES, SCORES, TIMES):
        sit = dict(down=down, dist_b=dist, zone_b=zone, score_b=score, time_b=time)
        for name in ["play_call", "dropback_split", "pass_depth"]:
            lookup(indexes[name], specs[name]["levels"], sit)
            checked += 1
        for d in DEPTHS:
            lookup(indexes["pass_result"], specs["pass_result"]["levels"], dict(sit, depth=d))
            checked += 1
    assert checked > 5000


def test_football_makes_sense():
    specs = load_specs()
    split = index_table(pd.read_csv(TABLES / "dropback_split.csv"), specs["dropback_split"]["levels"])
    result = index_table(pd.read_csv(TABLES / "pass_result.csv"), specs["pass_result"]["levels"])
    first = dict(down=1, dist_b="long", zone_b="own_half", score_b="tied", time_b="normal")
    third = dict(first, down=3)
    assert lookup(split, specs["dropback_split"]["levels"], third)[1]["p_sack"] > \
        1.5 * lookup(split, specs["dropback_split"]["levels"], first)[1]["p_sack"], "3rd and long should have far more sacks"
    levels = specs["pass_result"]["levels"]
    comp = {d: lookup(result, levels, dict(first, depth=d))[1]["p_complete"] for d in DEPTHS}
    assert comp["short"] > comp["medium"] > comp["deep"], "deeper throws should complete less"
    ints = {d: lookup(result, levels, dict(first, depth=d))[1]["p_int"] for d in DEPTHS}
    assert ints["short"] < ints["medium"] < ints["deep"], "deeper throws should be picked more"
    by_zone = {z: lookup(result, levels, dict(first, zone_b=z, depth="short"))[1]["p_complete"]
               for z in ["own_half", "red_zone", "goal_line"]}
    assert by_zone["own_half"] > by_zone["red_zone"] > by_zone["goal_line"], by_zone


def test_fumble_and_field_goal_tables():
    f = pd.read_csv(TABLES / "fumble_lost.csv").set_index("kind")["p_fumble_lost"]
    assert set(f.index) == {"sack", "scramble", "completion", "run"}
    assert ((f >= 0) & (f < 0.15)).all()
    assert f["sack"] == f.max(), "sacks should be the most fumble prone"
    fg = pd.read_csv(TABLES / "field_goal.csv")
    assert fg["make_prob"].is_monotonic_decreasing, "longer kicks must be harder"
    assert (fg["make_prob_rushed"] <= fg["make_prob"]).all(), "rushed kicks must not be easier"
    assert fg["make_prob"].between(0, 1).all()


def load_bank(name: str) -> pd.DataFrame:
    b = pd.read_csv(TABLES / f"{name}.csv")
    assert not b.isna().any().any(), f"{name} has missing values"
    assert set(b.columns) == {"key", "yardline_100", "yards", "oob", "td"}
    return b


def test_banks_cover_every_yard_line():
    comp = load_bank("bank_completion")
    for depth, first_yard in MIN_YARD_FOR_DEPTH.items():
        g = comp[comp["key"] == depth].groupby("yardline_100").size()
        assert list(g.index) == list(range(first_yard, 100)), f"completion bank {depth} has gaps"
        assert (g >= 100).all(), f"completion bank {depth} has a thin cell"
    run = load_bank("bank_run")
    for dist in DISTS:
        g = run[run["key"] == dist].groupby("yardline_100").size()
        assert list(g.index) == list(range(1, 100)) and (g >= 100).all(), f"run bank {dist}"
    for name in ["bank_scramble", "bank_sack"]:
        g = load_bank(name).groupby("yardline_100").size()
        assert list(g.index) == list(range(1, 100)) and (g >= 100).all(), name


def test_red_zone_is_not_inflated():
    comp = load_bank("bank_completion")
    at4 = comp[comp["yardline_100"] == 4]
    assert at4["yards"].max() < 11, "a completion from the 4 gained more than 10 yards"
    td_share = at4["td"].mean()
    assert 0.65 < td_share < 0.92, f"touchdown share on completions from the 4: {td_share:.2f}"
    run = load_bank("bank_run")
    run4 = run[run["yardline_100"] == 4]
    assert 0.15 < run4["td"].mean() < 0.40, "runs from the 4 should score sometimes (real: 26.6%), not usually"
    run50 = run[run["yardline_100"] == 50]
    assert 3.8 < run50["yards"].mean() < 5.3
    assert 0.02 < run50["oob"].mean() < 0.09


def test_open_field_yardage_is_realistic():
    comp = load_bank("bank_completion")
    c50 = comp[comp["yardline_100"] == 50].groupby("key")["yards"].mean()
    assert 6 < c50["short"] < 11 and 14 < c50["medium"] < 21 and 25 < c50["deep"] < 42, c50.to_dict()
    sack = load_bank("bank_sack")
    assert -8 < sack["yards"].mean() < -5
    assert (sack["yards"] <= 0).all()
    scr = load_bank("bank_scramble")
    s50 = scr[scr["yardline_100"] == 50]
    assert 5 < s50["yards"].mean() < 11


def test_interception_bank():
    b = pd.read_csv(TABLES / "bank_int.csv")
    assert set(b.columns) == {"depth", "air_yards", "return_yards", "return_td"}
    assert not b.isna().any().any()
    assert 0.05 < b["return_td"].mean() < 0.15
    assert 8 < b["return_yards"].mean() < 20


if __name__ == "__main__":
    test_probability_tables_are_valid()
    test_every_situation_has_an_answer()
    test_football_makes_sense()
    test_fumble_and_field_goal_tables()
    test_banks_cover_every_yard_line()
    test_red_zone_is_not_inflated()
    test_open_field_yardage_is_realistic()
    test_interception_bank()
    print("All table tests passed")
