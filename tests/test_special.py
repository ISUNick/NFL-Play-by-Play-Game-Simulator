import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.state import score_bucket
from test_tables import index_table, lookup

TABLES = ROOT / "data" / "tables"
PUNT_RESULTS = {"returned", "fair_catch", "oob", "downed", "touchback", "blocked", "return_td"}
KICK_KINDS = {"return", "return_td", "touchback", "oob", "fair_catch"}
FDISTS = ["1", "2", "3_4", "5_7", "8_10", "11_plus"]
FZONES = ["z1_5", "z6_20", "z21_30", "z31_36", "z37_40", "z41_50", "z51_65", "z66_80", "z81_99"]
SCORES = ["trailing_big", "trailing_close", "tied", "leading_close", "leading_big"]
TIMES = ["normal", "late"]


def read(name: str) -> pd.DataFrame:
    df = pd.read_csv(TABLES / f"{name}.csv")
    assert not df.isna().any().any(), f"{name} has missing values"
    return df


def specs() -> dict:
    return json.loads((TABLES / "table_specs.json").read_text())


def test_punt_bank_shape_and_coverage():
    b = read("bank_punt")
    assert set(b.columns) == {"yardline_100", "kick_distance", "result", "return_yards", "return_td",
                              "fumble_lost", "blocked_poss", "blocked_delta"}
    assert set(b["result"]) <= PUNT_RESULTS, set(b["result"]) - PUNT_RESULTS
    g = b.groupby("yardline_100").size()
    assert list(g.index) == list(range(1, 100)) and (g >= 100).all(), "every yard line needs 100 or more punts"
    assert b["kick_distance"].between(0, 90).all()
    assert b["blocked_poss"].isin([0, 1]).all() and b["return_td"].isin([0, 1]).all()
    blocked = b[b["result"] == "blocked"]
    assert (b["result"] == "blocked").mean() < 0.02
    assert blocked["blocked_poss"].mean() > 0.5, "most blocked punts should end with the receiving team"
    assert (b[b["result"] == "fair_catch"]["return_yards"] == 0).all(), "a fair catch has no return"


def test_punts_make_football_sense():
    b = read("bank_punt")
    live = b[b["result"] != "blocked"]
    short_field = live[live["yardline_100"] == 40]["kick_distance"].mean()
    deep = live[live["yardline_100"] == 80]["kick_distance"].mean()
    assert short_field < deep - 6, "punts from the opponent's half should be shorter"
    assert 44 < deep < 53

    def touchback_share(yard):
        s = b[b["yardline_100"] == yard]
        return ((yard - s["kick_distance"] <= 0) | (s["result"] == "touchback")).mean()
    assert 0.1 < touchback_share(40) < 0.4, "a punt from the opponent's 40 should often be a touchback"
    assert touchback_share(90) < 0.01, "a punt from your own 10 cannot be a touchback"

    s = b[(b["yardline_100"] == 80) & (b["result"] != "blocked")]
    land = 80 - s["kick_distance"]
    receiver_start = (100 - land - s["return_yards"]).where(land > 0, 80)
    net = 80 + receiver_start - 100
    assert 33 < net.mean() < 46, f"net punt average from your own 20 should be near 40, got {net.mean():.1f}"
    assert 0.001 < b["return_td"].mean() < 0.012


def test_kickoff_bank():
    k = read("bank_kickoff")
    assert set(k.columns) == {"kind", "start", "td"}
    assert len(k) > 2000 and set(k["kind"]) <= KICK_KINDS
    live = k[k["td"] == 0]
    assert live["start"].between(1, 99).all(), "a kickoff that is not a touchdown must start somewhere on the field"
    assert (k[k["td"] == 1]["start"] == 0).all()
    assert 0.0005 < k["td"].mean() < 0.01
    assert 60 < live["start"].mean() < 75, "average kickoff start should be near the 30 yard line"
    tb = k[k["kind"] == "touchback"]["start"]
    assert tb.isin([65, 80]).mean() > 0.8, "touchbacks come out to the 35 (or the 20)"
    assert 0.1 < (k["kind"] == "touchback").mean() < 0.6


def test_onside_bank():
    o = read("bank_onside")
    assert set(o.columns) == {"recovered", "start"}
    assert len(o) >= 200 and o["recovered"].isin([0, 1]).all() and o["start"].between(1, 99).all()
    assert 0.03 < o["recovered"].mean() < 0.25, "onside kicks usually fail but sometimes work"


def test_extra_point_and_two_point():
    xp = read("extra_point").iloc[0]
    assert abs(xp["p_good"] + xp["p_failed"] + xp["p_blocked"] - 1) < 1e-9
    assert 0.90 < xp["p_good"] < 0.99
    two = read("two_point").iloc[0]
    assert 0.40 < two["p_success"] < 0.56


def test_try_choice():
    t = read("try_choice")
    assert t["p_two"].between(0, 1).all() and (t["n"] >= 100).all()
    sp = specs()["try_choice"]
    idx = index_table(t, sp["levels"])
    sit = lambda sd, phase: {"sd_key": max(-20, min(20, sd)), "phase": phase, "score_b": score_bucket(sd)}
    for sd, phase in product(range(-60, 61), ["early", "late"]):
        lookup(idx, sp["levels"], sit(sd, phase))
    p = lambda sd, phase: lookup(idx, sp["levels"], sit(sd, phase))[1]["p_two"]
    assert p(6, "early") < 0.03, "up 6 after a touchdown in the first half: kick the extra point"
    assert p(14, "early") < 0.10, "up 14 early: real teams rarely go for two (about 4%)"
    assert p(18, "early") < 0.05, "a rare score gap must borrow from its score bucket, not the league average"
    assert p(40, "early") < 0.05
    assert p(-2, "late") > 0.5, "trailing by 2 late: go for two to tie"
    assert p(-5, "late") > p(-5, "early"), "trailing by 5 late should go for two more than early"


def test_fourth_down_table_is_valid():
    t = read("fourth_down")
    sp = specs()["fourth_down"]
    assert ((t[["p_go", "p_punt", "p_fg"]] >= 0) & (t[["p_go", "p_punt", "p_fg"]] <= 1)).all().all()
    assert ((t["p_go"] + t["p_punt"] + t["p_fg"]) - 1).abs().max() < 1e-6, "every fourth down is a go, punt or kick"
    assert (t["n"] >= 30).all()
    assert (t["level"] == sp["levels"][-1][0]).any(), "the catch all level must exist"
    idx = index_table(t, sp["levels"])
    checked = 0
    for fdist, fzone, score, time in product(FDISTS, FZONES, SCORES, TIMES):
        lookup(idx, sp["levels"], dict(fdist=fdist, fzone=fzone, score_b=score, time_b=time))
        checked += 1
    assert checked == 540


def test_fourth_down_makes_football_sense():
    t = read("fourth_down")
    sp = specs()["fourth_down"]
    idx = index_table(t, sp["levels"])

    def row(fdist, fzone, score="tied", time="normal"):
        return lookup(idx, sp["levels"], dict(fdist=fdist, fzone=fzone, score_b=score, time_b=time))[1]

    for fzone in ["z66_80", "z81_99"]:
        for fdist in FDISTS:
            assert row(fdist, fzone)["p_fg"] < 0.02, (fzone, fdist)
    for fzone in ["z1_5", "z6_20"]:
        for fdist in FDISTS:
            assert row(fdist, fzone)["p_punt"] < 0.02, (fzone, fdist)
    assert row("11_plus", "z81_99")["p_punt"] > 0.85, "4th and long deep in your own end: punt"
    assert row("3_4", "z21_30")["p_fg"] > 0.5, "4th and 3 inside the 30: kick"
    assert row("1", "z41_50")["p_go"] > row("8_10", "z41_50")["p_go"] + 0.3
    assert row("5_7", "z41_50", "trailing_big", "late")["p_go"] > \
        row("5_7", "z41_50", "leading_big", "late")["p_go"] + 0.2


if __name__ == "__main__":
    test_punt_bank_shape_and_coverage()
    test_punts_make_football_sense()
    test_kickoff_bank()
    test_onside_bank()
    test_extra_point_and_two_point()
    test_try_choice()
    test_fourth_down_table_is_valid()
    test_fourth_down_makes_football_sense()
    print("All special teams tests passed")
