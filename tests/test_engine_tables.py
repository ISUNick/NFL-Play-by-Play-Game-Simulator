import random
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
from engine.state import GameState
from engine.tables import Tables
from table_tools import lookup_row

TABLE_DIR = ROOT / "data" / "tables"
KINDS = ["sack", "interception", "pass_td", "complete_oob", "complete_in", "incomplete",
         "run_td", "run_fumble", "scramble_oob", "scramble_in", "run_oob", "run_in",
         "punt", "field_goal", "kickoff", "extra_point"]


SCORES = ["trailing_big", "trailing_close", "tied", "leading_close", "leading_big"]
DISTS = ["short", "medium", "long", "very_long"]
ZONES = ["own_10", "backed_up", "own_half", "opp_half", "red_zone", "goal_line"]
FDISTS = ["1", "2", "3_4", "5_7", "8_10", "11_plus"]
FZONES = ["z1_5", "z6_20", "z21_30", "z31_36", "z37_40", "z41_50", "z51_65", "z66_80", "z81_99"]

_tables = None


def tables() -> Tables:
    global _tables
    if _tables is None:
        _tables = Tables()
    return _tables


def test_banks_cover_every_yard_line():
    t = tables()
    first_yard = {"short": 1, "medium": 10, "deep": 20}
    for depth, first in first_yard.items():
        for yard in range(first, 100):
            assert len(t.completion[(depth, yard)]) >= 100, (depth, yard)
    for dist, yard in product(DISTS, range(1, 100)):
        assert len(t.run[(dist, yard)]) >= 100, (dist, yard)
    for yard in range(1, 100):
        assert len(t.scramble[yard]) >= 100 and len(t.sack[yard]) >= 100 and len(t.punt[yard]) >= 100, yard
    assert set(t.interception) == {"short", "medium", "deep"}
    assert len(t.kickoff) > 2000 and len(t.onside) > 200


def test_loader_matches_the_pandas_lookup():
    t = tables()
    rng = random.Random(7)
    import json
    specs = json.loads((TABLE_DIR / "table_specs.json").read_text())
    spaces = {
        "play_call": dict(down=[1, 2, 3, 4], dist_b=DISTS, zone_b=ZONES, score_b=SCORES, time_b=["normal", "late"]),
        "dropback_split": dict(down=[1, 2, 3, 4], dist_b=DISTS, zone_b=ZONES, score_b=SCORES, time_b=["normal", "late"]),
        "pass_depth": dict(down=[1, 2, 3, 4], dist_b=DISTS, zone_b=ZONES, score_b=SCORES, time_b=["normal", "late"]),
        "pass_result": dict(depth=["short", "medium", "deep"], down=[1, 2, 3, 4], dist_b=DISTS, zone_b=ZONES,
                            score_b=SCORES, time_b=["normal", "late"]),
        "fourth_down": dict(fdist=FDISTS, fzone=FZONES, score_b=SCORES, time_b=["normal", "late"]),
        "try_choice": dict(sd_key=list(range(-20, 21)), phase=["early", "late"], score_b=SCORES),
    }
    for name, space in spaces.items():
        spec = specs[name]
        frame = pd.read_csv(TABLE_DIR / spec["file"])
        for _ in range(300):
            sit = {k: rng.choice(v) for k, v in space.items()}
            level, row = t.lookup(name, sit)
            expected_level, expected = lookup_row(frame, spec["levels"], sit)
            assert level == expected_level, (name, sit, level, expected_level)
            for outcome in spec["outcomes"]:
                assert abs(row[outcome] - expected[outcome]) < 1e-9, (name, sit, outcome)
            assert row["n"] == int(expected["n"])


def test_draw_seconds():
    t = tables()
    for kind, tempo, score in product(KINDS, ["normal", "last5", "last2"], SCORES):
        assert 0 <= t.draw_seconds(kind, tempo, score, random.Random(1)) <= 120
    a = [t.draw_seconds("run_in", "normal", "tied", r) for r in [random.Random(5)] * 1 for _ in range(1)]
    r1, r2 = random.Random(9), random.Random(9)
    assert [t.draw_seconds("run_in", "normal", "tied", r1) for _ in range(50)] == \
        [t.draw_seconds("run_in", "normal", "tied", r2) for _ in range(50)]
    assert a


def test_field_goal_and_small_tables():
    t = tables()
    probs = [t.fg_make_prob(d) for d in range(18, 76)]
    assert all(a >= b for a, b in zip(probs, probs[1:])), "longer kicks must be harder"
    assert all(t.fg_make_prob(d, True) <= t.fg_make_prob(d) for d in range(18, 76))
    assert t.fg_make_prob(5) == t.fg_make_prob(18) and t.fg_make_prob(200) == t.fg_make_prob(75)
    assert set(t.fumble_lost) == {"sack", "scramble", "completion", "run"}
    assert abs(t.extra_point["p_good"] + t.extra_point["p_failed"] + t.extra_point["p_blocked"] - 1) < 1e-9


def test_late_game_tables_are_valid():
    t = tables()
    assert "play_call_late" in t._levels and "fourth_down_late" in t._levels
    for state in (GameState(quarter=4, clock=100, down=3, distance=5, yardline_100=40, offense_score=7, defense_score=14),
                  GameState(quarter=4, clock=400, down=1, distance=10, yardline_100=75, offense_score=14, defense_score=7)):
        level, row = t.lookup_late("play_call_late", "play_call", state)
        assert level.startswith("late_") and 0 <= row["p_dropback"] <= 1 and row["n"] >= 100, (level, row)
    for (level, key), row in t._index["fourth_down_late"].items():
        assert abs(row["p_go"] + row["p_punt"] + row["p_fg"] - 1) < 1e-6, (level, key)
        assert row["n"] >= 5, (level, key, row["n"])
    level, row = t.lookup_late("play_call_late", "play_call",
                               GameState(quarter=2, clock=100, down=1, distance=10, yardline_100=60))
    assert level.startswith("late_") and row["p_dropback"] > 0.6, (level, row)
    for quarter, clock in ((2, 600), (1, 100), (3, 100), (4, 600)):
        level, _ = t.lookup_late("play_call_late", "play_call",
                                 GameState(quarter=quarter, clock=clock, down=1, distance=10, yardline_100=60))
        assert not level.startswith("late_"), (quarter, clock, level)


if __name__ == "__main__":
    test_banks_cover_every_yard_line()
    test_loader_matches_the_pandas_lookup()
    test_draw_seconds()
    test_field_goal_and_small_tables()
    test_late_game_tables_are_valid()
    print("All engine table tests passed")
