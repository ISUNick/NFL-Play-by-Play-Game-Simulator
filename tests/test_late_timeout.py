import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))

import buckets
from engine import late
from engine.game import Game
from engine.state import GameState, late_score_class, late_window2
from engine.tables import Tables

T = Tables()
WINDOWS = ["w10", "w20", "w30", "w45", "w60", "w90", "w120", "w180", "w240"]
CLASSES = ["tied", "lead", "t1_3", "t4_8", "t9_16", "t17p"]


def q4(clock, diff=0, down=1, yl=57, running=True, to=3, dto=3):
    return GameState(quarter=4, clock=clock, down=down, distance=10, yardline_100=yl,
                     offense_score=20 + max(diff, 0), defense_score=20 + max(-diff, 0),
                     offense_timeouts=to, defense_timeouts=dto, clock_running=running)


def test_bucket_functions_match_the_build_script():
    diffs = pd.Series(range(-30, 31))
    assert [late_score_class(int(d)) for d in diffs] == list(buckets.late_score_class(diffs))
    secs = pd.Series(range(-2, 300))
    assert [late_window2(int(s)) for s in secs] == list(buckets.late_window2(secs))


def test_tables_are_valid_and_cover_every_situation():
    for name in ("late_timeout_off", "late_timeout_def"):
        for (level, key), row in T._index[name].items():
            assert 0 <= row["p_timeout"] <= 1 and row["n"] >= 30, (name, level, key, row)
        first = 1 if name == "late_timeout_off" else 11
        for clock in range(first, 241, 3):
            for diff in (-20, -10, -5, -2, 0):
                for yl in (5, 40, 60, 90):
                    for down in (1, 2, 3, 4):
                        st = q4(clock, diff, down, yl)
                        assert T.lookup_or_none(name, st.late_timeout_situation()) is not None, (name, clock, diff, yl, down)


def test_rates_follow_the_real_data():
    def p(clock, diff=0, table="late_timeout_off", yl=57, down=1):
        return T.lookup_or_none(table, q4(clock, diff, down, yl).late_timeout_situation())[1]["p_timeout"]
    assert 0.5 < p(8) < 0.9, p(8)
    assert 0.35 < p(18) < 0.7, p(18)
    assert 0.3 < p(24) < 0.65, p(24)
    assert 0.15 < p(40) < 0.5, p(40)
    assert p(100) < 0.2 and p(200) < 0.1
    assert p(8) > p(24) > p(55) > p(100) and p(100) < 0.1 and p(200) < 0.1
    d = p(30, 0, "late_timeout_def")
    assert 0.03 < d < 0.4, d


def test_the_engine_respects_who_can_call():
    rng = random.Random(1)
    n = 2000
    assert 0.3 < sum(late.offense_timeout(q4(23), T, rng) for _ in range(n)) / n < 0.65
    assert not any(late.offense_timeout(q4(23, to=0), T, rng) for _ in range(n)), "no timeouts left"
    assert not any(late.offense_timeout(q4(23, diff=3), T, rng) for _ in range(n)), "a team that is ahead does not stop its clock"
    assert not any(late.offense_timeout(q4(23, running=False), T, rng) for _ in range(n)), "clock stopped"
    assert not any(late.offense_timeout(q4(300), T, rng) for _ in range(n)), "more than four minutes left"
    assert not any(late.defense_timeout(q4(23, dto=0), T, rng) for _ in range(n)), "no timeouts left"
    assert not any(late.defense_timeout(q4(23, diff=7), T, rng) for _ in range(n)), "a defense that is behind uses clock_rules"
    st = q4(23)
    st.quarter = 3
    assert not any(late.offense_timeout(st, T, rng) for _ in range(n)), "only the 4th quarter"


def test_a_team_that_needs_the_ball_never_lets_the_clock_expire():
    assert late.hurry_runoff(q4(20), 25) == 17
    assert late.hurry_runoff(q4(2), 25) == 0
    assert late.hurry_runoff(q4(20, diff=-3), 20) == 17
    assert late.hurry_runoff(q4(20), 10) == 10
    assert late.hurry_runoff(q4(20, diff=3), 25) == 25
    assert late.hurry_runoff(q4(20, diff=-9), 25) == 25
    assert late.hurry_runoff(q4(20, diff=-8), 25) == 17
    assert late.hurry_runoff(q4(20, running=False), 25) == 25
    st = q4(20)
    st.quarter = 2
    assert late.hurry_runoff(st, 25) == 25


def test_games_never_end_with_a_trailing_offense_letting_time_expire():
    runs = used = 0
    for seed in range(300):
        g = Game(T, random.Random(seed))
        while not g.over or g._queue:
            e = g.step()
            if e.kind == "time" and e.state.quarter == 4:
                runs += 1
                assert not (-8 <= e.state.score_diff <= 0 and e.state.clock_running), ("a team that needed the ball let time expire", seed, e.state)
            if e.kind == "timeout" and e.state.quarter == 4:
                used += 1
    assert used > 100, used
    print(f"  {used} 4th quarter timeouts in 300 games, {runs} quarters ended by the clock with a leading offense")


def test_timeouts_between_20_seconds_and_4_minutes_now_happen():
    mid = 0
    for seed in range(300):
        g = Game(T, random.Random(seed))
        while not g.over or g._queue:
            e = g.step()
            if e.kind == "timeout" and e.state.quarter == 4 and e.state.score_diff <= 0 and 20 < e.state.clock <= 240:
                mid += 1
    assert mid > 20, mid
    print(f"  {mid} timeouts by or against a tied or trailing offense with 20 s to 4 min left, in 300 games")


def test_a_team_in_field_goal_range_stops_the_clock_when_time_is_nearly_out():
    tables = Tables()
    rng = random.Random(5)
    for clock in (1, 4, 6, 10):
        for diff in (0, -2, -3):
            s = GameState(quarter=4, clock=clock, down=2, distance=2, yardline_100=2, offense_score=14 + diff,
                          defense_score=14, offense_timeouts=3, defense_timeouts=3, clock_running=True)
            assert all(late.offense_timeout(s, tables, rng) for _ in range(200)), (clock, diff)
    for bad in (dict(offense_score=20), dict(offense_timeouts=0)):
        args = dict(quarter=4, clock=5, down=2, distance=2, yardline_100=2, offense_score=14, defense_score=14,
                    offense_timeouts=3, defense_timeouts=3, clock_running=True)
        args.update(bad)
        assert not any(late.offense_timeout(GameState(**args), tables, rng) for _ in range(200)), bad


def test_no_defense_timeout_with_nothing_to_save():
    tables = Tables()
    for seed in range(300):
        g = Game(tables, random.Random(9000 + seed))
        while not g.over or g._queue:
            e = g.step()
            st = e.state
            if st and e.kind == "timeout" and e.detail == "defense" and st.quarter in (2, 4):
                assert not (st.clock < 20 and st.down < 4), (seed, e.text, st)


def test_a_two_minute_drill_uses_the_real_hurry_up_clock():
    rng = random.Random(3)
    drill = [T.hurry_seconds(q4(73, diff=-1, to=2), "run_in", rng) for _ in range(3000)]
    drill.sort()
    assert 18 <= drill[1500] <= 25, drill[1500]
    assert sum(s >= 30 for s in drill) / len(drill) < 0.35
    assert T.hurry_seconds(q4(73, diff=3), "run_in", rng) is None, "a leading team is not hurrying"
    assert T.hurry_seconds(q4(400, diff=-1), "run_in", rng) is None, "only inside 2:00"
    assert T.hurry_seconds(q4(73, diff=-1), "incomplete", rng) is None, "an incomplete pass already stops the clock"
    assert T.hurry_seconds(GameState(quarter=3, clock=73), "run_in", rng) is None


if __name__ == "__main__":
    test_bucket_functions_match_the_build_script()
    test_tables_are_valid_and_cover_every_situation()
    test_rates_follow_the_real_data()
    test_the_engine_respects_who_can_call()
    test_a_team_that_needs_the_ball_never_lets_the_clock_expire()
    test_games_never_end_with_a_trailing_offense_letting_time_expire()
    test_timeouts_between_20_seconds_and_4_minutes_now_happen()
    test_no_defense_timeout_with_nothing_to_save()
    test_a_team_in_field_goal_range_stops_the_clock_when_time_is_nearly_out()
    print("All late timeout tests passed")
    test_a_two_minute_drill_uses_the_real_hurry_up_clock()
