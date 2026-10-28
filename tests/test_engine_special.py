import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.special import (
    FIELD_GOAL, GO, PUNT, apply_punt, choose_fourth_down, decide_two_point,
    resolve_kickoff, resolve_onside, resolve_punt, resolve_try,
)
from engine.clock_rules import kick_distance
from engine.state import GameState
from test_engine_tables import tables

PUNT_KINDS = {"touchback", "returned", "fair_catch", "oob", "downed", "blocked", "return_td", "fumble_lost"}


def test_punt_geometry_exact_cases():
    r = apply_punt(70, (45, "fair_catch", 0, 0, 0, 1, 0))
    assert (r.kind, r.receiver_has_ball, r.yardline_100, r.touchdown) == ("fair_catch", True, 75, False)
    r = apply_punt(70, (45, "returned", 10, 0, 0, 1, 0))
    assert (r.yardline_100, r.return_yards) == (65, 10)
    r = apply_punt(45, (55, "returned", 9, 0, 0, 1, 0))
    assert (r.kind, r.yardline_100) == ("touchback", 80)
    assert apply_punt(60, (40, "touchback", 0, 0, 0, 1, 0)).yardline_100 == 80
    r = apply_punt(70, (45, "return_td", 75, 1, 0, 1, 0))
    assert r.touchdown and r.yardline_100 == 0 and r.receiver_has_ball
    r = apply_punt(70, (45, "returned", 10, 0, 1, 1, 0))
    assert (r.kind, r.receiver_has_ball, r.yardline_100) == ("fumble_lost", False, 35)
    r = apply_punt(80, (0, "blocked", 0, 0, 0, 1, 8))
    assert (r.kind, r.receiver_has_ball, r.yardline_100) == ("blocked", True, 12)
    r = apply_punt(80, (0, "blocked", 0, 0, 0, 0, -6))
    assert (r.receiver_has_ball, r.yardline_100) == (False, 74)
    r = apply_punt(70, (45, "returned", 90, 0, 0, 1, 0))
    assert r.yardline_100 == 1 and not r.touchdown


def test_every_real_punt_gives_a_legal_result():
    t = tables()
    for yard in [1, 5, 20, 28, 35, 50, 65, 80, 95, 99]:
        for rows in t.punt.values():
            for sample in rows:
                r = apply_punt(yard, sample)
                assert r.kind in PUNT_KINDS, r
                if r.touchdown:
                    assert r.yardline_100 == 0 and r.receiver_has_ball
                else:
                    assert 1 <= r.yardline_100 <= 99, (yard, sample, r)


def test_punts_in_bulk():
    t = tables()
    rng = random.Random(11)
    nets, touchbacks = [], 0
    for _ in range(20000):
        r = resolve_punt(80, t, rng)
        if r.receiver_has_ball and not r.touchdown:
            nets.append(80 + r.yardline_100 - 100)
    assert 33 < sum(nets) / len(nets) < 46, "net punt from your own 20 should be near 40"
    for _ in range(20000):
        touchbacks += resolve_punt(40, t, rng).kind == "touchback"
    assert 0.1 < touchbacks / 20000 < 0.4, "punts from the opponent's 40 are often touchbacks"
    assert all(resolve_punt(92, t, rng).kind != "touchback" for _ in range(5000)), "no touchback from your own 8"


def test_kickoffs_and_onside_kicks():
    t = tables()
    rng = random.Random(12)
    starts, tds = [], 0
    for _ in range(20000):
        k = resolve_kickoff(t, rng)
        if k.touchdown:
            tds += 1
            assert k.yardline_100 == 0
        else:
            assert 1 <= k.yardline_100 <= 99
            starts.append(k.yardline_100)
    assert 62 < sum(starts) / len(starts) < 75, "kickoffs start around the 30"
    assert 0 < tds / 20000 < 0.01
    recovered = 0
    for _ in range(20000):
        o = resolve_onside(t, rng)
        assert 1 <= o.yardline_100 <= 99
        recovered += o.kicking_team_recovers
    assert 0.03 < recovered / 20000 < 0.25


def test_tries():
    t = tables()
    rng = random.Random(13)
    n = 20000
    xp = [resolve_try(False, t, rng) for _ in range(n)]
    assert abs(sum(r.points for r in xp) / n - t.extra_point["p_good"]) < 0.015
    assert {r.kind for r in xp} <= {"xp_good", "xp_failed", "xp_blocked"} and all(r.points in (0, 1) for r in xp)
    two = [resolve_try(True, t, rng) for _ in range(n)]
    assert abs(sum(r.points for r in two) / n / 2 - t.two_point["p_success"]) < 0.02
    assert all(r.points in (0, 2) for r in two)

    def go_rate(diff, quarter):
        return sum(decide_two_point(diff, quarter, t, rng) for _ in range(4000)) / 4000
    assert go_rate(6, 1) < 0.025, "up 6 after a first quarter touchdown: kick it"
    assert go_rate(-2, 4) > 0.9, "down 2 in the 4th after a touchdown: go for two"
    assert go_rate(30, 2) < 0.05 and go_rate(-40, 4) < 0.6


def make_state(quarter, clock, distance, yardline, offense_score, defense_score) -> GameState:
    return GameState(quarter=quarter, clock=clock, down=4, distance=distance, yardline_100=yardline,
                     offense_score=offense_score, defense_score=defense_score)


def test_fourth_down_guards_never_break():
    t = tables()
    rng = random.Random(14)
    counts = {GO: 0, PUNT: 0, FIELD_GOAL: 0}
    for _ in range(40000):
        s = make_state(rng.randint(1, 4), rng.randint(0, 900), rng.randint(1, 25), rng.randint(1, 99),
                       rng.randint(0, 40), rng.randint(0, 40))
        call = choose_fourth_down(s, t, rng)
        counts[call] += 1
        late = s.quarter == 4 and s.game_seconds_remaining <= 120
        if call == PUNT:
            assert s.yardline_100 >= 28, f"punted from inside the opponent's 28: {s}"
        if call == FIELD_GOAL:
            assert kick_distance(s) <= 66, f"kicked a field goal over 66 yards: {s}"
        if late and s.score_diff < 0:
            assert call != PUNT, f"punted while trailing in the last two minutes: {s}"
        if late and s.score_diff > 0 and s.yardline_100 >= 50:
            assert call == PUNT, f"did not punt leading in the last two minutes in own half: {s}"
        if late and -8 <= s.score_diff <= -4:
            assert call != FIELD_GOAL, f"kicked a field goal that cannot tie or win: {s}"
    assert all(v > 1000 for v in counts.values()), counts


def test_fourth_down_follows_the_data():
    t = tables()
    rng = random.Random(15)

    def share(state, call, n=4000):
        return sum(choose_fourth_down(state, t, rng) == call for _ in range(n)) / n
    assert share(make_state(2, 600, 1, 38, 10, 10), GO) > 0.85, "4th and 1 at the opponent's 38: go"
    assert share(make_state(2, 600, 10, 70, 10, 10), PUNT) > 0.9, "4th and 10 at your own 30: punt"
    assert share(make_state(2, 600, 3, 22, 10, 10), FIELD_GOAL) > 0.6, "4th and 3 at the opponent's 22: kick"


def test_endgame_scenarios():
    t = tables()
    rng = random.Random(16)
    s = make_state(4, 40, 8, 65, 17, 21)
    assert all(choose_fourth_down(s, t, rng) != PUNT for _ in range(3000))
    s = make_state(4, 60, 6, 70, 24, 21)
    assert all(choose_fourth_down(s, t, rng) == PUNT for _ in range(3000))
    s = make_state(4, 90, 12, 50, 10, 24)
    assert all(choose_fourth_down(s, t, rng) != PUNT for _ in range(3000))


def test_late_game_table_follows_the_data():
    t = tables()
    rng = random.Random(17)
    s = make_state(4, 23, 7, 7, 21, 27)
    assert all(choose_fourth_down(s, t, rng) == GO for _ in range(3000))
    s = make_state(4, 23, 7, 7, 24, 27)
    assert sum(choose_fourth_down(s, t, rng) == FIELD_GOAL for _ in range(3000)) > 2400
    level, row = t.lookup_late("fourth_down_late", "fourth_down", make_state(4, 200, 4, 25, 17, 24))
    assert level.startswith("late_") and row["p_go"] > 0.5, (level, row)
    level, _ = t.lookup_late("fourth_down_late", "fourth_down", make_state(4, 500, 4, 25, 17, 24))
    assert not level.startswith("late_"), level


def test_late_fourth_and_short_is_not_averaged_with_fourth_and_long():
    t = tables()
    rng = random.Random(3)

    def go_share(quarter, clock, distance, yl, offense_score, defense_score, n=3000):
        s = make_state(quarter, clock, distance, yl, offense_score, defense_score)
        return sum(choose_fourth_down(s, t, rng) == GO for _ in range(n)) / n
    short = go_share(4, 300, 1, 55, 21, 24)
    long_ = go_share(4, 300, 10, 55, 21, 24)
    assert short > 0.6 and short > long_ + 0.35, (short, long_)
    assert go_share(2, 300, 1, 50, 10, 10) > go_share(2, 300, 8, 50, 10, 10) + 0.25
    cells = {}
    for (level, key), r in t._index["fourth_down_late"].items():
        if level == "L3":
            cells[key] = r
    keys = [k for l, k in t._levels["fourth_down_late"] if l == "L3"][0]
    assert keys == ["late_zone", "late_fdist", "late_score"], keys
    checked = 0
    for (zone, dist, score), r in cells.items():
        other = cells.get((zone, "7p", score))
        if dist == "1" and other is not None and r["n"] >= 30 and other["n"] >= 30:
            checked += 1
            assert r["p_go"] >= other["p_go"] - 0.05, (zone, score, r["p_go"], other["p_go"])
    assert checked >= 3, checked


if __name__ == "__main__":
    test_punt_geometry_exact_cases()
    test_every_real_punt_gives_a_legal_result()
    test_punts_in_bulk()
    test_kickoffs_and_onside_kicks()
    test_tries()
    test_fourth_down_guards_never_break()
    test_fourth_down_follows_the_data()
    test_endgame_scenarios()
    test_late_game_table_follows_the_data()
    test_late_fourth_and_short_is_not_averaged_with_fourth_and_long()
    print("All engine special teams tests passed")
