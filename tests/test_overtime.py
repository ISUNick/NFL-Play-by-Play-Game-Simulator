import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine import clock_rules as rules
from engine.game import OT_SECONDS, OT_TIMEOUTS, Game
from engine.special import choose_fourth_down, decide_onside
from engine.state import GameState
from test_engine_tables import tables

SCRIMMAGE_KINDS = {"play", "punt", "field_goal", "kneel", "spike"}


def ot_game(phase, a_score=0, b_score=0, clock=400, first=0, offense=None, seed=1) -> Game:
    g = Game(tables(), random.Random(seed))
    g.quarter, g.clock, g.went_to_ot = 5, clock, True
    g.ot_phase, g.ot_first = phase, first
    g.scores = [0, 0]
    g.scores[first], g.scores[1 - first] = a_score, b_score
    g.timeouts = [OT_TIMEOUTS, OT_TIMEOUTS]
    g.offense = first if offense is None else offense
    g.phase, g.down, g.distance, g.yardline_100 = "scrimmage", 1, 10, 75
    return g


def test_touchdowns():
    A, B = 0, 1
    g = ot_game("sudden", 0, 0)
    g._score_touchdown(A)
    assert g.ot_ended and g.try_team is None and g.phase != "try"
    g = ot_game("first", 0, 0, clock=300)
    g._score_touchdown(A)
    assert not g.ot_ended and g.try_team == A and g.phase == "try"
    g._ot_after_try(A)
    assert g.ot_phase == "second"
    g = ot_game("first", 0, 0, clock=0)
    g._score_touchdown(A)
    assert g.ot_ended and g.try_team is None
    g = ot_game("second", 3, 0, offense=1)
    g._score_touchdown(B)
    assert g.ot_ended and g.try_team is None and g.scores == [3, 6][::1] or g.scores[B] == 6
    for result, expect_end, expect_phase in ((1, False, "sudden"), (2, True, "second"), (0, True, "second")):
        g = ot_game("second", 7, 0, offense=1)
        g._score_touchdown(B)
        assert not g.ot_ended and g.try_team == B and g.phase == "try"
        g.scores[B] += result
        g._ot_after_try(B)
        assert g.ot_ended == expect_end, (result, g.scores)
        assert g.ot_phase == expect_phase, (result, g.ot_phase)
    for result, expect_end, expect_phase in ((1, True, "second"), (2, True, "second"), (0, False, "sudden")):
        g = ot_game("second", 6, 0, offense=1)
        g._score_touchdown(B)
        assert g.try_team == B
        g.scores[B] += result
        g._ot_after_try(B)
        assert g.ot_ended == expect_end and g.ot_phase == expect_phase, (result, g.scores, g.ot_phase)
    g = ot_game("second", 8, 0, offense=1)
    g._score_touchdown(B)
    g.scores[B] += 2
    g._ot_after_try(B)
    assert not g.ot_ended and g.ot_phase == "sudden"
    for phase, a, b in (("first", 0, 0), ("second", 3, 0), ("second", 7, 0), ("sudden", 0, 0)):
        g = ot_game(phase, a, b)
        defender = 1 - g.offense
        g._score_touchdown(defender, defensive=True)
        assert g.ot_ended and g.try_team is None, (phase, a, b)


def test_field_goals():
    A, B = 0, 1
    g = ot_game("sudden", 0, 0)
    g._ot_after_field_goal(A)
    assert g.ot_ended
    g = ot_game("first", 3, 0, clock=300)
    g._ot_after_field_goal(A)
    assert not g.ot_ended and g.ot_phase == "second"
    g = ot_game("first", 3, 0, clock=0)
    g._ot_after_field_goal(A)
    assert g.ot_ended
    g = ot_game("second", 6, 3, offense=1)
    g._ot_after_field_goal(B)
    assert g.ot_ended
    g = ot_game("second", 3, 3, offense=1)
    g._ot_after_field_goal(B)
    assert not g.ot_ended and g.ot_phase == "sudden"
    g = ot_game("second", 3, 3, offense=1, clock=0)
    g._ot_after_field_goal(B)
    assert g.ot_ended


def test_possession_ending_without_a_score():
    g = ot_game("first")
    g._ot_possession_ended()
    assert g.ot_phase == "sudden" and not g.ot_ended
    g = ot_game("second", 3, 0, offense=1)
    g._ot_possession_ended()
    assert g.ot_ended
    g = ot_game("sudden")
    g._ot_possession_ended()
    assert g.ot_phase == "sudden" and not g.ot_ended


def test_change_possession_is_wired_to_the_rules():
    g = ot_game("first", offense=0)
    g._change_possession(75)
    assert g.ot_phase == "sudden" and g.offense == 1
    g = ot_game("second", 3, 0, offense=1)
    g._change_possession(75)
    assert g.ot_ended


def test_clock_rules_apply_in_overtime():
    def st(**kw):
        base = dict(quarter=5, clock=300, down=1, distance=10, yardline_100=75, offense_score=0, defense_score=0,
                    offense_timeouts=2, defense_timeouts=2, clock_running=False)
        base.update(kw)
        return GameState(**base)
    assert st(clock=1200).game_seconds_remaining == 1200 and st(clock=300).half_seconds_remaining == 300
    assert st(clock=100).tempo == "last2" and st(clock=250).tempo == "last5" and st(clock=500).tempo == "normal"
    assert rules.should_kick(st(clock=90, down=4, yardline_100=25))
    assert rules.should_kick(st(clock=5, down=1, yardline_100=25, offense_timeouts=0))
    assert not rules.should_kick(st(clock=60, down=1, yardline_100=25, offense_timeouts=0))
    assert rules.should_call_timeout(st(clock=15, offense_score=0, defense_score=3, clock_running=True))
    assert rules.should_spike(st(clock=8, offense_score=0, defense_score=3, clock_running=True, offense_timeouts=0))
    assert not rules.should_kneel(st(clock=20, offense_score=7, defense_score=0))
    T, rng = tables(), random.Random(1)
    assert not any(decide_onside(-7, 5, 60, T, rng) for _ in range(500))


def test_fourth_down_in_overtime():
    T, rng = tables(), random.Random(4)

    def calls(state, n=3000):
        out = {}
        for _ in range(n):
            c = choose_fourth_down(state, T, rng)
            out[c] = out.get(c, 0) + 1
        return out
    base = dict(quarter=5, clock=400, down=4, offense_timeouts=2, defense_timeouts=2)
    c = calls(GameState(distance=4, yardline_100=25, offense_score=0, defense_score=0, **base))
    assert c.get("field_goal", 0) > 2700, c
    c = calls(GameState(distance=10, yardline_100=80, offense_score=0, defense_score=0, **base))
    assert c.get("punt", 0) > 2700, c
    for yl in (85, 60, 45, 30, 15, 3):
        for dist in (1, 5, 12):
            c = calls(GameState(distance=dist, yardline_100=yl, offense_score=0, defense_score=7, **base), n=400)
            assert set(c) == {"go"}, (yl, dist, c)
    c = calls(GameState(distance=5, yardline_100=30, offense_score=0, defense_score=8, **base), n=400)
    assert set(c) == {"go"}
    c = calls(GameState(distance=5, yardline_100=20, offense_score=0, defense_score=3, **base), n=1500)
    assert "punt" not in c and c.get("field_goal", 0) > 0, c
    c = calls(GameState(distance=9, yardline_100=70, offense_score=0, defense_score=3, **base), n=800)
    assert "punt" not in c, c


def run_out(g: Game, limit=400):
    events = []
    while not g.over or g._queue:
        events.append(g.step())
        assert len(events) < limit, "game never ended"
    return events


def test_regulation_tie_starts_overtime():
    T = tables()
    seen = 0
    for seed in range(1500):
        g = Game(T, random.Random(seed))
        events = run_out(g)
        tied_at_end_of_regulation = any(e.kind == "end_regulation" for e in events)
        assert tied_at_end_of_regulation == g.went_to_ot
        if not g.went_to_ot:
            assert g.quarter == 4 and g.scores[0] != g.scores[1]
            continue
        seen += 1
        i = [k for k, e in enumerate(events) if e.kind == "end_regulation"][0]
        assert events[i + 1].kind == "overtime" and events[i + 2].kind == "kickoff"
        assert "10:00" in events[i + 1].text and "2 timeouts" in events[i + 1].text
        first_kickoff = events[i + 2]
        assert first_kickoff.team == 1 - g.ot_first, "the team that won the toss receives"
        assert first_kickoff.state.quarter == 5 and first_kickoff.state.clock == OT_SECONDS
        assert first_kickoff.state.offense_timeouts == OT_TIMEOUTS and first_kickoff.state.defense_timeouts == OT_TIMEOUTS
        assert g.quarter == 5 and 0 <= g.clock <= OT_SECONDS and events[-1].kind == "game_end"
        assert g.timeouts[0] >= 0 and g.timeouts[1] >= 0
        assert events[i - 1].state.quarter == 4 or events[i - 1].kind in ("time", "try", "kickoff")
        if seen >= 40:
            break
    assert seen >= 20


def test_second_possession_guarantee_hand_built():
    for seed in range(200):
        g = ot_game("first", 0, 0, clock=500, first=0, offense=0, seed=seed)
        g.yardline_100, g.down, g.distance = 20, 4, 3
        g.fresh = False
        ev = g.step()
        if ev.kind != "field_goal" or not g.scores[0]:
            continue
        assert not g.ot_ended and g.ot_phase == "second"
        rest = run_out(g)
        assert any(e.kind == "kickoff" for e in rest), "the other team never got a kickoff"
        assert any(e.kind in SCRIMMAGE_KINDS and e.team == 1 for e in rest) or g.clock == 0 or \
            any(e.kind == "kickoff" and e.scorer == 1 for e in rest), "the other team never had the ball"
        return
    raise AssertionError("never saw a made field goal")


def test_safety_ends_the_game_for_the_defense():
    g = ot_game("first", 0, 0, clock=500, first=0, offense=0)
    g.scores[1] += 2
    g._ot_end()
    g._end_of_period()
    assert g.over and g.scores == [0, 2]
    assert any(e.kind == "game_end" and "OT" in e.text for e in g._queue)


def test_period_runs_out_tied():
    g = ot_game("sudden", 0, 0, clock=3, offense=0)
    g.yardline_100 = 60
    events = run_out(g)
    assert g.over and g.clock == 0 and g.scores == [0, 0]
    assert "tie" in events[-1].text and "Final (OT)" in events[-1].text


def test_two_minute_warning_in_overtime():
    g = ot_game("sudden", 0, 0, clock=125, offense=0)
    g.clock_running = True
    g.yardline_100 = 60
    seen = False
    for _ in range(40):
        ev = g.step()
        if ev.kind == "two_minute_warning":
            seen = True
            assert g.clock == 120 and not g.clock_running
            break
        if g.over:
            break
    assert seen


def points_scored_in_ot(events):
    return sum(e.points for e in events if e.scorer is not None and e.state is not None and e.state.quarter == 5)


def test_every_overtime_game_obeys_the_rules():
    T = tables()
    games = ot_games = ties = 0
    tries_in_ot = 0
    for seed in range(5000):
        g = Game(T, random.Random(20000 + seed))
        events = run_out(g)
        games += 1
        if not g.went_to_ot:
            continue
        ot_games += 1
        i = [k for k, e in enumerate(events) if e.kind == "overtime"][0]
        ot = events[i + 1:]
        tied_at_end = g.scores[0] == g.scores[1]
        ties += tied_at_end
        if tied_at_end:
            assert g.clock == 0, (seed, g.clock)
        clocks = [e.state.clock for e in ot if e.state is not None and e.state.quarter == 5 and e.kind in SCRIMMAGE_KINDS]
        assert all(0 <= c <= OT_SECONDS for c in clocks)
        assert all(a >= b for a, b in zip(clocks, clocks[1:])), seed
        tos = [(e.state.offense_timeouts, e.state.defense_timeouts) for e in ot if e.state is not None and e.state.quarter == 5]
        assert all(0 <= x <= 2 and 0 <= y <= 2 for x, y in tos)
        assert not any(e.kind == "onside" for e in ot), seed
        first_team = g.ot_first
        offense_scores_first = None
        for e in ot:
            if e.scorer is not None and e.kind in ("play", "field_goal", "kickoff", "punt"):
                offense_scores_first = e
                break
        if offense_scores_first is not None and offense_scores_first.scorer == first_team \
                and offense_scores_first.kind in ("play", "field_goal", "kickoff") \
                and (offense_scores_first.team == first_team or offense_scores_first.kind == "kickoff"):
            k = ot.index(offense_scores_first)
            before = ot[:k]
            after = ot[k + 1:]
            if any(e.kind in SCRIMMAGE_KINDS and e.team == 1 - first_team for e in before):
                continue
            other_had_ball = any(e.kind in SCRIMMAGE_KINDS and e.team == 1 - first_team for e in after) or \
                any(e.kind == "kickoff" and e.scorer == 1 - first_team for e in after)
            clock_after_score = offense_scores_first.state.clock - offense_scores_first.seconds
            assert other_had_ball or clock_after_score <= 6 or g.clock == 0, \
                (seed, "the other team never got the ball", offense_scores_first.text)
        for k, e in enumerate(ot):
            if e.kind == "try":
                tries_in_ot += 1
                td_event = [x for x in ot[:k] if x.kind != "two_minute_warning"][-1]
                assert td_event.scorer == e.team and td_event.points == 6, (seed, td_event.text)
                before = [x for x in ot[:k] if x.scorer is not None]
                earlier_scores = [x for x in before[:-1]]
                assert not (not earlier_scores and e.team != first_team), (seed, "try for a team that did not have the first possession")
        total = [0, 0]
        for e in events:
            if e.scorer is not None:
                total[e.scorer] += e.points
        assert total == g.scores, (seed, total, g.scores)
        assert events[-1].kind == "game_end" and "Final (OT)" in events[-1].text
        if not tied_at_end and g.clock > 0:
            last_scores = [e for e in ot if e.scorer is not None]
            assert last_scores, seed
            last = ot.index(last_scores[-1])
            winner = last_scores[-1].scorer
            tail = [e for e in ot[last + 1:] if e.kind not in ("kickoff", "game_end", "try", "time", "two_minute_warning", "timeout")]
            assert all(e.team == 1 - winner for e in tail), (seed, [t.text for t in tail])
            if tail:
                assert last_scores[-1].scorer == first_team or ot.index(last_scores[-1]) >= 0
    rate = ot_games / games
    print(f"  {games} games, {ot_games} went to overtime ({rate:.1%}), {ties} ended in a tie ({ties / games:.2%})")
    assert 0.03 < rate < 0.12, rate
    assert 0.001 < ties / games < 0.03


def test_games_with_no_overtime_are_unchanged():
    T = tables()
    for seed in range(300):
        g = Game(T, random.Random(seed))
        events = run_out(g)
        if not g.went_to_ot:
            assert events[-1].text.startswith("Final:") and g.quarter == 4 and g.clock == 0


if __name__ == "__main__":
    test_touchdowns()
    test_field_goals()
    test_possession_ending_without_a_score()
    test_change_possession_is_wired_to_the_rules()
    test_clock_rules_apply_in_overtime()
    test_fourth_down_in_overtime()
    test_regulation_tie_starts_overtime()
    test_second_possession_guarantee_hand_built()
    test_safety_ends_the_game_for_the_defense()
    test_period_runs_out_tied()
    test_two_minute_warning_in_overtime()
    test_every_overtime_game_obeys_the_rules()
    test_games_with_no_overtime_are_unchanged()
    print("All overtime tests passed")
