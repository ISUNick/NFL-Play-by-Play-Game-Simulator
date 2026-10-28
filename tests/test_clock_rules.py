import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.clock_rules import (
    Action, KNEEL_AFTER_TIMEOUT_SECONDS, KNEEL_SECONDS, can_run_out_clock,
    decide_after_play, decide_before_snap, kick_distance,
)
from engine.state import GameState

K, N, FG, SP, TO = (Action.KNEEL, Action.NORMAL, Action.FIELD_GOAL,
                    Action.SPIKE, Action.TIMEOUT)


def state(seconds, down=1, def_timeouts=0, lead=3, yardline_100=70, quarter=4,
          off_timeouts=0, running=False):
    return GameState(quarter=quarter, clock=seconds, down=down,
                     offense_score=20 + lead, defense_score=20,
                     defense_timeouts=def_timeouts, offense_timeouts=off_timeouts,
                     yardline_100=yardline_100, clock_running=running)


def test_kneel_situations():
    cases = [
        ("2:00 left, up 3, 1st down, defense has no timeouts", state(120), K),
        ("2:30 left, up 3, 1st down, defense has no timeouts (not enough kneels)", state(150), N),
        ("2:00 left, up 3, 1st down, defense has 1 timeout", state(120, def_timeouts=1), N),
        ("1:20 left, up 3, 1st down, defense has 1 timeout", state(80, def_timeouts=1), K),
        ("1:00 left, up 3, 1st down, defense has 3 timeouts", state(60, def_timeouts=3), N),
        ("1:00 left, up 3, 1st down, defense has 0 timeouts", state(60), K),
        ("1:20 left, up 3, 2nd down, defense has 0 timeouts", state(80), K),
        ("0:30 left, up 3, 3rd down, defense has 0 timeouts", state(30, down=3), K),
        ("0:45 left, up 3, 3rd down, defense has 0 timeouts", state(45, down=3), N),
        ("0:30 left, up 3, 3rd down, defense has 1 timeout", state(30, down=3, def_timeouts=1), N),
        ("0:30 left, up 3, 4th down (teams punt on 4th)", state(30, down=4), N),
        ("1:00 left, tied", state(60, lead=0), N),
        ("1:00 left, trailing by 3", state(60, lead=-3), N),
        ("1:00 left in the 3rd quarter, up 3", state(60, quarter=3), N),
        ("3:00 left, up 3, 1st down, defense has no timeouts", state(180), N),
        ("1:00 left, up 3, own 1 yard line (safety danger)", state(60, yardline_100=99), N),
        ("1:00 left, up 3, own 5 yard line (fine to kneel)", state(60, yardline_100=95), K),
        ("game over, 0 seconds", state(0), N),
        ("0:50 left, up 17, defense has 3 timeouts", state(50, lead=17, def_timeouts=3), K),
        ("0:50 left, up 16, defense has 3 timeouts (two scores, not decided)",
         state(50, lead=16, def_timeouts=3), N),
        ("1:30 left, up 17, defense has 3 timeouts (too early)", state(90, lead=17, def_timeouts=3), N),
        ("0:50 left, up 17, 4th down (teams punt on 4th)", state(50, lead=17, down=4, def_timeouts=3), N),
        ("0:50 left, up 17, 3rd down, own 1 yard line (safety danger)",
         state(50, lead=17, down=3, def_timeouts=3, yardline_100=99), N),
    ]
    for description, s, expected in cases:
        got = decide_before_snap(s)
        assert got == expected, f"{description}: expected {expected}, got {got}"


def test_kneel_sequence_never_flips():
    checked = 0
    for lead in (3, 17):
        for seconds in range(1, 241):
            for down in range(1, 4):
                for timeouts in range(0, 4):
                    start = state(seconds, down=down, def_timeouts=timeouts, lead=lead)
                    if decide_before_snap(start) != Action.KNEEL:
                        continue
                    checked += 1
                    clock_kill = can_run_out_clock(start)
                    stack = [start]
                    while stack:
                        s = stack.pop()
                        assert decide_before_snap(s) == Action.KNEEL, (
                            f"flipped to NORMAL: seconds={s.clock} down={s.down} "
                            f"timeouts={s.defense_timeouts} (started at {seconds}s, "
                            f"down {down}, {timeouts} timeouts, lead {lead})")
                        follow(state(s.clock - KNEEL_SECONDS, down=s.down + 1,
                                     def_timeouts=s.defense_timeouts, lead=lead,
                                     yardline_100=s.yardline_100 + 1), stack, clock_kill)
                        if s.defense_timeouts > 0:
                            follow(state(s.clock - KNEEL_AFTER_TIMEOUT_SECONDS, down=s.down + 1,
                                         def_timeouts=s.defense_timeouts - 1, lead=lead,
                                         yardline_100=s.yardline_100 + 1), stack, clock_kill)
    assert checked > 100, "test did not check enough starting situations"


def follow(next_state, stack, clock_kill):
    if next_state.clock <= 0:
        return
    if next_state.down == 4:
        assert not clock_kill, (
            f"ran out of kneel downs with {next_state.clock}s left in a clock-kill "
            f"sequence, the other team gets the ball")
        return
    stack.append(next_state)


def test_kick_situations():
    cases = [
        ("4th down, down 2, 48 yards, 1:40 left", state(100, down=4, lead=-2, yardline_100=30), FG),
        ("4th down, down 3, 48 yards, 1:40 left (a kick ties it)", state(100, down=4, lead=-3, yardline_100=30), FG),
        ("4th down, tied, 48 yards, 0:20 left (a kick wins it)", state(20, down=4, lead=0, yardline_100=30), FG),
        ("4th down, down 4, 48 yards (a kick only gets within 1)", state(100, down=4, lead=-4, yardline_100=30), N),
        ("4th down, down 2, 48 yards, 4:00 left (too early, general 4th down logic)",
         state(240, down=4, lead=-2, yardline_100=30), N),
        ("4th down, down 2, 58 yards, 0:20 left (long, but time is gone)",
         state(20, down=4, lead=-2, yardline_100=40), FG),
        ("4th down, down 2, 58 yards, 1:40 left (long, plenty of time)",
         state(100, down=4, lead=-2, yardline_100=40), N),
        ("4th down, down 2, 62 yards, 0:05 left (out of range, desperation)",
         state(5, down=4, lead=-2, yardline_100=44), N),
        ("4th down, leading by 2 (not a rule for this team)", state(20, down=4, lead=2, yardline_100=30), N),
        ("3rd down, down 2, 48 yards, 10s left, clock stopped, no timeouts",
         state(10, down=3, lead=-2, yardline_100=30), FG),
        ("3rd down, down 2, 48 yards, 11s left, clock stopped, no timeouts (run another play)",
         state(11, down=3, lead=-2, yardline_100=30), N),
        ("3rd down, down 2, 48 yards, 7s left, clock stopped, has a timeout (squeeze in a play)",
         state(7, down=3, lead=-2, yardline_100=30, off_timeouts=1), N),
        ("3rd down, down 2, 48 yards, 6s left, clock stopped, has a timeout",
         state(6, down=3, lead=-2, yardline_100=30, off_timeouts=1), FG),
        ("1st down, down 2, 48 yards, 8s left, clock RUNNING, no timeouts (spike first, never kick into a running clock)",
         state(8, down=1, lead=-2, yardline_100=30, running=True), SP),
        ("1st down, down 2, 48 yards, 8s left, clock RUNNING, has a timeout (the timeout comes after the play, so no kick yet)",
         state(8, down=1, lead=-2, yardline_100=30, running=True, off_timeouts=1), N),
        ("1st down, down 2, 58 yards, 5s left, clock stopped (too far for the 1st-3rd down kick)",
         state(5, down=1, lead=-2, yardline_100=40), N),
        ("3rd quarter, 4th down, down 2", state(20, down=4, lead=-2, yardline_100=30, quarter=3), N),
    ]
    for description, s, expected in cases:
        got = decide_before_snap(s)
        assert got == expected, f"{description}: expected {expected}, got {got}"
    assert kick_distance(state(10, yardline_100=33)) == 51
    assert kick_distance(state(10, yardline_100=1)) == 19


def test_spike_and_timeout_situations():
    spike_cases = [
        ("clock running, down 2, in range, 15s, no timeouts, 2nd down",
         state(15, down=2, lead=-2, yardline_100=30, running=True), SP),
        ("same but 21s left (gray zone, data decides)",
         state(21, down=2, lead=-2, yardline_100=30, running=True), N),
        ("same but the clock is stopped (nothing to save)",
         state(15, down=2, lead=-2, yardline_100=30, running=False), N),
        ("same but 3rd down (a spike would use the last chance before 4th)",
         state(15, down=3, lead=-2, yardline_100=30, running=True), N),
        ("same but the offense has a timeout (use it instead)",
         state(15, down=2, lead=-2, yardline_100=30, running=True, off_timeouts=1), N),
        ("out of range, down 5, 8s left, needs a touchdown",
         state(8, down=1, lead=-5, yardline_100=60, running=True), SP),
        ("out of range, down 5, 15s left (needs a TD, too early to spike by rule)",
         state(15, down=1, lead=-5, yardline_100=60, running=True), N),
        ("down 9, needs two scores, 8s left (not one score)",
         state(8, down=1, lead=-9, yardline_100=60, running=True), N),
        ("leading by 2, never spike (it kneels instead)",
         state(8, down=1, lead=2, yardline_100=30, running=True), K),
        ("1s left (no time to snap a spike)",
         state(1, down=1, lead=-2, yardline_100=30, running=True), N),
    ]
    for description, s, expected in spike_cases:
        got = decide_before_snap(s)
        assert got == expected, f"{description}: expected {expected}, got {got}"

    timeout_cases = [
        ("20s left, clock running, has timeout, 1st down, down 2",
         state(20, down=1, lead=-2, running=True, off_timeouts=1), TO),
        ("21s left (gray zone)", state(21, down=1, lead=-2, running=True, off_timeouts=1), N),
        ("20s left but no timeouts", state(20, down=1, lead=-2, running=True), N),
        ("20s left but the clock is stopped", state(20, down=1, lead=-2, off_timeouts=1), N),
        ("20s left, 4th down (kick or go, the clock does not matter yet)",
         state(20, down=4, lead=-2, running=True, off_timeouts=1), N),
        ("20s left, leading by 2", state(20, down=1, lead=2, running=True, off_timeouts=1), N),
        ("20s left, down 12, game is not close enough",
         state(20, down=1, lead=-12, running=True, off_timeouts=1), N),
    ]
    for description, s, expected in timeout_cases:
        got = decide_after_play(s)
        assert got == expected, f"{description}: expected {expected}, got {got}"


def test_thirty_second_scenario():
    s = state(24, down=2, lead=-2, yardline_100=30, running=True)
    assert decide_after_play(s) == N, "no timeouts to call"
    s = state(12, down=2, lead=-2, yardline_100=30, running=True)
    assert decide_before_snap(s) == SP, "12s left, clock running, in range: spike"
    s = state(11, down=3, lead=-2, yardline_100=30, running=False)
    assert decide_before_snap(s) == N, "11s: time for one more play to improve the kick"
    s = state(6, down=4, lead=-2, yardline_100=30, running=False)
    assert decide_before_snap(s) == FG, "4th down, in range, a kick wins: kick"

    s = state(18, down=2, lead=-2, yardline_100=30, running=True, off_timeouts=1)
    assert decide_after_play(s) == TO, "18s and a timeout in hand: stop the clock"
    s = state(18, down=2, lead=-2, yardline_100=30, running=False, off_timeouts=0)
    assert decide_before_snap(s) == N, "clock is stopped, 18s: run another play"
    s = state(8, down=4, lead=-2, yardline_100=25, running=False)
    assert decide_before_snap(s) == FG, "closer now (43 yards), 4th down: kick"


def test_rules_never_fire_when_senseless():
    leads = (-30, -12, -9, -8, -5, -4, -3, -2, 0, 1, 3, 8, 9, 17, 30)
    checked = 0
    for quarter, seconds, down, lead, off_to, running, yard in product(
            (1, 3, 4), (0, 1, 2, 5, 6, 10, 11, 20, 21, 30, 45, 60, 90, 120, 121, 300),
            (1, 2, 3, 4), leads, (0, 1, 3), (False, True), (3, 30, 60, 95, 99)):
        s = state(seconds, down=down, lead=lead, quarter=quarter,
                  off_timeouts=off_to, running=running, yardline_100=yard)
        before, after = decide_before_snap(s), decide_after_play(s)
        checked += 1
        if quarter != 4:
            assert before == N and after == N, "rules are 4th quarter only so far"
        if before == K:
            assert lead > 0 and down <= 3
        if before == FG:
            assert -3 <= lead <= 0, "a kick must tie or win"
            assert kick_distance(s) <= 60
            assert seconds > 0
            if down <= 3:
                assert not running and kick_distance(s) <= 55 and seconds <= 10
        if before == SP:
            assert running and off_to == 0 and down <= 2 and -8 <= lead <= 0
            assert 1 < seconds <= 20
        if after == TO:
            assert running and off_to > 0 and down <= 3 and -8 <= lead <= 0
            assert 0 < seconds <= 20
        if lead > 0:
            assert before in (K, N), "a team that is ahead never kicks or spikes here"
    assert checked > 50000


if __name__ == "__main__":
    test_kneel_situations()
    test_kneel_sequence_never_flips()
    test_kick_situations()
    test_spike_and_timeout_situations()
    test_thirty_second_scenario()
    test_rules_never_fire_when_senseless()
    print("All clock rule tests passed")
