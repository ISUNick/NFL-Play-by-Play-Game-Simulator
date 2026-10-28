import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine import clock_rules as rules
from engine import half
from engine.clock_rules import Action
from engine.game import Game
from test_engine_tables import tables

SCRIMMAGE_KINDS = {"play", "punt", "field_goal", "kneel", "spike"}


def play_whole_game(seed: int):
    g = Game(tables(), random.Random(seed))
    events = []
    last_scores = [0, 0]
    while not g.over or g._queue:
        ev = g.step()
        events.append(ev)
        check_game_state(g, last_scores)
        last_scores = list(g.scores)
    return g, events


def check_game_state(g: Game, last_scores) -> None:
    assert 1 <= g.quarter <= 5, g.quarter
    assert 0 <= g.clock <= (600 if g.quarter == 5 else 900), g.clock
    assert not (g.quarter == 5 and any(t > 2 for t in g.timeouts)), g.timeouts
    assert all(g.scores[i] >= last_scores[i] for i in (0, 1)), "the score went down"
    assert all(0 <= t <= 3 for t in g.timeouts), g.timeouts
    assert g.offense in (0, 1) and g.phase in ("kickoff", "scrimmage", "try")
    if g.phase == "scrimmage" and not g.over:
        assert 1 <= g.down <= 4, g.down
        assert 1 <= g.yardline_100 <= 99, g.yardline_100
        assert g.distance >= 1, g.distance
        assert g.distance <= g.yardline_100, ("more yards to go than to the end zone", g.distance, g.yardline_100)
    if g.phase == "try":
        assert g.try_team in (0, 1)


def test_two_hundred_full_games():
    for seed in range(200):
        g, events = play_whole_game(seed)
        if g.went_to_ot:
            assert g.over and g.quarter == 5 and (g.clock == 0 or g.scores[0] != g.scores[1]), (seed, g.clock)
        else:
            assert g.over and g.quarter == 4 and g.clock == 0, (seed, g.quarter, g.clock)
            assert g.scores[0] != g.scores[1], "a tied game must go to overtime"
        assert events[-1].kind == "game_end"
        total = [0, 0]
        for ev in events:
            if ev.scorer is not None:
                assert 1 <= ev.points <= 8
                total[ev.scorer] += ev.points
        assert total == g.scores, f"seed {seed}: events add up to {total} but the score is {g.scores}"


def test_clock_rules_are_obeyed_every_time():
    for seed in range(300):
        g, events = play_whole_game(seed)
        for ev in events:
            quarter = ev.state.quarter if ev.state is not None else None
            if ev.kind == "timeout" and quarter == 4 and ev.state.tempo != "normal":
                assert ev.state.clock_running and 0 < ev.state.clock <= 240 or ev.state.score_diff > 0, (seed, ev.text)
                if ev.detail == "defense":
                    assert ev.state.score_diff > 0 and rules.defense_timeout_probability(ev.state) > 0 \
                        or ev.state.score_diff <= 0, (seed, ev.text)
                else:
                    assert ev.state.score_diff <= 0, (seed, "an offense that is ahead stopped its own clock", ev.text)
            if ev.kind not in SCRIMMAGE_KINDS:
                continue
            if quarter == 4:
                action = rules.decide_before_snap(ev.state)
                if action == Action.KNEEL:
                    assert ev.kind == "kneel", (seed, ev.text)
                elif action == Action.FIELD_GOAL:
                    assert ev.kind == "field_goal", (seed, ev.text)
                elif action == Action.SPIKE:
                    assert ev.kind == "spike", (seed, ev.text)
                else:
                    assert ev.kind in ("play", "punt", "field_goal"), f"seed {seed}: {ev.kind} when no rule applied: {ev.text}"
            elif quarter == 2:
                if ev.kind == "kneel":
                    assert half.can_kneel(ev.state) and ev.state.clock <= half.SNAP_SECONDS, (seed, ev.text)
                if ev.kind == "spike":
                    assert ev.state.clock_running and ev.state.clock <= half.SNAP_SECONDS, (seed, ev.text)
                if ev.kind == "field_goal":
                    assert rules.kick_distance(ev.state) <= 66, (seed, ev.text)


def test_never_kneel_then_run_then_kneel():
    kneels = 0
    for seed in range(400):
        _, events = play_whole_game(seed)
        kneeling = None
        for ev in events:
            if ev.kind in ("kickoff", "punt", "field_goal", "try", "quarter_end", "halftime"):
                kneeling = None
            elif ev.kind == "kneel":
                kneeling = ev.team
                kneels += 1
            elif ev.kind == "play":
                pinned = ev.state.yardline_100 + rules.KNEEL_YARDS_LOST >= 100
                fourth = ev.state.down == 4
                assert ev.team != kneeling or pinned or fourth, f"seed {seed}: ran a play after kneeling: {ev.text}"
    assert kneels > 300, f"only saw {kneels} kneels in 400 games, real games have about 1.5 per game"


def test_two_minute_warning_and_halftime():
    g = Game(tables(), random.Random(0))
    g.quarter, g.clock = 2, 135
    assert g._tick(20, True) is True and g.clock == 120, "a running clock stops at 2:00"
    g.clock = 135
    assert g._tick(20, False) is False and g.clock == 115, "a stopped clock is not stopped again"
    g.quarter, g.clock = 1, 135
    assert g._tick(20, True) is False and g.clock == 115, "no warning in the 1st quarter"
    g.quarter, g.clock = 4, 121
    assert g._tick(1, True) is True and g.clock == 120

    first_kickers = set()
    for seed in range(200):
        g = Game(tables(), random.Random(seed))
        opening_receiver = g.offense
        warnings = {2: 0, 4: 0, 5: 0}
        halftime_seen = False
        while not g.over or g._queue:
            ev = g.step()
            if ev.kind == "two_minute_warning":
                assert g.clock == 120 and g.quarter in (2, 4, 5) and not g.clock_running
                warnings[g.quarter] += 1
            if ev.kind == "halftime":
                halftime_seen = True
                assert g.timeouts == [3, 3], "timeouts reset at halftime"
                assert g.kicker == opening_receiver, "the other team kicks off the second half"
        assert halftime_seen and warnings[2] <= 1 and warnings[4] <= 1 and warnings[5] <= 1, (seed, warnings)
        first_kickers.add(opening_receiver)
    assert first_kickers == {0, 1}, "both teams should get the opening kickoff sometimes"


def test_same_seed_same_game():
    _, a = play_whole_game(77)
    _, b = play_whole_game(77)
    assert [e.text for e in a] == [e.text for e in b]
    _, c = play_whole_game(78)
    assert [e.text for e in a] != [e.text for e in c]


def scenario(quarter, clock, down, distance, yardline_100, off_score, def_score, off_to=0, def_to=0,
             running=True, seed=0) -> Game:
    g = Game(tables(), random.Random(seed))
    g.quarter, g.clock, g.down, g.distance, g.yardline_100 = quarter, clock, down, distance, yardline_100
    g.offense, g.kicker, g.phase = 0, 1, "scrimmage"
    g.scores = [off_score, def_score]
    g.timeouts = [off_to, def_to]
    g.clock_running, g.runoff = running, 0
    return g


def run_until(g: Game, stop):
    events = []
    while not g.over or g._queue:
        ev = g.step()
        events.append(ev)
        if stop(ev, g):
            break
    return events


def test_leading_late_means_kneels_not_plays():
    for seed in range(50):
        g = scenario(4, 70, 1, 10, 70, 24, 17, off_to=1, def_to=0, running=False, seed=seed)
        events = run_until(g, lambda ev, game: game.over)
        offense_events = [e for e in events if e.kind in SCRIMMAGE_KINDS and e.team == 0]
        assert [e.kind for e in offense_events] == ["kneel", "kneel"] or all(e.kind == "kneel" for e in offense_events), \
            [e.text for e in offense_events]
        assert g.scores == [24, 17]


def test_the_thirty_second_drill():
    for seed in range(30):
        g = scenario(4, 9, 1, 10, 35, 10, 12, off_to=0, def_to=0, running=True, seed=seed)
        e1, e2 = g.step(), g.step()
        assert (e1.kind, e2.kind) == ("spike", "field_goal"), (e1.text, e2.text)
        assert "spikes" in e1.text and "field goal" in e2.text


def test_timeout_when_trailing_late():
    n, called = 400, 0
    for seed in range(n):
        g = scenario(4, 15, 1, 10, 50, 7, 10, off_to=1, def_to=0, running=True, seed=seed)
        ev = g.step()
        assert ev.kind != "time", (seed, "the trailing offense let the clock run out", ev.text)
        if ev.kind == "timeout":
            called += 1
            assert g.timeouts[0] == 0 and not g.clock_running and g.clock <= 15
    assert 0.35 < called / n < 0.75, called / n


def test_defense_sometimes_stops_the_clock_after_a_kneel():
    n, used = 600, 0
    for seed in range(n):
        g = scenario(4, 60, 1, 10, 70, 24, 17, off_to=0, def_to=1, running=False, seed=seed)
        ev = g.step()
        if ev.kind == "kneel":
            if g.timeouts[1] == 0:
                used += 1
                assert not g.clock_running
            else:
                assert g.timeouts[1] == 1
    rate = used / n
    assert 0.15 < rate < 0.38, f"defense timeout after a kneel: {rate:.2f}, real is about 0.25"


def test_games_look_like_football():
    n = 300
    totals = {"points": 0, "plays": 0, "punts": 0, "fg_att": 0, "turnovers": 0, "td": 0, "kneel": 0, "sacks": 0}
    margins = []
    for seed in range(n):
        g, _ = play_whole_game(1000 + seed)
        margins.append(abs(g.scores[0] - g.scores[1]))
        for s in g.stats:
            totals["points"] += 0
            totals["plays"] += s["plays"]
            totals["punts"] += s["punt"]
            totals["fg_att"] += s["fg_att"]
            totals["turnovers"] += s["interceptions_thrown"] + s["fumbles_lost"]
            totals["td"] += s["td"]
            totals["kneel"] += s["kneel"]
            totals["sacks"] += s["sack"]
        totals["points"] += sum(g.scores)
    per_game = {k: v / n for k, v in totals.items()}
    print("  per game, both teams combined:", {k: round(v, 1) for k, v in per_game.items()})
    assert 30 < per_game["points"] < 65, per_game
    assert 105 < per_game["plays"] < 165, per_game
    assert 5 < per_game["punts"] < 14, per_game
    assert 2 < per_game["fg_att"] < 9, per_game
    assert 1 < per_game["turnovers"] < 5, per_game
    assert 2 < per_game["sacks"] < 8, per_game
    assert 0.5 < per_game["kneel"] < 3, per_game
    assert max(margins) > 20 and min(margins) <= 3, "games should range from close to blowouts"


def test_free_kick_after_a_safety():
    safeties = 0
    starts = []
    between = {"quarter_end", "halftime", "two_minute_warning", "end_regulation", "overtime", "game_end"}
    for seed in range(1500):
        g = Game(tables(), random.Random(seed))
        tackled = None
        while not g.over or g._queue:
            e = g.step()
            if tackled is not None:
                if e.kind in between:
                    if e.kind in ("game_end", "end_regulation"):
                        tackled = None
                    continue
                assert e.kind == "kickoff", e.kind
                assert "free kick after the safety" in e.text and "kicks off" not in e.text, e.text
                assert e.team == tackled, "the team that was tackled in its end zone kicks"
                if not e.detail.touchdown:
                    starts.append(e.detail.yardline_100)
                    assert g.offense == 1 - tackled, "the other team gets the ball"
                safeties += 1
                tackled = None
            elif e.kind == "play" and e.detail.safety:
                tackled = e.team
    assert safeties >= 40, safeties
    assert 55 < sum(starts) / len(starts) < 68, sum(starts) / len(starts)


def test_no_onside_choice_on_a_free_kick():
    g = Game(tables(), random.Random(3))
    g.phase, g.kicker, g.quarter, g.clock, g.scores, g.free_kick = "kickoff", 0, 4, 100, [14, 21], True
    assert g.step().kind == "kickoff" and not g.free_kick


if __name__ == "__main__":
    test_free_kick_after_a_safety()
    test_no_onside_choice_on_a_free_kick()
    test_two_hundred_full_games()
    test_clock_rules_are_obeyed_every_time()
    test_never_kneel_then_run_then_kneel()
    test_two_minute_warning_and_halftime()
    test_same_seed_same_game()
    test_leading_late_means_kneels_not_plays()
    test_the_thirty_second_drill()
    test_timeout_when_trailing_late()
    test_defense_sometimes_stops_the_clock_after_a_kneel()
    test_games_look_like_football()
    print("All game tests passed")
