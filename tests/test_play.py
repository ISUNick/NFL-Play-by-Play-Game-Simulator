import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.play import run_scrimmage_play
from engine.state import GameState
from test_engine_tables import tables

KINDS = {"run", "scramble", "sack", "complete", "incomplete", "interception"}
TIMING = {"run_in", "run_oob", "run_td", "run_fumble", "scramble_in", "scramble_oob", "sack",
          "complete_in", "complete_oob", "pass_td", "incomplete", "interception"}


def random_state(rng) -> GameState:
    return GameState(quarter=rng.randint(1, 4), clock=rng.randint(0, 900), down=rng.randint(1, 4),
                     distance=rng.randint(1, 25), yardline_100=rng.randint(1, 99),
                     offense_score=rng.randint(0, 40), defense_score=rng.randint(0, 40))


def check_outcome(s: GameState, o) -> None:
    yl = s.yardline_100
    assert o.kind in KINDS and o.timing_kind in TIMING, o
    assert sum([o.touchdown, o.safety, o.turnover and not o.defensive_td, o.defensive_td]) <= 1, o
    if o.touchdown:
        assert o.new_yardline_100 == 0 and o.yards == yl and o.first_down and not o.turnover, (s, o)
        assert o.kind in ("run", "scramble", "complete") and o.timing_kind in ("run_td", "pass_td"), o
    elif o.safety:
        assert o.new_yardline_100 == 0 and o.yards <= -(100 - yl) and not o.turnover, (s, o)
    elif o.defensive_td:
        assert o.kind == "interception" and o.new_yardline_100 == 0 and o.turnover, (s, o)
    elif o.turnover:
        assert 1 <= o.new_yardline_100 <= 99 and not o.first_down, (s, o)
        assert o.kind == "interception" or o.fumble_lost, o
    else:
        assert 1 <= o.new_yardline_100 <= 99, (s, o)
        assert o.new_yardline_100 == yl - o.yards, f"the ball is not where the yards say: {s} {o}"
        assert o.first_down == (o.yards >= s.distance), (s, o)
    if o.kind == "incomplete":
        assert o.yards == 0 and o.new_yardline_100 == yl and o.out_of_bounds, o
    if o.kind == "sack":
        assert o.yards <= 0 or o.touchdown, o
    if o.depth == "deep":
        assert yl >= 20, f"a deep throw from inside the 20: {s}"
    if o.depth == "medium":
        assert yl >= 10, f"a medium throw from inside the 10: {s}"
    if o.kind != "complete" and o.kind != "incomplete" and o.kind != "interception":
        assert o.depth is None


def test_one_hundred_thousand_random_plays():
    t = tables()
    rng = random.Random(21)
    seen = set()
    for _ in range(100000):
        s = random_state(rng)
        o = run_scrimmage_play(s, t, rng)
        check_outcome(s, o)
        seen.add(o.kind)
        if o.touchdown:
            seen.add("td")
        if o.safety:
            seen.add("safety")
        if o.fumble_lost:
            seen.add("fumble")
        if o.defensive_td:
            seen.add("pick six")
    assert seen == KINDS | {"td", "safety", "fumble", "pick six"}, f"never saw: {(KINDS | {'td', 'safety', 'fumble', 'pick six'}) - seen}"


def test_red_zone_is_not_inflated():
    t = tables()
    rng = random.Random(22)
    s = GameState(down=1, distance=4, yardline_100=4)
    plays = [run_scrimmage_play(s, t, rng) for _ in range(30000)]
    assert max(p.yards for p in plays) <= 4
    td = sum(p.touchdown for p in plays) / len(plays)
    assert 0.15 < td < 0.45, f"touchdown rate per play from the 4 yard line: {td:.1%}"


def test_rates_follow_the_tables():
    t = tables()
    rng = random.Random(23)
    s = GameState(down=1, distance=10, yardline_100=75, offense_score=10, defense_score=10, quarter=2, clock=600)
    n = 40000
    plays = [run_scrimmage_play(s, t, rng) for _ in range(n)]
    _, call = t.lookup("play_call", s.situation())
    _, split = t.lookup("dropback_split", s.situation())
    dropbacks = [p for p in plays if p.kind != "run"]
    assert abs(len(dropbacks) / n - call["p_dropback"]) < 0.012
    assert abs(sum(p.kind == "sack" for p in dropbacks) / len(dropbacks) - split["p_sack"]) < 0.01
    throws = [p for p in dropbacks if p.kind in ("complete", "incomplete", "interception")]
    comp = sum(p.kind == "complete" for p in throws) / len(throws)
    assert 0.55 < comp < 0.72, f"completion rate {comp:.1%}"
    yards_per_play = sum(p.yards for p in plays) / n
    assert 4.5 < yards_per_play < 6.5, f"yards per play {yards_per_play:.2f}"


def test_third_and_long_looks_like_third_and_long():
    t = tables()
    rng = random.Random(24)
    s = GameState(down=3, distance=12, yardline_100=60)
    plays = [run_scrimmage_play(s, t, rng) for _ in range(20000)]
    dropbacks = sum(p.kind != "run" for p in plays) / len(plays)
    assert dropbacks > 0.9, f"3rd and 12 should be nearly all passes, got {dropbacks:.1%}"
    sacks = sum(p.kind == "sack" for p in plays) / len(plays)
    assert sacks > 0.07, f"3rd and long should have lots of sacks, got {sacks:.1%}"


def test_late_game_play_calling():
    t = tables()
    rng = random.Random(25)

    def run_share(clock, offense_score, defense_score, n=6000):
        s = GameState(quarter=4, clock=clock, down=1, distance=10, yardline_100=55,
                      offense_score=offense_score, defense_score=defense_score)
        return sum(run_scrimmage_play(s, t, rng).kind == "run" for _ in range(n)) / n
    trailing_9 = run_share(175, 18, 27)
    assert trailing_9 < 0.12, f"down 9 with 2:55 left should almost always pass, ran {trailing_9:.1%}"
    leading_big = run_share(100, 34, 14)
    assert leading_big > 0.80, f"up 20 with 1:40 left should almost always run, ran {leading_big:.1%}"
    trailing_6 = run_share(90, 20, 26)
    assert trailing_6 < 0.18, f"down 6 with 1:30 left should almost always pass, ran {trailing_6:.1%}"
    early = run_share(800, 20, 20)
    assert 0.3 < early < 0.6, f"a tied first down in the 4th quarter with 13 minutes left is normal, ran {early:.1%}"


def test_a_trailing_offense_with_seconds_left_throws():
    t, rng = tables(), random.Random(11)
    desperate = GameState(quarter=4, clock=3, down=2, distance=3, yardline_100=41, offense_score=23, defense_score=27)
    runs = sum(run_scrimmage_play(desperate, t, rng).kind in ("run",) for _ in range(400))
    assert runs < 20, runs
    normal = GameState(quarter=3, clock=500, down=2, distance=3, yardline_100=41, offense_score=23, defense_score=27)
    assert sum(run_scrimmage_play(normal, t, rng).kind == "run" for _ in range(400)) > 80


def test_a_team_in_range_kicks_at_the_end_of_the_half():
    from engine.special import FIELD_GOAL, choose_fourth_down
    t, rng = tables(), random.Random(5)
    state = GameState(quarter=2, clock=1, down=4, distance=1, yardline_100=27, offense_score=10, defense_score=9)
    kicks = sum(choose_fourth_down(state, t, rng) == FIELD_GOAL for _ in range(400))
    assert kicks > 370, kicks
    far = GameState(quarter=2, clock=1, down=4, distance=1, yardline_100=48, offense_score=10, defense_score=9)
    assert sum(choose_fourth_down(far, t, rng) == FIELD_GOAL for _ in range(400)) < 380


if __name__ == "__main__":
    test_one_hundred_thousand_random_plays()
    test_red_zone_is_not_inflated()
    test_rates_follow_the_tables()
    test_third_and_long_looks_like_third_and_long()
    test_late_game_play_calling()
    test_a_trailing_offense_with_seconds_left_throws()
    test_a_team_in_range_kicks_at_the_end_of_the_half()
    print("All play tests passed")
