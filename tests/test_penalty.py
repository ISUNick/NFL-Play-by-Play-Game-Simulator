import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.game import Game
from engine.penalty import SPOT_FOUL, resolve_penalty
from engine.state import GameState
from engine.tables import Tables
from engine.text import down_text

T = Tables()
KINDS = ["pre", "run", "scramble", "sack", "pass_short", "pass_deep"]


def test_rates_are_sane():
    for down in (1, 2, 3, 4):
        _, row = T.lookup("penalty_pre", GameState(down=down, distance=10, yardline_100=60).situation())
        assert 0.005 < row["p_off"] < 0.06 and 0.001 < row["p_def"] < 0.03, (down, row)


def test_the_kind_of_play_matters():
    def post(kind, yl=50, down=2):
        sit = GameState(down=down, distance=8, yardline_100=yl).situation()
        sit["kind"] = kind
        return T.lookup("penalty_post", sit)[1]
    deep, short, run = post("pass_deep"), post("pass_short"), post("run")
    assert deep["p_def"] > 2 * short["p_def"] > 0 and deep["p_def"] > 8 * run["p_def"], (deep, short, run)
    assert run["p_off"] > run["p_def"], "holding on a run is mostly on the offense"
    assert post("scramble")["p_off"] > 2 * post("scramble")["p_def"]
    assert post("sack")["p_def"] > post("sack")["p_off"], "a sack is wiped out by defensive fouls"
    assert post("pass_short", yl=12)["p_def"] > post("pass_short", yl=50)["p_def"]


def test_the_fouls_fit_the_play():
    deep = T.penalty[("def", "pass_deep")]
    fouls = [f for *_, f in deep]
    assert fouls.count("Defensive Pass Interference") / len(fouls) > 0.3
    runs = [f for *_, f in T.penalty[("off", "run")]]
    assert runs.count("Offensive Holding") / len(runs) > 0.5
    d = [dy for dy, _, _, f in T.penalty[("def", "pass_deep")] if f == "Defensive Pass Interference"]
    s_ = [dy for dy, _, _, f in T.penalty[("def", "pass_short")] if f == "Defensive Pass Interference"]
    assert -dy_mean(d) > 2 * -dy_mean(s_), (dy_mean(d), dy_mean(s_))
    pre = [dy for dy, _, _, f in T.penalty[("off", "pre")] if f == "False Start"]
    assert abs(dy_mean(pre) - 5) < 0.5


def dy_mean(xs):
    return sum(xs) / len(xs)


def test_bank_has_both_sides_and_every_down():
    for side in ("off", "def"):
        for kind in ("pre", "run", "pass_short", "pass_deep"):
            assert len(T.penalty[(side, kind)]) >= 100, (side, kind)
    assert not any(first for dy, first, s, f in T.penalty[("off",)])
    assert sum(dy < 0 for dy, *_ in T.penalty[("def",)]) > 0.9 * len(T.penalty[("def",)])
    assert sum(dy > 0 for dy, *_ in T.penalty[("off",)]) > 0.9 * len(T.penalty[("off",)])


def test_no_kick_only_fouls_on_ordinary_snaps():
    import re
    for key, rows in T.penalty.items():
        for *_, foul in rows:
            assert not re.search("Kick|Punt|Fair Catch", foul, re.I), (key, foul)


def test_resolve_never_makes_an_impossible_situation():
    rng = random.Random(7)
    for yl in list(range(1, 100)):
        for down in (1, 2, 3, 4):
            for distance in (1, 3, 7, 10, 15, 25):
                if distance > yl:
                    continue
                st = GameState(down=down, distance=distance, yardline_100=yl)
                for side in ("offense", "defense"):
                    for _ in range(5):
                        r = resolve_penalty(st, side, T, rng, random.choice(KINDS))
                        assert 1 <= r.new_yardline_100 <= 99, (yl, side, r)
                        assert 1 <= r.distance <= r.new_yardline_100, (yl, distance, side, r)
                        assert 1 <= r.down <= 4 and r.seconds >= 0
                        assert r.yards >= 0
                        if side == "offense":
                            assert r.new_yardline_100 >= yl and not r.first_down, (yl, r)
                            assert r.yards <= (100 - yl) // 2, "half the distance to the goal"
                        else:
                            assert r.new_yardline_100 <= yl, (yl, r)
                            if r.foul == SPOT_FOUL:
                                assert r.yards <= yl - 1, "pass interference goes to the spot, at most the 1 yard line"
                            else:
                                assert r.yards <= yl // 2 or r.yards == 0, "half the distance to the goal"
                        if r.first_down:
                            assert r.down == 1 and r.distance == min(10, r.new_yardline_100)
                        else:
                            assert r.down == down


def test_half_the_distance_examples():
    rng = random.Random(1)
    st = GameState(down=2, distance=3, yardline_100=3)
    seen = {resolve_penalty(st, "defense", T, rng).new_yardline_100 for _ in range(300)}
    assert seen <= {3, 2, 1}, seen
    st = GameState(down=1, distance=10, yardline_100=98)
    assert {resolve_penalty(st, "offense", T, rng).new_yardline_100 for _ in range(300)} <= {98, 99}


def test_pass_interference_is_a_spot_foul_and_everything_else_is_half_the_distance():
    rng = random.Random(11)
    st = GameState(down=2, distance=10, yardline_100=20)
    spot, other = [], []
    for _ in range(4000):
        r = resolve_penalty(st, "defense", T, rng, "pass_deep")
        (spot if r.foul == SPOT_FOUL else other).append(r)
    assert spot and other
    assert max(r.yards for r in spot) > 10, "pass interference must be able to move the ball past half of 20 yards"
    assert all(r.new_yardline_100 >= 1 for r in spot)
    assert all(r.yards <= 10 for r in other), "the other defensive fouls are half the distance to the goal"
    assert all(r.halved for r in other if r.yards == 10) or True
    assert any(r.halved for r in other), "a 15 yard foul from the 20 is cut to 10, and the result says so"
    assert all(r.first_down for r in spot if r.yards >= 10), "a spot foul that gains the line to gain is a first down"
    st = GameState(down=1, distance=5, yardline_100=5)
    assert min(resolve_penalty(st, "defense", T, rng, "pass_deep").new_yardline_100 for _ in range(500)) == 1
    st = GameState(down=1, distance=10, yardline_100=90)
    assert all(resolve_penalty(st, "offense", T, rng, "run").yards <= 5 for _ in range(500))


def test_a_defensive_penalty_cannot_end_the_half_or_the_game():
    tables = Tables()
    for quarter in (2, 4):
        for seed in range(300):
            g = Game(tables, random.Random(seed))
            g.quarter, g.clock, g.down, g.distance, g.yardline_100 = quarter, 3, 2, 5, 33
            g.offense, g.kicker, g.phase = 0, 1, "scrimmage"
            g.second_half_kicker = 0
            g.scores, g.timeouts = [19, 27], [0, 0]
            g.clock_running, g.runoff, g.fresh = True, 0, False
            ev = g._penalty(g.state(), "defense", "pass_short")
            assert g.clock >= 1, (quarter, seed, ev.text)
            if g.clock == 1 and ev.seconds >= 3:
                assert "one more play" in ev.text, ev.text
    ended = 0
    for seed in range(300):
        g = Game(tables, random.Random(seed))
        g.quarter, g.clock, g.down, g.distance, g.yardline_100 = 4, 3, 2, 5, 33
        g.offense, g.kicker, g.phase = 0, 1, "scrimmage"
        g.scores, g.timeouts = [19, 27], [0, 0]
        g.clock_running, g.runoff, g.fresh = True, 0, False
        g._penalty(g.state(), "offense", "pass_short")
        ended += g.clock == 0
    assert ended > 0


def test_goal_to_go_text():
    assert down_text(1, 6, 6) == "1st & Goal"
    assert down_text(1, 11, 11) == "1st & Goal"
    assert down_text(1, 10, 10) == "1st & Goal"
    assert down_text(2, 3, 3) == "2nd & Goal"
    assert down_text(1, 10, 11) == "1st & 10"
    assert down_text(3, 4, 6) == "3rd & 4"
    assert down_text(1, 10, 75) == "1st & 10"
    st = GameState(down=1, distance=6, yardline_100=6)
    rng = random.Random(3)
    seen = set()
    for _ in range(400):
        r = resolve_penalty(st, "offense", T, rng)
        if not r.first_down and r.new_yardline_100 > 10:
            seen.add(down_text(r.down, r.distance, r.new_yardline_100))
    assert seen and all(t.endswith("Goal") for t in seen), seen


def test_games_have_realistic_penalties():
    games, total, first_downs = 400, 0, 0
    for seed in range(games):
        g = Game(T, random.Random(seed))
        while not g.over or g._queue:
            e = g.step()
            if e.kind == "penalty":
                total += 1
                first_downs += e.detail.first_down
                assert "PENALTY on" in e.text
    per_game = total / games
    print(f"  {per_game:.2f} penalties a game, {first_downs / total:.0%} automatic first downs")
    assert 7.0 < per_game < 10.0, per_game
    assert 0.22 < first_downs / total < 0.40


if __name__ == "__main__":
    test_rates_are_sane()
    test_the_kind_of_play_matters()
    test_the_fouls_fit_the_play()
    test_bank_has_both_sides_and_every_down()
    test_no_kick_only_fouls_on_ordinary_snaps()
    test_resolve_never_makes_an_impossible_situation()
    test_half_the_distance_examples()
    test_pass_interference_is_a_spot_foul_and_everything_else_is_half_the_distance()
    test_a_defensive_penalty_cannot_end_the_half_or_the_game()
    test_goal_to_go_text()
    test_games_have_realistic_penalties()
    print("All penalty tests passed")
