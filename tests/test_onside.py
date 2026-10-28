import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.game import Game
from engine.special import decide_onside, resolve_onside
from test_engine_tables import tables

SCORE_BUCKETS = ["trail_17p", "trail_9_16", "trail_4_8", "trail_1_3"]
LEADS = {"trail_1_3": -2, "trail_4_8": -6, "trail_9_16": -12, "trail_17p": -24}


def rate(lead, quarter, seconds, n=3000, seed=1) -> float:
    t, rng = tables(), random.Random(seed)
    return sum(decide_onside(lead, quarter, seconds, t, rng) for _ in range(n)) / n


def test_table_is_valid():
    t = tables()
    rows = t._index["onside_choice"]
    assert rows
    for key, row in rows.items():
        assert 0 <= row["p_onside"] <= 1 and row["n"] >= 30, (key, row)
    for score in SCORE_BUCKETS:
        for kwin in ("k120", "k300", "k600", "k900"):
            assert t.lookup_or_none("onside_choice", dict(late_score=score, kwin=kwin)) is not None, (score, kwin)


def test_decisions_follow_real_teams():
    for score in ("trail_1_3", "trail_4_8", "trail_9_16"):
        assert rate(LEADS[score], 4, 100) > 0.75, score
    assert rate(-3, 4, 800) < 0.03
    assert rate(-6, 4, 800) < 0.03
    assert rate(-2, 4, 900) < 0.03
    for lead in (0, 1, 7, 21):
        assert rate(lead, 4, 100, n=500) == 0.0
    for quarter in (1, 2, 3):
        assert rate(-6, quarter, 100, n=500) == 0.0
    assert rate(-6, 4, 0, n=500) == 0.0


def test_resolve_onside_is_legal_and_matches_real_recovery():
    t, rng = tables(), random.Random(3)
    n, rec = 20000, 0
    for _ in range(n):
        r = resolve_onside(t, rng)
        assert 1 <= r.yardline_100 <= 99, r
        rec += r.kicking_team_recovers
    assert 0.05 < rec / n < 0.10, f"recovery {rec / n:.3f}, real is 0.073"


def test_game_handles_the_kick():
    t = tables()
    seen = {"recovered": 0, "lost": 0}
    for seed in range(400):
        g = Game(t, random.Random(seed))
        g.quarter, g.clock, g.phase = 4, 90, "kickoff"
        g.kicker, g.offense, g.scores = 0, 1, [14, 24]
        g.timeouts, g.clock_running, g.try_team, g.kneeling_team = [2, 2], False, None, None
        before = [dict(s) for s in g.stats]
        ev = g.step()
        if ev.kind != "onside":
            assert ev.kind == "kickoff"
            continue
        assert ev.team == 0 and g.phase == "scrimmage" and g.down == 1 and g.distance == min(10, g.yardline_100)
        assert 1 <= g.yardline_100 <= 99 and not g.clock_running and g.fresh
        assert g.stats[0]["onside_att"] == before[0].get("onside_att", 0) + 1
        assert g.clock <= 90 and g.scores == [14, 24]
        if g.offense == 0:
            seen["recovered"] += 1
            assert g.stats[0]["onside_rec"] == before[0].get("onside_rec", 0) + 1
        else:
            seen["lost"] += 1
            assert g.stats[0]["onside_rec"] == before[0].get("onside_rec", 0)
    assert seen["lost"] > 200 and seen["recovered"] >= 1, seen


def test_a_team_that_wins_does_not_onside_kick():
    t = tables()
    for seed in range(300):
        g = Game(t, random.Random(seed))
        g.quarter, g.clock, g.phase = 4, 90, "kickoff"
        g.kicker, g.offense, g.scores = 0, 1, [24, 14]
        g.timeouts, g.clock_running, g.try_team, g.kneeling_team = [2, 2], False, None, None
        assert g.step().kind == "kickoff"


def test_full_games_have_realistic_onside_numbers():
    t = tables()
    n, attempts, rec = 800, 0, 0
    for seed in range(n):
        g = Game(t, random.Random(9000 + seed))
        while not g.over or g._queue:
            ev = g.step()
            if ev.kind == "onside":
                st = ev.state
                assert st.quarter == 4, ev.text
                assert st.score_diff > 0, ("kicker is the team that is behind", ev.text)
                attempts += 1
        rec += sum(s["onside_rec"] for s in g.stats)
    per_game = attempts / n
    print(f"  onside attempts per game {per_game:.3f}, recovered {rec}/{attempts}")
    assert 0.08 < per_game < 0.35, per_game
    assert attempts < 20 or rec / attempts < 0.2


if __name__ == "__main__":
    test_table_is_valid()
    test_decisions_follow_real_teams()
    test_resolve_onside_is_legal_and_matches_real_recovery()
    test_game_handles_the_kick()
    test_a_team_that_wins_does_not_onside_kick()
    test_full_games_have_realistic_onside_numbers()
    print("All onside tests passed")
