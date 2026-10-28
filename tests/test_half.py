import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine import clock_rules as rules
from engine import half
from engine.clock_rules import Action, defense_kneel_timeout_probability
from engine.game import Game
from engine.state import GameState
from test_engine_tables import tables

SNAP_WINDOWS = ["s3", "s6", "s10", "s14", "s20", "s30", "s45", "s60", "s90", "s120"]
TIMEOUT_WINDOWS = SNAP_WINDOWS + ["s180", "s240"]
ZONES = ["h_fg", "h_mid", "h_own", "h_deep"]
SCRIMMAGE_KINDS = {"play", "punt", "field_goal", "kneel", "spike"}


def q2(clock, down=1, distance=10, yl=75, off_to=3, def_to=3, running=False, new=False,
       off_score=10, def_score=10) -> GameState:
    return GameState(quarter=2, clock=clock, down=down, distance=distance, yardline_100=yl,
                     offense_score=off_score, defense_score=def_score, offense_timeouts=off_to,
                     defense_timeouts=def_to, clock_running=running, new_possession=new)


def test_tables_are_valid_and_cover_every_situation():
    t = tables()
    for name in ("half_snap", "half_timeout_off", "half_timeout_def"):
        assert name in t._index, name
    for (level, key), row in t._index["half_snap"].items():
        total = row["p_kneel"] + row["p_spike"] + row["p_fg"]
        assert all(0 <= row[k] <= 1 for k in ("p_kneel", "p_spike", "p_fg")) and total <= 1 + 1e-9, (level, key, row)
        assert row["n"] >= 60, (level, key, row["n"])
    for name in ("half_timeout_off", "half_timeout_def"):
        for (level, key), row in t._index[name].items():
            assert 0 <= row["p_timeout"] <= 1 and row["n"] >= 40, (name, level, key, row)

    for w in SNAP_WINDOWS:
        for z in ZONES:
            for rs in ("newposs", "running", "stopped"):
                for has_to in (0, 1):
                    for dto in (0, 1, 2):
                        for down in (1, 2, 3):
                            sit = dict(hwin=w, rs=rs, hzone=z, dto=dto, has_to=has_to, hdown=down)
                            assert t.lookup_or_none("half_snap", sit) is not None, sit
    for name in ("half_timeout_off", "half_timeout_def"):
        for w in TIMEOUT_WINDOWS:
            for z in ZONES:
                for down in (1, 2, 3, 4):
                    assert t.lookup_or_none(name, dict(hwin=w, hzone=z, hdown=down)) is not None, (name, w, z, down)


def test_tables_make_football_sense():
    t = tables()
    rows = list(t._index["half_snap"].items())
    assert max(r["p_kneel"] for (lv, k), r in rows if "h_fg" in k or "h_mid" in k) <= 0.06
    assert max((r["p_fg"] for (lv, k), r in rows if "h_own" in k or "h_deep" in k), default=0) == 0
    assert max((r["p_spike"] for (lv, k), r in rows if "newposs" in k), default=0) == 0
    _, r = t.lookup("half_snap", dict(hwin="s10", rs="newposs", hzone="h_deep", dto=0, has_to=1, hdown=1))
    assert r["p_kneel"] > 0.6, r
    _, r = t.lookup("half_snap", dict(hwin="s90", rs="newposs", hzone="h_deep", dto=0, has_to=1, hdown=1))
    assert r["p_kneel"] < 0.03, r
    _, none = t.lookup("half_snap", dict(hwin="s30", rs="newposs", hzone="h_own", dto=0, has_to=1, hdown=1))
    _, two = t.lookup("half_snap", dict(hwin="s30", rs="newposs", hzone="h_own", dto=2, has_to=1, hdown=1))
    assert none["p_kneel"] > two["p_kneel"], (none, two)
    _, r = t.lookup("half_snap", dict(hwin="s3", rs="stopped", hzone="h_fg", dto=1, has_to=1, hdown=2))
    assert r["p_fg"] > 0.8, r
    _, r = t.lookup("half_snap", dict(hwin="s10", rs="running", hzone="h_fg", dto=1, has_to=0, hdown=1))
    assert r["p_spike"] > 0.3, r
    for name in ("half_timeout_off", "half_timeout_def"):
        _, late = t.lookup(name, dict(hwin="s14", hzone="h_fg", hdown=2))
        _, early = t.lookup(name, dict(hwin="s240", hzone="h_fg", hdown=2))
        assert late["p_timeout"] > 0.1 > early["p_timeout"], (name, late, early)
    _, off = t.lookup("half_timeout_off", dict(hwin="s14", hzone="h_fg", hdown=2))
    assert off["p_timeout"] > 0.4, off


def share(state, action, tables_, rng, kneeling=False, n=4000) -> float:
    return sum(half.decide_before_snap(state, tables_, rng, kneeling) == action for _ in range(n)) / n


def test_decisions_follow_the_data():
    t, rng = tables(), random.Random(31)
    assert share(q2(6, yl=75, new=True, def_to=0), Action.KNEEL, t, rng) > 0.6
    assert share(q2(70, yl=75, new=True, def_to=0), Action.KNEEL, t, rng) < 0.03
    assert share(q2(2, down=3, distance=7, yl=25), Action.FIELD_GOAL, t, rng) > 0.8
    assert share(q2(8, yl=25, running=True, off_to=0), Action.SPIKE, t, rng, n=500) == 0.0
    assert share(q2(300), Action.NORMAL, t, rng) == 1.0
    assert share(q2(125, yl=30), Action.NORMAL, t, rng) == 1.0
    for quarter in (1, 3, 4):
        s = GameState(quarter=quarter, clock=6, down=1, distance=10, yardline_100=75, new_possession=True)
        assert share(s, Action.NORMAL, t, rng, n=500) == 1.0


def test_guards_never_break():
    t, rng = tables(), random.Random(32)
    seen = {Action.KNEEL: 0, Action.SPIKE: 0, Action.FIELD_GOAL: 0, Action.NORMAL: 0}
    for _ in range(80000):
        s = q2(rng.randint(1, 140), down=rng.randint(1, 4), distance=rng.randint(1, 20), yl=rng.randint(1, 99),
               off_to=rng.randint(0, 3), def_to=rng.randint(0, 3), running=rng.random() < 0.5,
               new=rng.random() < 0.3)
        kneeling = rng.random() < 0.3
        action = half.decide_before_snap(s, t, rng, kneeling)
        seen[action] += 1
        in_window = s.clock <= half.SNAP_SECONDS
        if not in_window or s.down == 4:
            assert action == Action.NORMAL, (s, action)
        if action == Action.KNEEL:
            assert half.can_kneel(s) and s.yardline_100 >= 56 and s.down <= 3, s
            assert s.yardline_100 + rules.KNEEL_YARDS_LOST < 100, s
        if action == Action.SPIKE:
            assert s.clock_running and s.down <= 3, s
        if action == Action.FIELD_GOAL:
            assert rules.kick_distance(s) <= 66 and s.down <= 3, s
        if kneeling and in_window and s.down <= 3 and half.can_kneel(s):
            assert action == Action.KNEEL, ("a kneeling team must keep kneeling", s, action)
    assert seen[Action.SPIKE] == 0, "spikes are decided right after the play now (offense_spike)"
    assert all(v > 500 for k, v in seen.items() if k != Action.SPIKE), seen
    for _ in range(20000):
        s = q2(rng.randint(1, 140), down=rng.randint(1, 4), distance=rng.randint(1, 20), yl=rng.randint(1, 99),
               off_to=rng.randint(0, 3), def_to=rng.randint(0, 3), running=rng.random() < 0.5)
        w = half.offense_spike(s, t, rng)
        if w is not None:
            assert s.clock_running and s.offense_timeouts == 0 and s.down <= 3 and 0 <= w <= s.clock, (s, w)


def test_timeout_decisions():
    t, rng = tables(), random.Random(33)
    s = q2(12, down=2, distance=8, yl=30, running=True)
    off = sum(half.offense_timeout(s, t, rng) for _ in range(4000)) / 4000
    assert 0.4 < off < 0.9, f"offense timeout chance with 12 seconds left near the goal: {off:.2f}"
    assert not any(half.defense_timeout(s, t, rng) for _ in range(1000))
    s4 = q2(12, down=4, distance=3, yl=30, running=True)
    assert any(half.defense_timeout(s4, t, rng) for _ in range(2000))
    dfn = sum(half.defense_timeout(q2(40, down=2, distance=8, yl=50, running=True), t, rng)
              for _ in range(4000)) / 4000
    assert 0.03 < dfn < 0.6, f"defense timeout chance with 40 seconds left: {dfn:.2f}"
    for bad in (q2(12, down=2, yl=30, running=False), q2(12, down=2, yl=30, running=True, off_to=0),
                q2(300, down=2, yl=30, running=True),
                GameState(quarter=4, clock=12, down=2, yardline_100=30, clock_running=True),
                GameState(quarter=1, clock=12, down=2, yardline_100=30, clock_running=True)):
        assert not any(half.offense_timeout(bad, t, rng) for _ in range(500)), bad
    assert not any(half.defense_timeout(q2(12, down=2, yl=30, running=True, def_to=0), t, rng) for _ in range(500))
    assert not any(half.defense_timeout(q2(12, down=2, yl=30, running=False), t, rng) for _ in range(500))


def test_offense_spike():
    t, rng = tables(), random.Random(34)
    s = q2(25, down=1, distance=10, yl=40, off_to=0, running=True)
    waits = [half.offense_spike(s, t, rng) for _ in range(4000)]
    share_spiking = sum(w is not None for w in waits) / len(waits)
    assert 0.3 < share_spiking < 0.95, share_spiking
    assert all(0 <= w <= s.clock for w in waits if w is not None)
    for bad in (q2(25, yl=40, off_to=1, running=True), q2(25, down=4, yl=40, off_to=0, running=True),
                q2(25, yl=40, off_to=0, running=False), q2(300, yl=40, off_to=0, running=True),
                GameState(quarter=4, clock=25, down=1, yardline_100=40, offense_timeouts=0, clock_running=True)):
        assert all(half.offense_spike(bad, t, rng) is None for _ in range(500)), bad


def test_defense_timeout_after_a_kneel_is_measured_not_automatic():
    def q4(lead, to=2):
        return GameState(quarter=4, clock=90, down=1, yardline_100=70, offense_score=20 + lead, defense_score=20,
                         offense_timeouts=3, defense_timeouts=to, clock_running=True)
    assert defense_kneel_timeout_probability(q4(2)) == 0.207
    assert defense_kneel_timeout_probability(q4(6)) == 0.254
    assert defense_kneel_timeout_probability(q4(12)) == 0.076
    assert defense_kneel_timeout_probability(q4(21)) == 0.008
    assert defense_kneel_timeout_probability(q4(6, to=0)) == 0.0
    assert defense_kneel_timeout_probability(q4(0)) == 0.0 and defense_kneel_timeout_probability(q4(-3)) == 0.0
    assert defense_kneel_timeout_probability(q2(20, yl=75)) == 0.014
    assert defense_kneel_timeout_probability(q2(50, yl=75)) == 0.07
    assert defense_kneel_timeout_probability(q2(20, yl=75, def_to=0)) == 0.0


def half_scenario(clock, down, distance, yl, off_to=2, def_to=2, running=False, new=False, seed=0,
                  scores=(10, 10)) -> Game:
    g = Game(tables(), random.Random(seed))
    g.quarter, g.clock, g.down, g.distance, g.yardline_100 = 2, clock, down, distance, yl
    g.offense, g.kicker, g.phase = 0, 1, "scrimmage"
    g.second_half_kicker = 0
    g.scores, g.timeouts = list(scores), [off_to, def_to]
    g.clock_running, g.runoff, g.fresh = running, 0, new
    return g


def run_to_halftime(g: Game):
    events = []
    while True:
        ev = g.step()
        events.append(ev)
        if ev.kind == "halftime":
            return events
        assert not g.over and len(events) < 200, "the half never ended"


def test_a_team_that_starts_kneeling_keeps_kneeling():
    kneel_games = 0
    for seed in range(300):
        g = half_scenario(30, 1, 10, 75, off_to=0, def_to=0, new=True, seed=seed)
        events = run_to_halftime(g)
        mine = [e for e in events if e.kind in SCRIMMAGE_KINDS and e.team == 0]
        kinds = [e.kind for e in mine]
        if "kneel" in kinds:
            kneel_games += 1
            first = kinds.index("kneel")
            assert all(k == "kneel" for k in kinds[first:]), (seed, [e.text for e in mine])
    assert kneel_games > 40, kneel_games


def test_the_half_ends_cleanly():
    for seed in range(300):
        g = half_scenario(5 + seed % 40, 1 + seed % 3, 10, 20 + seed % 60, off_to=seed % 4, def_to=(seed // 4) % 4,
                          running=seed % 2 == 0, new=seed % 5 == 0, seed=seed)
        events = run_to_halftime(g)
        assert events[-1].kind == "halftime" and g.quarter == 3 and g.clock == 900
        assert g.timeouts == [3, 3], "timeouts reset for the second half"
        assert g.kicker == g.second_half_kicker and g.phase == "kickoff"
        assert not g.clock_running and g.try_team is None
        nxt = g.step()
        assert nxt.kind == "kickoff", nxt.text
        for e in events:
            assert e.state.quarter == 2


def test_new_possession_flag_follows_the_drives():
    for seed in range(60):
        g = Game(tables(), random.Random(seed))
        events = []
        while not g.over or g._queue:
            events.append(g.step())
        previous = None
        for e in events:
            if e.kind in SCRIMMAGE_KINDS:
                if previous is not None and previous.kind in ("kickoff", "onside"):
                    assert e.state.new_possession, (seed, previous.text, e.text)
                if previous is not None and previous.kind in SCRIMMAGE_KINDS and previous.team == e.team \
                        and previous.kind in ("play", "kneel", "spike") and not _ball_changed(previous):
                    assert not e.state.new_possession, (seed, previous.text, e.text)
            if e.kind not in ("timeout", "two_minute_warning"):
                previous = e


def _ball_changed(ev) -> bool:
    out = ev.detail
    if ev.kind != "play":
        return False
    return bool(out.turnover or out.defensive_td or out.touchdown or out.safety) or (
        ev.state.down == 4 and not out.first_down)


def test_first_half_numbers_look_like_real_football():
    n = 500
    c = {"kneel": 0, "spike": 0, "off_to": 0, "def_to": 0, "fg_last10": 0}
    for seed in range(n):
        g = Game(tables(), random.Random(7000 + seed))
        while not g.over or g._queue:
            e = g.step()
            st = e.state
            if st is None or st.quarter != 2 or not 0 < st.clock <= 240:
                continue
            if e.kind in ("kneel", "spike"):
                c[e.kind] += 1
            elif e.kind == "field_goal" and st.clock <= 10:
                c["fg_last10"] += 1
            elif e.kind == "timeout":
                c["def_to" if e.detail == "defense" else "off_to"] += 1
    per = {k: v / n for k, v in c.items()}
    print("  first half, last 4 minutes, per game:", {k: round(v, 2) for k, v in per.items()})
    assert 0.2 < per["kneel"] < 0.5, per
    assert 0.05 < per["spike"] < 0.2, per
    assert 1.0 < per["off_to"] < 2.0, per
    assert 0.5 < per["def_to"] < 1.6, per
    assert 0.12 < per["fg_last10"] < 0.5, per


if __name__ == "__main__":
    test_tables_are_valid_and_cover_every_situation()
    test_tables_make_football_sense()
    test_decisions_follow_the_data()
    test_guards_never_break()
    test_timeout_decisions()
    test_offense_spike()
    test_defense_timeout_after_a_kneel_is_measured_not_automatic()
    test_a_team_that_starts_kneeling_keeps_kneeling()
    test_the_half_ends_cleanly()
    test_new_possession_flag_follows_the_drives()
    test_first_half_numbers_look_like_real_football()
    print("All end of half tests passed")
