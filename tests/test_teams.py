import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.game import Game
from engine.play import run_scrimmage_play
from engine.state import GameState
from engine.teams import (ALL_TEAMS, CHANNELS, DEFAULT_MODEL, ZERO_EDGE, Edge, TeamModel, find_team,
                          logit_shift, pick_tilted)


from test_engine_tables import tables as make_tables

tables = make_tables()


def test_team_list():
    assert len(ALL_TEAMS) == 32 and len({t.abbr for t in ALL_TEAMS}) == 32 and len({t.name for t in ALL_TEAMS}) == 32
    rams = find_team("rams")
    assert (rams.abbr, rams.offense, rams.defense, rams.overall) == ("LAR", 87, 86, 87)
    assert find_team("49ERS").abbr == "SF" and find_team(" kc ").name == "Chiefs" and find_team("Bills").offense == 90
    try:
        find_team("Whalers")
        raise AssertionError("an unknown team must be refused")
    except ValueError:
        pass


def test_logit_shift():
    assert logit_shift(0.3, 0) == 0.3
    assert abs(logit_shift(0.5, math.log(3)) - 0.75) < 1e-12
    assert logit_shift(0.6, 1) > 0.6 > logit_shift(0.6, -1)
    assert logit_shift(0.0, 5) == 0.0 and logit_shift(1.0, -5) == 1.0


def test_pick_tilted():
    bank = [(y, 0, 0) for y in range(-5, 40)]
    a, b = random.Random(1), random.Random(1)
    assert [pick_tilted(bank, 0, a, tables) for _ in range(50)] == [b.choice(bank) for _ in range(50)]
    rng = random.Random(2)
    base = sum(pick_tilted(bank, 0, rng, tables)[0] for _ in range(20000)) / 20000
    up = sum(pick_tilted(bank, 0.03, rng, tables)[0] for _ in range(20000)) / 20000
    down = sum(pick_tilted(bank, -0.03, rng, tables)[0] for _ in range(20000)) / 20000
    assert down < base < up, (down, base, up)
    assert all(-5 <= pick_tilted(bank, 5.0, rng, tables)[0] <= 39 for _ in range(100))


def sample(edge, n):
    rng = random.Random(7)
    s = GameState(quarter=1, clock=900, down=1, distance=10, yardline_100=60, offense_score=0, defense_score=0)
    return [run_scrimmage_play(s, tables, rng, edge) for _ in range(n)]


PASSES = ("complete", "incomplete", "interception")
DROPBACKS = PASSES + ("sack", "scramble")


def rate(outs, kind, among):
    pool = [o for o in outs if o.kind in among]
    return sum(o.kind == kind for o in pool) / len(pool)


def avg_yards(outs, kind):
    ys = [o.yards for o in outs if o.kind == kind]
    return sum(ys) / len(ys)


def test_edges_move_the_right_way():
    n = 40000
    base = sample(ZERO_EDGE, n)
    up, down = sample(Edge(completion=0.3), n), sample(Edge(completion=-0.3), n)
    assert rate(up, "complete", PASSES) > rate(base, "complete", PASSES) + 0.02
    assert rate(down, "complete", PASSES) < rate(base, "complete", PASSES) - 0.02
    assert rate(sample(Edge(interception=0.5), n), "interception", PASSES) > rate(base, "interception", PASSES)
    assert rate(sample(Edge(sack=0.5), n), "sack", DROPBACKS) > rate(base, "sack", DROPBACKS)
    assert avg_yards(sample(Edge(run_tilt=0.02), n), "run") > avg_yards(base, "run") + 0.2
    assert avg_yards(sample(Edge(run_tilt=-0.02), n), "run") < avg_yards(base, "run") - 0.2
    assert avg_yards(sample(Edge(completion_tilt=0.01), n), "complete") > avg_yards(base, "complete") + 0.3


def test_zero_edge_changes_nothing():
    for seed in range(6):
        events = []
        for edges in (None, (ZERO_EDGE, ZERO_EDGE)):
            g = Game(tables, random.Random(seed), edges=edges)
            lines = []
            while not g.over or g._queue:
                lines.append(g.step().text)
            events.append(lines)
        assert events[0] == events[1], seed


def test_probabilities_stay_legal():
    wild = Edge(completion=3, interception=3, sack=3, completion_tilt=0.05, run_tilt=0.05)
    sample(wild, 12000)


def test_game_with_edges_finishes():
    model = TeamModel()
    for seed, (a, b) in enumerate([("rams", "dolphins"), ("bills", "seahawks"), ("titans", "titans")]):
        g = Game(tables, random.Random(seed), names=("A", "B"), edges=model.edge_pair(find_team(a), find_team(b)))
        while not g.over or g._queue:
            g.step()
        assert g.scores[0] >= 0 and g.scores[1] >= 0


def test_model():
    model = TeamModel()
    p = json.loads(Path(DEFAULT_MODEL).read_text())
    for side in ("offense", "defense"):
        assert abs(sum(p["channels"][ch][f"share_{side}"] for ch in CHANNELS) - 1) < 1e-9
        assert abs(sum(model.z(t, side) for t in ALL_TEAMS)) < 1e-9
        assert abs(sum(model.z(t, side) ** 2 for t in ALL_TEAMS) / 32 - 1) < 1e-9
    rams, dolphins, bills, raiders = (find_team(x) for x in ("rams", "dolphins", "bills", "raiders"))
    assert model.edge(rams, dolphins).completion > 0 > model.edge(dolphins, rams).completion
    add, take = model.points_edge(bills, raiders)
    assert add > 0
    nets = [model.points_edge(t, t)[0] + model.points_edge(t, t)[1] for t in ALL_TEAMS]
    mean = sum(nets) / 32
    sd = (sum((n - mean) ** 2 for n in nets) / 32) ** 0.5
    assert abs(sd - 5.17 * p["league_correction"]) < 1e-6, sd
    assert p["points_per_rating_point_offense"] == p["points_per_rating_point_defense"]
    e = model.edge(rams, dolphins)
    assert e.completion > 0 and e.interception < 0 and e.sack < 0 and e.completion_tilt > 0 and e.run_tilt > 0


def test_centering_keeps_league_averages():
    model = TeamModel()
    pairs = [(o, d) for o in ALL_TEAMS for d in ALL_TEAMS if o is not d]
    n = len(pairs)
    for field, p0, tol in (("completion", 0.641, 0.0015), ("interception", 0.0225, 0.0004), ("sack", 0.0628, 0.0005)):
        mean = sum(logit_shift(p0, getattr(model.edge(o, d), field)) for o, d in pairs) / n
        assert abs(mean - p0) < tol, (field, mean)
    assert max(abs(v) for v in model.params["channel_centering"].values()) < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All team checks passed.")
