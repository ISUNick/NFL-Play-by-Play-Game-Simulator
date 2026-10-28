import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine import winprob
from engine.game import Game
from test_engine_tables import tables as make_tables


def new_game(seed=1):
    return Game(make_tables(), random.Random(seed), names=("HOME", "AWAY"), home_field=True)


def finish(g):
    while not (g.over and not g._queue):
        g.step()
    return g


def test_fork_is_independent_of_the_original():
    g = new_game()
    for _ in range(40):
        g.step()
    before = (list(g.scores), g.quarter, g.clock, g.steps, g.yardline_100)
    twin = g.fork("x")
    finish(twin)
    assert (list(g.scores), g.quarter, g.clock, g.steps, g.yardline_100) == before


def test_fork_with_the_same_seed_plays_out_the_same_way():
    g = new_game()
    for _ in range(40):
        g.step()
    a, b = finish(g.fork("same")), finish(g.fork("same"))
    assert a.scores == b.scores


def test_rollout_of_a_finished_game_is_exact():
    g = finish(new_game(3))
    p = winprob.rollout(g, 5, "done")
    a, b = g.scores
    assert p == (1.0 if a > b else 0.5 if a == b else 0.0)


def test_big_late_lead_is_nearly_certain_for_the_rollouts():
    g = new_game()
    g.scores, g.quarter, g.clock = [28, 0], 4, 90
    assert winprob.rollout(g, 40, "lead") > 0.97


def test_seconds_left():
    g = new_game()
    assert winprob.seconds_left(g) == 3600
    g.quarter, g.clock = 3, 100
    assert winprob.seconds_left(g) == 1000
    g.quarter, g.clock = 5, 420
    assert winprob.seconds_left(g) == 420


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40000, 3))
    true = np.array([0.3, 1.0, -0.5, 0.2])
    y = (rng.random(40000) < 1 / (1 + np.exp(-(true[0] + x @ true[1:])))).astype(float)
    beta = winprob.fit(x, y)
    assert np.allclose(beta, true, atol=0.07)


def test_the_shipped_model_behaves():
    m = winprob.WinProb()
    g = new_game()
    start = m.home(g)
    assert 0.4 < start < 0.7
    g.scores = [14, 0]
    assert m.home(g) > start
    g.scores = [0, 14]
    assert m.home(g) < start
    g.scores, g.quarter, g.clock = [21, 0], 4, 60
    assert m.home(g) > 0.97
    g.scores = [0, 21]
    assert m.home(g) < 0.03


def test_pregame_edge_matters_early_and_fades_late():
    m = winprob.WinProb()
    g = new_game()
    early = m.home(g, 0.8) - m.home(g, 0.5)
    g.quarter, g.clock = 4, 120
    late = m.home(g, 0.8) - m.home(g, 0.5)
    assert early > 0.1 and late < early / 2


def test_track_ends_at_the_real_result_and_stays_in_range():
    g = new_game(9)
    t = winprob.track(g, winprob.WinProb())
    assert all(0.0 <= p <= 1.0 for *_, p in t)
    a, b = g.scores
    assert t[-1][3] == (1.0 if a > b else 0.5 if a == b else 0.0)


def test_the_shipped_model_agrees_with_rollouts():
    m = winprob.WinProb()
    errors = []
    for seed in range(6):
        g = new_game(100 + seed)
        for _ in range(60 + 25 * seed):
            if g.over and not g._queue:
                break
            g.step()
        errors.append(abs(m.home(g) - winprob.rollout(g, 300, f"agree{seed}")))
    assert np.mean(errors) < 0.08, errors


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All win probability checks passed.")
