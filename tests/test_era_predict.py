import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
from calibration.era_fit import update
from calibration.predict import standings
from engine.tables import Tables


def test_era_file_has_every_dial_and_pace_is_near_one():
    era = json.loads((ROOT / "data" / "tables" / "era.json").read_text())
    for key in ("pace", "completion", "interception", "sack", "completion_tilt", "run_tilt"):
        assert key in era
    assert 0.9 < era["pace"] < 1.3


def test_pace_stretches_clock_seconds_on_average():
    t = Tables()
    base = Tables(pace=1.0)
    t.pace = 1.2
    rng = random.Random(1)
    slow = sum(t.draw_seconds("complete_in", "last2", "leading_big", rng) for _ in range(4000))
    rng = random.Random(1)
    base.pace = 1.0
    fast = sum(base.draw_seconds("complete_in", "last2", "leading_big", rng) for _ in range(4000))
    assert slow > fast * 1.1


def test_update_moves_dials_toward_the_real_numbers():
    real = {"plays": 60, "pass_att": 32, "comp": 20, "pass_yards": 220, "rush_att": 25, "rush_yards": 110, "sacks": 2.4,
            "ints": 0.7, "first_downs": 17, "points": 22}
    sim = {**real, "plays": 66, "comp": 22, "pass_yards": 250}
    dials = {"pace": 1.0, "completion": 0.0, "interception": 0.0, "sack": 0.0, "completion_tilt": 0.0, "run_tilt": 0.0}
    new = update(dials, real, sim, {"pass": 60.0, "run": 30.0})
    assert new["pace"] > 1.0 and new["completion"] < 0 and new["completion_tilt"] < 0


def test_standings_add_played_games_to_projected_wins():
    games = pd.DataFrame([{"home_team": "A", "away_team": "B", "result": 3.0},
                          {"home_team": "A", "away_team": "B", "result": float("nan")}])
    pred = pd.DataFrame([{"home_team": "A", "away_team": "B", "home_win": 0.75, "margin": 4.0}])
    t = standings(games, pred).set_index("team")
    assert t.loc["A", "w"] == 1 and t.loc["B", "l"] == 1
    assert abs(t.loc["A", "proj_wins"] - 1.75) < 1e-9 and abs(t.loc["B", "proj_wins"] - 0.25) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("All era and predict checks passed.")
