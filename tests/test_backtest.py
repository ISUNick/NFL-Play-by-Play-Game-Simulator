import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))
sys.path.insert(0, str(ROOT / "data_prep"))
import backtest
import build_players as bp


def fake_games(n=400, seed=0):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 6, n)
    result = np.round(strength + rng.normal(0, 13, n))
    result[result == 0] = 1
    p = 1 / (1 + np.exp(-strength / 6))
    return pd.DataFrame({"win_prob": p, "sim_margin": strength, "sim_total": 45.0, "result": result, "total": 45.0,
                         "spread_line": strength, "total_line": 45.0})


def test_normal_cdf():
    assert abs(backtest.normal_cdf(0) - 0.5) < 1e-12
    assert abs(backtest.normal_cdf(1.96) - 0.975) < 1e-3


def test_score_on_a_model_that_knows_the_truth():
    s = backtest.score(fake_games())
    assert s["games"] == 400
    assert s["straight_up_accuracy"] > 0.6
    assert s["brier"] < 0.25
    assert abs(s["margin_corr_with_result"] - s["vegas_margin_corr_with_result"]) < 1e-9


def test_score_notices_a_useless_model():
    df = fake_games()
    df["win_prob"] = 0.5
    df["sim_margin"] = 0.0 + np.random.default_rng(1).normal(0, 1, len(df))
    assert abs(backtest.score(df)["brier"] - 0.25) < 1e-9
    assert backtest.score(df)["margin_corr_with_result"] < 0.2


def test_calibration_table_bins_add_up():
    rows = backtest.calibration_table(fake_games())
    assert sum(r["games"] for r in rows) == 400
    for r in rows:
        assert abs(r["predicted"] - r["actual"]) < 0.2


def test_ratings_use_nothing_from_the_week_they_predict():
    if not (bp.RAW_DIR / "pbp_2025.parquet").exists():
        return
    plays = bp.load_plays([2023, 2024, 2025], before=(2025, 10))
    assert plays.time.max() == 2025 * 22 + 9
    assert (plays.time < 2025 * 22 + 10).all()


def test_real_games_have_lines_and_results():
    if not (bp.RAW_DIR / "pbp_2025.parquet").exists():
        return
    g = backtest.real_games([2025])
    assert 250 < len(g) <= 272
    assert g.spread_line.notna().all() and g.result.notna().all()
    assert set(g.home_team) <= set(bp.latest_roster().team) | {"LAR"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All backtest checks passed.")
