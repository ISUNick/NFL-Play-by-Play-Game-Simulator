import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calibration.backtest import score

FILES = ROOT / "data" / "backtest"


def load(season: int) -> pd.DataFrame:
    return pd.read_csv(FILES / f"games_{season}_players_t1.5_p1.5_wf.csv")


def fit_line(df: pd.DataFrame) -> tuple:
    # result margin as a straight line of the sim margin, so the spread of predictions matches how much they explain
    a, b = np.polyfit(df.sim_margin, df.result, 1)
    return float(a), float(b)


def fit_win(df: pd.DataFrame, sd: float) -> float:
    # win chance = cdf(slope * margin / sd), slope picked by log loss on earlier seasons only
    from math import erf, sqrt
    won = (df.result > 0).to_numpy(float)
    best, best_loss = 1.0, 9.0
    for k in np.arange(0.2, 1.6, 0.02):
        p = np.clip([0.5 * (1 + erf(k * m / sd / sqrt(2))) for m in df.sim_margin], 0.01, 0.99)
        loss = -np.mean(won * np.log(p) + (1 - won) * np.log(1 - p))
        if loss < best_loss:
            best, best_loss = k, loss
    return float(best)


def main():
    from math import erf, sqrt
    seasons = [s for s in (2022, 2023, 2024, 2025) if (FILES / f"games_{s}_players_t1.5_p1.5_wf.csv").exists()]
    for i in range(1, len(seasons)):
        train = pd.concat([load(s) for s in seasons[:i]])
        test = load(seasons[i]).copy()
        a, b = fit_line(train)
        k = fit_win(train, 13.0)
        test["sim_margin"] = a * test.sim_margin + b
        test["win_prob"] = [0.5 * (1 + erf(k * m / 13.0 / sqrt(2))) for m in test.sim_margin]
        r = score(test)
        print(seasons[i], f"slope {a:.2f} int {b:.2f} k {k:.2f}", {x: round(r[x], 3) for x in (
            "straight_up_accuracy", "vegas_straight_up_accuracy", "brier", "vegas_brier", "logloss", "vegas_logloss",
            "margin_mae", "vegas_margin_mae")})


if __name__ == "__main__":
    main()
