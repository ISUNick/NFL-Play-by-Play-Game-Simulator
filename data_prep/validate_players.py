import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
from players import shrink as sh
import build_players as bp

TEST_WEEKS = 12


def throws_table(plays: pd.DataFrame) -> pd.DataFrame:
    t = plays[plays.is_throw & plays.cp.notna() & plays.receiver_player_id.notna()].copy()
    t["y"] = (t.complete_pass == 1).astype(float)
    return t


def logloss(y, p, w=None):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))


def predict(test: pd.DataFrame, offset: float, qb_effect: pd.Series, rec_effect: pd.Series) -> np.ndarray:
    eta = sh.logit(test.cp.to_numpy()) + offset
    eta += test.qb_id.map(qb_effect).fillna(0.0).to_numpy()
    eta += test.receiver_player_id.map(rec_effect).fillna(0.0).to_numpy()
    return sh.expit(eta)


def out_of_sample(throws: pd.DataFrame) -> dict:
    # fit on everything before the last TEST_WEEKS weeks, score the throws after
    cut = throws.time.max() - TEST_WEEKS
    train, test = throws[throws.time <= cut].copy(), throws[throws.time > cut]
    train["w"] = 0.5 ** ((train.time.max() - train.time) / bp.HALF_LIFE_WEEKS)
    base = sh.logit(train.cp.to_numpy())
    offset = bp.calibrate_offset(base, train.y.to_numpy(), train.w.to_numpy())
    fit, tau = bp.fit_players(train.y.to_numpy(), base + offset,
                              {"qb": train.qb_id, "receiver": train.receiver_player_id},
                              train.w.to_numpy(), train.game_id, bp.TAU_GRID)
    none = pd.Series(dtype=float)
    y = test.y.to_numpy()
    cp_only = logloss(y, predict(test, offset, none, none))
    both = logloss(y, predict(test, offset, fit["qb"], fit["receiver"]))
    qb_only = logloss(y, predict(test, offset, fit["qb"], none))
    p = predict(test, offset, fit["qb"], fit["receiver"])
    return {"test_throws": len(test), "logloss_cp_only": cp_only, "logloss_qb_only": qb_only,
            "logloss_qb_and_receiver": both, "improvement_vs_cp_only": cp_only - both,
            "real_completion_rate": float(y.mean()), "predicted_completion_rate": float(p.mean()),
            "tau_qb": tau["qb"], "tau_receiver": tau["receiver"]}


def split_half(throws: pd.DataFrame) -> dict:
    # fit the same model on odd and even games separately, then see if the same players come out on top
    out = {}
    base = sh.logit(throws.cp.to_numpy())
    offset = bp.calibrate_offset(base, throws.y.to_numpy(), np.ones(len(throws)))
    half = pd.factorize(throws.game_id)[0] % 2
    fits = []
    for h in (0, 1):
        part = throws[half == h]
        eff = sh.fit_logistic_effects(part.y.to_numpy(), sh.logit(part.cp.to_numpy()) + offset,
                                      {"qb": (pd.factorize(part.qb_id)[0], part.qb_id.nunique()),
                                       "receiver": (pd.factorize(part.receiver_player_id)[0], part.receiver_player_id.nunique())},
                                      {"qb": 0.2 ** 2, "receiver": 0.2 ** 2})
        fits.append({"qb": pd.Series(eff["qb"], index=pd.factorize(part.qb_id)[1]),
                     "receiver": pd.Series(eff["receiver"], index=pd.factorize(part.receiver_player_id)[1])})
    for role in ("qb", "receiver"):
        both = pd.concat([fits[0][role], fits[1][role]], axis=1, join="inner").dropna()
        both = both[both.abs().sum(axis=1) > 0]
        out[f"split_half_corr_{role}"] = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        out[f"split_half_n_{role}"] = int(len(both))
    return out


def main():
    last = bp.latest_season()
    seasons = list(range(last - bp.SEASONS_BACK, last + 1))
    plays = bp.load_plays(seasons)
    throws = throws_table(plays)
    result = out_of_sample(throws)
    result.update(split_half(throws))

    qbs = pd.read_csv(bp.OUT_DIR / "qb_ratings.csv")
    result["qb_comp_shift_sd"] = float(qbs[qbs.dropbacks_w > 200].comp_shift.std())
    (bp.OUT_DIR / "validation.json").write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
