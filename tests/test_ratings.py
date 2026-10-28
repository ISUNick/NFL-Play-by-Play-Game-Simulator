import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from players import shrink as sh

PLAYERS = ROOT / "data" / "players"


def test_logistic_effects_recover_true_talent():
    rng = np.random.default_rng(1)
    n_players, n = 40, 40000
    truth = rng.normal(0, 0.3, n_players)
    codes = rng.integers(0, n_players, n)
    base = np.zeros(n)
    y = (rng.random(n) < sh.expit(truth[codes])).astype(float)
    eff = sh.fit_logistic_effects(y, base, {"p": (codes, n_players)}, {"p": 0.3 ** 2})["p"]
    assert np.corrcoef(eff, truth)[0, 1] > 0.95


def test_small_samples_are_pulled_in_harder():
    codes = np.concatenate([np.zeros(20, int), np.ones(2000, int)])
    y = np.ones(len(codes))
    eff = sh.fit_logistic_effects(y, np.zeros(len(codes)), {"p": (codes, 2)}, {"p": 0.2 ** 2})["p"]
    assert abs(eff[0]) < abs(eff[1]) and eff[0] > 0


def test_no_data_returns_the_prior():
    shape, rate = 3.0, 30.0
    assert sh.shrink_rate([0.0], [0.0], shape, rate)[0] == shape / rate
    means = sh.shrink_mean(np.array([5.0, 0.0, 1.0, -1.0, 0.5, 2.0]), np.array([0, 100, 100, 100, 100, 100.0]), 4.0)[0]
    assert abs(means[0] - 0.5) < 0.5


def test_lots_of_data_approaches_the_raw_rate():
    shape, rate = 3.0, 30.0
    got = sh.shrink_rate([3000.0], [10000.0], shape, rate)[0]
    assert abs(got - 0.3) < 0.005


def test_poisson_prior_finds_the_league_rate():
    rng = np.random.default_rng(3)
    exposure = rng.integers(100, 600, 300).astype(float)
    counts = rng.poisson(0.12 * exposure).astype(float)
    shape, rate = sh.poisson_prior(counts, exposure)
    assert abs(shape / rate - 0.12) < 0.01


def test_depth_shares_sum_to_one():
    counts = np.array([[10, 5, 1], [0, 0, 0], [50, 40, 30.0]])
    s = sh.shrink_shares(counts, 20)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert np.allclose(s[1], counts.sum(axis=0) / counts.sum())


def test_tau_choice_prefers_small_penalty_when_talent_is_real():
    rng = np.random.default_rng(4)
    n_players, n = 30, 30000
    truth = rng.normal(0, 0.4, n_players)
    codes = rng.integers(0, n_players, n)
    y = (rng.random(n) < sh.expit(truth[codes])).astype(float)
    fold = rng.integers(0, 2, n)
    tau = sh.choose_tau(y, np.zeros(n), {"p": (codes, n_players)}, np.ones(n), fold, {"p": [0.01, 0.4, 1.0]})
    assert tau["p"] > 0.01


def test_shipped_ratings_look_right():
    qbs = pd.read_csv(PLAYERS / "qb_ratings.csv")
    rec = pd.read_csv(PLAYERS / "receiver_ratings.csv")
    run = pd.read_csv(PLAYERS / "runner_ratings.csv")
    assert len(qbs) > 50 and len(rec) > 300 and len(run) > 200
    big = qbs[qbs.dropbacks_w > 200]
    assert abs(np.average(big.comp_shift, weights=big.dropbacks_w)) < 0.1
    assert 0.55 < big.completion_pct.mean() < 0.75
    assert rec.comp_shift.abs().max() < 1.0
    for g, lo, hi in (("WR", 0.05, 0.3), ("TE", 0.03, 0.25), ("RB", 0.02, 0.25)):
        r = rec[(rec.position == g) & (rec.pass_snaps_w > 200)].target_rate.mean()
        assert lo < r < hi, (g, r)
    mix = rec[["share_short", "share_medium", "share_deep"]].dropna().sum(axis=1)
    assert np.allclose(mix, 1.0, atol=1e-3)
    assert run.yards_over_expected.abs().max() < 2.5
    assert run[run.position != "QB"].yards_over_expected.abs().max() < 1.5


def test_validation_shows_players_add_information():
    v = json.loads((PLAYERS / "validation.json").read_text())
    assert v["logloss_qb_and_receiver"] <= v["logloss_cp_only"]
    assert v["split_half_corr_qb"] > 0.2
    assert abs(v["predicted_completion_rate"] - v["real_completion_rate"]) < 0.03


def test_current_roster_has_every_team():
    cur = pd.read_csv(PLAYERS / "current_roster.csv")
    assert cur.team.nunique() == 32
    assert cur.gsis_id.is_unique


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All rating checks passed.")
