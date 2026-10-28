import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))

from buckets import (
    add_buckets, fourth_distance_bucket as fdist_pd, fourth_zone_bucket as fzone_pd,
    late_distance_bucket as late_dist_pd, fourth_late_distance_bucket as flate_dist_pd, late_quarter_bucket as lateq_pd, late_score_bucket as late_score_pd, late_time_bucket as late_time_pd,
    late_zone2_bucket as late_zone2_pd, late_zone_bucket as late_zone_pd, tempo_bucket as tempo_bucket_pd,
    half_window_bucket as half_window_pd, half_zone_bucket as half_zone_pd,
)
from engine.state import (
    GameState, distance_bucket, fourth_distance_bucket, fourth_zone_bucket, score_bucket,
    half_window_bucket, half_zone_bucket, fourth_late_distance_bucket, late_quarter_bucket, late_distance_bucket, late_score_bucket, late_time_bucket,
    late_zone2_bucket, late_zone_bucket, tempo_bucket, time_bucket, zone_bucket,
)


def test_buckets_match_pandas():
    rows = []
    for ydstogo in range(1, 31):
        for yardline in range(1, 100):
            rows.append((ydstogo, yardline))
    df = pd.DataFrame(rows, columns=["ydstogo", "yardline_100"])
    df["score_differential"] = 0
    df["game_seconds_remaining"] = 1000
    out = add_buckets(df)
    for _, r in out.iterrows():
        assert distance_bucket(r.ydstogo) == r.dist_b, ("distance", r.ydstogo)
        assert zone_bucket(r.yardline_100) == r.zone_b, ("zone", r.yardline_100)

    scores = pd.DataFrame({"score_differential": range(-40, 41)})
    scores["ydstogo"], scores["yardline_100"] = 10, 50
    scores["game_seconds_remaining"] = 1000
    out = add_buckets(scores)
    for _, r in out.iterrows():
        assert score_bucket(r.score_differential) == r.score_b, ("score", r.score_differential)

    times = pd.DataFrame({"game_seconds_remaining": range(0, 3601)})
    times["ydstogo"], times["yardline_100"], times["score_differential"] = 10, 50, 0
    out = add_buckets(times)
    for _, r in out.iterrows():
        assert time_bucket(r.game_seconds_remaining) == r.time_b, ("time", r.game_seconds_remaining)


def test_tempo_bucket_matches_pandas():
    rows = [(q, h) for q in range(1, 6) for h in range(0, 1801)]
    df = pd.DataFrame(rows, columns=["qtr", "half_seconds_remaining"])
    df["tempo"] = tempo_bucket_pd(df["qtr"], df["half_seconds_remaining"])
    for r in df.itertuples():
        assert tempo_bucket(r.qtr, r.half_seconds_remaining) == r.tempo, (r.qtr, r.half_seconds_remaining)
    assert GameState(quarter=2, clock=100).tempo == "last2"
    assert GameState(quarter=4, clock=200).tempo == "last5"
    assert GameState(quarter=1, clock=60).tempo == "normal"
    assert GameState(quarter=3, clock=60).tempo == "normal"
    assert GameState(quarter=5, clock=100).tempo == "last2" and GameState(quarter=5, clock=500).tempo == "normal"
    assert GameState(quarter=5, clock=420).half_seconds_remaining == 420
    assert GameState(quarter=5, clock=420).game_seconds_remaining == 420


def test_fourth_down_buckets_match_pandas():
    d = pd.Series(range(0, 41))
    for a, b in zip(d, fdist_pd(d)):
        assert fourth_distance_bucket(a) == b, ("fdist", a)
    y = pd.Series(range(1, 100))
    for a, b in zip(y, fzone_pd(y)):
        assert fourth_zone_bucket(a) == b, ("fzone", a)
    s = GameState(quarter=4, clock=100, down=4, distance=1, yardline_100=33,
                  offense_score=10, defense_score=13)
    assert s.fourth_down_situation() == {"fdist": "1", "fzone": "z31_36",
                                         "score_b": "trailing_close", "time_b": "late"}


def test_late_game_buckets_match_pandas():
    scores = pd.Series(range(-60, 61))
    for a, b in zip(scores, late_score_pd(scores)):
        assert late_score_bucket(a) == b, ("late score", a)
    rows = [(q, (4 - q) * 900 + clock) for q in (1, 2, 3, 4) for clock in range(0, 901)]
    df = pd.DataFrame(rows, columns=["qtr", "game_seconds_remaining"])
    df["b"] = late_time_pd(df["qtr"], df["game_seconds_remaining"])
    for r in df.itertuples():
        assert late_time_bucket(r.qtr, r.game_seconds_remaining) == r.b, (r.qtr, r.game_seconds_remaining)
    y = pd.Series(range(1, 100))
    for a, b, c in zip(y, late_zone_pd(y), late_zone2_pd(y)):
        assert late_zone_bucket(a) == b and late_zone2_bucket(a) == c, ("late zone", a)
    d = pd.Series(range(1, 41))
    for a, b in zip(d, late_dist_pd(d)):
        assert late_distance_bucket(a) == b, ("late dist", a)
    for a, b in zip(d, flate_dist_pd(d)):
        assert fourth_late_distance_bucket(a) == b, ("late 4th dist", a)
    assert [fourth_late_distance_bucket(x) for x in (1, 2, 3, 6, 7, 30)] == ["1", "2", "3_6", "3_6", "7p", "7p"]
    lt = pd.Series(["q2_last2", "q2_last7", "q4_last2", "q4_last5", "q4_last7", "none"])
    for a, b in zip(lt, lateq_pd(lt)):
        assert late_quarter_bucket(a) == b, ("late q", a)
    s = GameState(quarter=4, clock=23, down=4, distance=7, yardline_100=7, offense_score=21, defense_score=27)
    assert s.late_situation() == {"down": 4, "dist_b": "long", "late_score": "trail_4_8", "late_time": "q4_last2",
                                  "late_zone": "goal", "late_zone2": "near", "late_dist": "long",
                                  "late_q": "q4", "late_fdist": "7p"}
    assert GameState(quarter=3, clock=10).late_situation()["late_time"] == "none"
    assert GameState(quarter=2, clock=100).late_situation()["late_time"] == "q2_last2"
    assert GameState(quarter=2, clock=250).late_situation()["late_time"] == "q2_last5"
    assert GameState(quarter=2, clock=400).late_situation()["late_time"] == "q2_last7"
    assert GameState(quarter=2, clock=500).late_situation()["late_time"] == "none"
    assert GameState(quarter=2, clock=0).late_situation()["late_time"] == "none"
    assert GameState(quarter=1, clock=100).late_situation()["late_time"] == "none"
    assert GameState(quarter=4, clock=0).late_situation()["late_time"] == "none"
    assert GameState(quarter=4, clock=421).late_situation()["late_time"] == "none"


def test_half_buckets_match_pandas():
    seconds = pd.Series(range(-5, 400))
    for a, b in zip(seconds, half_window_pd(seconds)):
        assert half_window_bucket(a) == b, ("half window", a)
    assert half_window_bucket(3) == "s3" and half_window_bucket(4) == "s6" and half_window_bucket(120) == "s120"
    assert half_window_bucket(241) == "none" and half_window_bucket(0) == "none"
    y = pd.Series(range(1, 100))
    for a, b in zip(y, half_zone_pd(y)):
        assert half_zone_bucket(a) == b, ("half zone", a)
    s = GameState(quarter=2, clock=25, down=1, distance=10, yardline_100=70, offense_timeouts=2,
                  defense_timeouts=3, new_possession=True)
    assert s.half_situation() == {"hwin": "s30", "rs": "newposs", "hzone": "h_own", "dto": 2, "has_to": 1, "hdown": 1}
    s.new_possession, s.clock_running, s.offense_timeouts, s.down = False, True, 0, 4
    assert s.half_situation() == {"hwin": "s30", "rs": "running", "hzone": "h_own", "dto": 2, "has_to": 0, "hdown": 4}
    s.clock_running = False
    assert s.half_situation()["rs"] == "stopped"
    assert GameState(quarter=4, clock=25).half_situation()["hwin"] == "none"


def test_score_labels():
    expected = {-30: "trailing_big", -9: "trailing_big", -8: "trailing_close",
                -1: "trailing_close", 0: "tied", 1: "leading_close",
                8: "leading_close", 9: "leading_big", 30: "leading_big"}
    for diff, label in expected.items():
        assert score_bucket(diff) == label, (diff, label)


def test_clock_math():
    assert GameState(quarter=1, clock=900).game_seconds_remaining == 3600
    assert GameState(quarter=4, clock=60).game_seconds_remaining == 60
    assert GameState(quarter=2, clock=120).half_seconds_remaining == 120
    assert GameState(quarter=3, clock=900).half_seconds_remaining == 1800
    assert GameState(quarter=1, clock=900).half_seconds_remaining == 1800


def test_score_and_situation():
    s = GameState(offense_score=10, defense_score=24, down=3, distance=8,
                  yardline_100=95, quarter=4, clock=120)
    assert s.score_diff == -14
    assert s.situation() == {
        "down": 3, "dist_b": "long", "zone_b": "own_10",
        "score_b": "trailing_big", "time_b": "late",
    }


if __name__ == "__main__":
    test_buckets_match_pandas()
    test_tempo_bucket_matches_pandas()
    test_fourth_down_buckets_match_pandas()
    test_late_game_buckets_match_pandas()
    test_half_buckets_match_pandas()
    test_score_labels()
    test_clock_math()
    test_score_and_situation()
    print("All state tests passed")
