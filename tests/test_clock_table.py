from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KINDS = ["sack", "interception", "pass_td", "complete_oob", "complete_in", "incomplete",
         "run_td", "run_fumble", "scramble_oob", "scramble_in", "run_oob", "run_in",
         "punt", "field_goal", "kickoff", "extra_point"]


TEMPOS = ["normal", "last5", "last2"]
SCORES = ["trailing_big", "trailing_close", "tied", "leading_close", "leading_big"]
LEVELS = [("L1", ["kind", "tempo", "score_b"]), ("L2", ["kind", "tempo"]), ("L3", ["kind"])]


def load_index() -> dict:
    bank = pd.read_csv(ROOT / "data" / "tables" / "bank_time.csv")
    assert not bank.isna().any().any()
    assert bank["seconds"].between(0, 120).all()
    index = {}
    for (level, kind, tempo, score), g in bank.groupby(["level", "kind", "tempo", "score_b"]):
        index[(level, kind, tempo, score)] = g["seconds"].to_numpy()
    return index


def draw(index: dict, kind: str, tempo: str, score: str) -> np.ndarray:
    for level, keys in LEVELS:
        key = (level, kind, tempo if "tempo" in keys else "ALL", score if "score_b" in keys else "ALL")
        if key in index:
            return index[key]
    raise AssertionError(f"no timing for {kind}, {tempo}, {score}")


def test_every_play_has_a_timing():
    index = load_index()
    for kind, tempo, score in product(KINDS, TEMPOS, SCORES):
        assert len(draw(index, kind, tempo, score)) >= 100


def test_timings_make_football_sense():
    index = load_index()
    med = lambda k, t="normal", s="tied": float(np.median(draw(index, k, t, s)))
    assert med("incomplete") < 8, "clock stops on an incomplete pass"
    assert 33 <= med("run_in") <= 44 and 33 <= med("complete_in") <= 44, "running clock plays use about a play clock"
    assert med("run_oob") < med("run_in") and med("complete_oob") < med("complete_in"), "out of bounds stops the clock"
    assert med("pass_td") < 10 and med("run_td") < 10, "scores stop the clock"
    assert 6 <= med("punt") <= 13
    assert med("extra_point") <= 1
    assert med("run_in", "last2", "trailing_close") < med("run_in", "normal", "trailing_close") - 5
    assert med("complete_in", "last2", "trailing_big") < med("complete_in", "normal", "trailing_big") - 5


if __name__ == "__main__":
    test_every_play_has_a_timing()
    test_timings_make_football_sense()
    print("All clock table tests passed")
