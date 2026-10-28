import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
import madden
from players.lineup import Ratings


def test_clean_drops_suffix_and_punctuation():
    assert madden.clean("Patrick Mahomes II") == "patrick mahomes"
    assert madden.clean("C.J. Stroud") == "cj stroud"


def test_qb_file_is_loaded_and_ordered():
    r = madden.ratings("QB")
    assert len(r) > 60
    assert r["josh allen"] > r["tanner mckee"]


def test_z_scores_floor_unrated_players():
    roster = pd.DataFrame({"gsis_id": ["a", "b"], "full_name": ["Josh Allen", "Nobody Atall"], "position": ["QB", "QB"]})
    z = madden.z_scores("QB", roster)
    assert z["a"] > 1
    assert z["b"] == -1.5


def test_missing_position_file_gives_empty():
    assert madden.ratings("XX").empty


def test_qb_scale_changes_only_quarterbacks():
    base = Ratings.load()
    a, b = base.scaled(1.0, 1.0), base.scaled(1.0, 1.0, 3.0)
    assert np.allclose(a.receiver["comp_shift"], b.receiver["comp_shift"])
    assert np.allclose(a.qb["comp_shift"] * 3.0, b.qb["comp_shift"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All madden checks passed.")
