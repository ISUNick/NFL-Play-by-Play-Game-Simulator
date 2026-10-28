import json
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "tables" / "home_field.json"
_DATA = None


def data() -> dict:
    global _DATA
    if _DATA is None:
        _DATA = json.loads(PATH.read_text())
    return _DATA


def rating_points() -> float:
    return float(data().get("rating_points", 0.0))


def multiplier(phase: str, side: str, at_home: bool) -> float:
    return data()["penalties"][f"{phase}_{side}"]["multiplier"]["home" if at_home else "away"]


def foul_weight(side: str, foul: str, at_home: bool) -> float:
    weights = data()["penalties"][f"pre_{side}"]["foul_weights"]
    return weights.get(foul, weights["other"])["home" if at_home else "away"]


def max_foul_weight(side: str, at_home: bool) -> float:
    loc = "home" if at_home else "away"
    return max(w[loc] for w in data()["penalties"][f"pre_{side}"]["foul_weights"].values())
