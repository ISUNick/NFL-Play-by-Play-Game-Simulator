import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "data" / "tables" / "team_model.json"

TEAMS = [
    ("Rams", "LAR", 87, 86, 87), ("49ers", "SF", 89, 84, 86), ("Ravens", "BAL", 88, 84, 86),
    ("Chiefs", "KC", 88, 82, 85), ("Seahawks", "SEA", 79, 89, 84), ("Lions", "DET", 89, 76, 83),
    ("Patriots", "NE", 83, 83, 83), ("Jaguars", "JAX", 82, 83, 83), ("Broncos", "DEN", 79, 87, 83),
    ("Texans", "HOU", 78, 87, 83), ("Bills", "BUF", 90, 75, 82), ("Vikings", "MIN", 76, 87, 82),
    ("Bengals", "CIN", 84, 78, 81), ("Bears", "CHI", 81, 80, 81), ("Packers", "GB", 79, 83, 81),
    ("Eagles", "PHI", 79, 82, 81), ("Steelers", "PIT", 76, 84, 80), ("Panthers", "CAR", 81, 78, 79),
    ("Cowboys", "DAL", 86, 72, 79), ("Falcons", "ATL", 79, 80, 79), ("Buccaneers", "TB", 77, 81, 79),
    ("Chargers", "LAC", 75, 82, 78), ("Commanders", "WAS", 79, 76, 77), ("Colts", "IND", 79, 75, 77),
    ("Giants", "NYG", 78, 77, 77), ("Raiders", "LV", 77, 77, 77), ("Jets", "NYJ", 75, 80, 77),
    ("Browns", "CLE", 73, 82, 77), ("Saints", "NO", 77, 73, 75), ("Cardinals", "ARI", 76, 71, 74),
    ("Titans", "TEN", 71, 75, 73), ("Dolphins", "MIA", 72, 71, 71),
]
CHANNELS = ("completion", "interception", "sack", "completion_yards", "run_yards")


@dataclass(frozen=True)
class Team:
    name: str
    abbr: str
    offense: int
    defense: int
    overall: int


@dataclass(frozen=True)
class Edge:
    completion: float = 0.0
    interception: float = 0.0
    sack: float = 0.0
    completion_tilt: float = 0.0
    run_tilt: float = 0.0


ZERO_EDGE = Edge()


def add_edges(a: Edge, b: Edge) -> Edge:
    return Edge(a.completion + b.completion, a.interception + b.interception, a.sack + b.sack,
                a.completion_tilt + b.completion_tilt, a.run_tilt + b.run_tilt)

ALL_TEAMS = [Team(*t) for t in TEAMS]


def find_team(text: str) -> Team:
    key = text.strip().lower()
    for t in ALL_TEAMS:
        if key in (t.name.lower(), t.abbr.lower()):
            return t
    raise ValueError(f"no team called {text!r}")


def logit_shift(p: float, delta: float) -> float:
    if delta == 0 or p <= 0.0 or p >= 1.0:
        return p
    x = math.log(p / (1.0 - p)) + delta
    return 1.0 / (1.0 + math.exp(-x))


TILT_STEP = 0.001


def pick_tilted(bank: list, tilt: float, rng, tables):
    if tilt == 0:
        return rng.choice(bank)
    step = round(tilt / TILT_STEP)
    if step == 0:
        return rng.choice(bank)
    cache = tables.__dict__.setdefault("_tilt_cache", {})
    key = (id(bank), step)
    cum = cache.get(key)
    if cum is None:
        t = step * TILT_STEP
        top = max(b[0] for b in bank) if t > 0 else min(b[0] for b in bank)
        total, cum = 0.0, []
        for b in bank:
            total += math.exp(t * (b[0] - top))
            cum.append(total)
        cache[key] = cum
    return bank[min(len(bank) - 1, bisect.bisect_left(cum, rng.random() * cum[-1]))]


class TeamModel:
    def __init__(self, path=DEFAULT_MODEL):
        self.params = json.loads(Path(path).read_text())
        offs = [t.offense for t in ALL_TEAMS]
        defs = [t.defense for t in ALL_TEAMS]
        self._stats = {"offense": _mean_sd(offs), "defense": _mean_sd(defs)}
        from engine import home_field
        self.home_points = home_field.rating_points()

    def z(self, team: Team, side: str) -> float:
        mean, sd = self._stats[side]
        return ((team.offense if side == "offense" else team.defense) - mean) / sd

    def points_edge(self, offense: Team, defense: Team, off_bonus: float = 0.0, def_bonus: float = 0.0) -> tuple:
        p = self.params
        return (p["points_per_rating_point_offense"] * (offense.offense + off_bonus - self._stats["offense"][0]),
                p["points_per_rating_point_defense"] * (defense.defense + def_bonus - self._stats["defense"][0]))

    def raw_shifts(self, offense: Team, defense: Team, off_bonus: float = 0.0, def_bonus: float = 0.0) -> dict:
        p = self.params
        add, take = self.points_edge(offense, defense, off_bonus, def_bonus)
        out = {}
        for ch in CHANNELS:
            c = p["channels"][ch]
            out[ch] = add * c["share_offense"] / c["sensitivity"] - take * c["share_defense"] / c["sensitivity"]
        return out

    def edge(self, offense: Team, defense: Team, off_bonus: float = 0.0, def_bonus: float = 0.0) -> Edge:
        centering = self.params.get("channel_centering", {})
        shifts = {ch: v + centering.get(ch, 0.0) for ch, v in self.raw_shifts(offense, defense, off_bonus, def_bonus).items()}
        return Edge(shifts["completion"], shifts["interception"], shifts["sack"],
                    shifts["completion_yards"], shifts["run_yards"])

    def edge_pair(self, team0: Team, team1: Team, home_field: bool = False) -> tuple:
        h = self.home_points if home_field else 0.0
        return self.edge(team0, team1, off_bonus=h), self.edge(team1, team0, def_bonus=h)


def _mean_sd(values):
    m = sum(values) / len(values)
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / len(values))
