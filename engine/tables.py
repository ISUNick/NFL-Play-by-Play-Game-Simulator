import csv
import json
import math
from pathlib import Path

from engine.teams import Edge

FG_TAPER_START = 60
FG_TAPER_LINEAR = 0.058
FG_TAPER_QUAD = 0.0292
FINAL_EDGES = [0, 5, 10, 20, 30, 45, 60]
ZONE_EDGES = [0, 10, 30, 38, 50]
HURRY_KIND = {"run_in": "run", "scramble_in": "run", "complete_in": "complete", "sack": "sack"}
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "tables"

LEVEL_TABLES = ["play_call", "play_call_fourth", "play_call_late", "dropback_split", "pass_depth", "pass_result",
                "pass_depth_fourth", "pass_result_fourth", "pass_depth_third", "pass_result_third", "try_choice",
                "fourth_down", "fourth_down_late", "onside_choice", "half_snap", "half_timeout_off",
                "half_timeout_def", "fourth_down_ot", "penalty_pre", "penalty_post",
                "late_timeout_off", "late_timeout_def", "half_spike"]


class Tables:
    def __init__(self, table_dir=DEFAULT_DIR, pace: float = None):
        self.dir = Path(table_dir)
        era = self._read_era()
        self.pace = pace if pace is not None else era.get("pace", 1.0)
        self.era_edge = Edge(era.get("completion", 0.0), era.get("interception", 0.0), era.get("sack", 0.0),
                             era.get("completion_tilt", 0.0), era.get("run_tilt", 0.0)) if era else None
        specs = json.loads((self.dir / "table_specs.json").read_text())

        self._levels = {}
        self._index = {}
        for name in LEVEL_TABLES:
            self._load_level_table(name, specs[name])

        self.completion = self._bank("bank_completion", lambda r: ((r[0], int(r[1])), (int(r[2]), int(r[3]), int(r[4]))))
        self.run = self._bank("bank_run", lambda r: ((r[0], int(r[1])), (int(r[2]), int(r[3]), int(r[4]))))
        self.scramble = self._bank("bank_scramble", lambda r: (int(r[1]), (int(r[2]), int(r[3]), int(r[4]))))
        self.sack = self._bank("bank_sack", lambda r: (int(r[1]), (int(r[2]), int(r[3]), int(r[4]))))
        self.interception = self._bank("bank_int", lambda r: (r[0], (int(r[1]), int(r[2]), int(r[3]))))
        self.time = self._bank("bank_time", lambda r: ((r[0], r[1], r[2], r[3]), int(r[4])))
        self.punt = self._bank("bank_punt", lambda r: (int(r[0]), (
            int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7]))))
        self.timeout_clock = {}
        for r in self._dict_rows("bank_timeout"):
            self.timeout_clock.setdefault(r["tempo"], []).append((int(r["d1"]), int(r["d2"])))
        early = next(self._dict_rows("timeout_early"))
        self.timeout_early = {"offense": float(early["p_off"]), "defense": float(early["p_def"])}
        self.penalty = {}
        for r in self._dict_rows("bank_penalty"):
            item = (int(r["dy"]), r["rel"] == "first", int(r["seconds"]), r["foul"])
            for key in ((r["side"], r["kind"], r["zone_b"]), (r["side"], r["kind"]), (r["side"],)):
                self.penalty.setdefault(key, []).append(item)
        self.spike_runoff = [int(r[0]) for r in self._rows("bank_spike_runoff")]
        self.kickoff = [(r[0], int(r[1]), int(r[2])) for r in self._rows("bank_kickoff")]
        self.free_kick = [(r[0], int(r[1]), int(r[2])) for r in self._rows("bank_free_kick")]
        self.onside = [(int(r[0]), int(r[1])) for r in self._rows("bank_onside")]

        self.final_calls = {}
        if (self.dir / "play_call_final.csv").exists():
            self.final_calls = {(int(r["time_f"]), int(r["trail_f"])): float(r["p_dropback"]) for r in self._dict_rows("play_call_final")}

        self.half_end_fourth = {}
        if (self.dir / "fourth_half_end.csv").exists():
            self.half_end_fourth = {(int(r["near"]), int(r["zone"])): (float(r["p_fg"]), float(r["p_go"]))
                                    for r in self._dict_rows("fourth_half_end")}

        self.hurry = {}
        if (self.dir / "bank_hurry.csv").exists():
            for r in self._dict_rows("bank_hurry"):
                self.hurry.setdefault((r["kind"], int(r["has_to"])), []).append(int(r["seconds"]))

        xp = next(self._dict_rows("extra_point"))
        self.extra_point = {k: float(v) for k, v in xp.items()}
        two = next(self._dict_rows("two_point"))
        self.two_point = {k: float(v) for k, v in two.items()}
        self.fumble_lost = {r["kind"]: float(r["p_fumble_lost"]) for r in self._dict_rows("fumble_lost")}
        self._fg = {int(r["kick_distance"]): (float(r["make_prob"]), float(r["make_prob_rushed"]))
                    for r in self._dict_rows("field_goal")}
        self._fg_min, self._fg_max = min(self._fg), max(self._fg)

    def _rows(self, name):
        with open(self.dir / f"{name}.csv", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            yield from reader

    def _dict_rows(self, name):
        with open(self.dir / f"{name}.csv", newline="") as f:
            yield from csv.DictReader(f)

    def _bank(self, name, parse) -> dict:
        bank = {}
        for row in self._rows(name):
            key, value = parse(row)
            bank.setdefault(key, []).append(value)
        return bank

    def _load_level_table(self, name, spec):
        levels = [(level, keys) for level, keys in spec["levels"]]
        keys_of = dict(levels)
        outcomes = spec["outcomes"]
        index = {}
        for row in self._dict_rows(spec["file"].removesuffix(".csv")):
            level = row["level"]
            key = (level, tuple(row[k] for k in keys_of[level]))
            values = {o: float(row[o]) for o in outcomes}
            values["n"] = int(float(row["n"]))
            values["level"] = level
            index[key] = values
        self._levels[name] = levels
        self._index[name] = index

    def lookup(self, name: str, situation: dict):
        index = self._index[name]
        for level, keys in self._levels[name]:
            row = index.get((level, tuple(str(situation[k]) for k in keys)))
            if row is not None:
                return level, row
        raise KeyError(f"no row in {name} for {situation}")

    def lookup_or_none(self, name: str, situation: dict):
        index = self._index[name]
        for level, keys in self._levels[name]:
            row = index.get((level, tuple(str(situation[k]) for k in keys)))
            if row is not None:
                return level, row
        return None

    def final_minute_dropback(self, state):
        # a trailing offense in the last minute throws almost every snap. None when this is not that spot
        left = state.game_seconds_remaining
        if state.quarter != 4 or state.score_diff >= 0 or not 0 < left <= 60 or not self.final_calls:
            return None
        time_f = next(i for i, edge in enumerate(FINAL_EDGES[1:]) if left <= edge)
        return self.final_calls.get((time_f, int(state.score_diff < -8)))

    def half_end_fourth_call(self, state):
        # (p_field_goal, p_go) for a fourth down in the last seconds of the first half, None in any other spot
        left = state.half_seconds_remaining
        if state.quarter != 2 or not 0 < left <= 12 or state.yardline_100 > 50 or not self.half_end_fourth:
            return None
        zone = next(i for i, edge in enumerate(ZONE_EDGES[1:]) if state.yardline_100 <= edge)
        return self.half_end_fourth.get((int(left <= 5), zone))

    def hurry_seconds(self, state, timing_kind: str, rng):
        # snap to snap time of a real two minute drill: 4th quarter, down 1 to 8, clock kept running.
        # None anywhere else, then the normal timing bank decides
        kind = HURRY_KIND.get(timing_kind)
        if kind is None or state.quarter != 4 or not -8 <= state.score_diff <= -1 or not 0 < state.clock <= 120:
            return None
        bank = self.hurry.get((kind, int(state.offense_timeouts > 0)))
        return rng.choice(bank) if bank else None

    def lookup_late(self, late_name: str, name: str, state):
        late = state.late_situation()
        if late["late_time"] != "none":
            index = self._index[late_name]
            for level, keys in self._levels[late_name]:
                row = index.get((level, tuple(str(late[k]) for k in keys)))
                if row is not None:
                    return f"late_{level}", row
        sit = state.fourth_down_situation() if name == "fourth_down" else state.situation()
        return self.lookup(name, sit)

    def _read_era(self) -> dict:
        # the league wide dials (pace and the five edges) that make an average game match the real season
        path = self.dir / "era.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def draw_seconds(self, kind: str, tempo: str, score_b: str, rng) -> int:
        for key in (("L1", kind, tempo, score_b), ("L2", kind, tempo, "ALL"), ("L3", kind, "ALL", "ALL")):
            values = self.time.get(key)
            if values:
                seconds = rng.choice(values)
                if self.pace == 1.0:
                    return seconds
                stretched = seconds * self.pace
                return int(stretched) + (rng.random() < stretched - int(stretched))
        raise KeyError(f"no timing for {kind}, {tempo}, {score_b}")

    def fg_make_prob(self, kick_distance: int, rushed: bool = False) -> float:
        d = max(self._fg_min, min(self._fg_max, kick_distance))
        normal, hurried = self._fg[d]
        p = hurried if rushed else normal
        if d > FG_TAPER_START:
            p *= self._taper(d)
        return p

    @staticmethod
    def _taper(d: int) -> float:
        x = d - FG_TAPER_START
        return math.exp(-FG_TAPER_LINEAR * x - FG_TAPER_QUAD * x * x)
