import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine.teams import Edge, Team, TeamModel, add_edges

PLAYER_DIR = Path(__file__).resolve().parent.parent / "data" / "players"
DEPTHS = ("short", "medium", "deep")
OUT_STATUSES = ("Out", "Doubtful")


@dataclass
class Ratings:
    qb: pd.DataFrame
    receiver: pd.DataFrame
    runner: pd.DataFrame
    team: pd.DataFrame
    info: dict

    @classmethod
    def load(cls, folder=PLAYER_DIR):
        folder = Path(folder)
        read = lambda name: pd.read_csv(folder / name, index_col=0)
        return cls(read("qb_ratings.csv"), read("receiver_ratings.csv"), read("runner_ratings.csv"),
                   read("team_units.csv"), json.loads((folder / "priors.json").read_text()))

    def scaled(self, team_scale: float = 1.0, player_scale: float = 1.0, qb_scale: float = None):
        # stretch or shrink every effect. Ratings are pulled toward average, so a factor above 1 undoes part of that
        qb, receiver, runner, team = self.qb.copy(), self.receiver.copy(), self.runner.copy(), self.team.copy()
        qb_scale = player_scale if qb_scale is None else qb_scale
        for df, cols, k in ((qb, ["comp_shift", "int_shift", "sack_shift", "scramble_shift", "yards_effect"], qb_scale),
                            (receiver, ["comp_shift", "yards_effect"], player_scale), (runner, ["yards_over_expected"], player_scale)):
            df[cols] = df[cols] * k
        team[team.columns] = team * team_scale
        return Ratings(qb, receiver, runner, team, self.info)

    @classmethod
    def from_built(cls, built):
        return cls(built["tables"]["qb"], built["tables"]["receiver"], built["tables"]["runner"],
                   built["tables"]["team"], built["info"])


@dataclass
class QB:
    name: str
    comp_shift: float = 0.0
    int_shift: float = 0.0
    sack_shift: float = 0.0
    scramble_shift: float = 0.0
    yards_effect: float = 0.0


@dataclass
class Receiver:
    name: str
    position: str
    weight: float
    depth_share: dict
    comp_shift: float = 0.0
    yards_effect: float = 0.0


@dataclass
class Runner:
    name: str
    weight: float
    yards_effect: float = 0.0


@dataclass
class Offense:
    qb: QB
    receivers: list
    runners: list
    league_mix: dict
    comp_variance: float
    run_variance: float
    roster: list = field(default_factory=list)
    _receiver_cum: list = field(default_factory=list)
    _runner_cum: list = field(default_factory=list)

    def __post_init__(self):
        self._receiver_cum = _cumulative([r.weight for r in self.receivers])
        self._runner_cum = _cumulative([r.weight for r in self.runners])

    def pick_receiver(self, rng) -> Receiver:
        return self.receivers[_draw(self._receiver_cum, rng)]

    def pick_runner(self, rng) -> Runner:
        return self.runners[_draw(self._runner_cum, rng)]


def _cumulative(weights):
    total, out = 0.0, []
    for w in weights:
        total += w
        out.append(total)
    return out


def _draw(cum, rng) -> int:
    return min(len(cum) - 1, bisect.bisect_left(cum, rng.random() * cum[-1]))


def _available_fractions(group: pd.DataFrame) -> pd.Series:
    # the group usually has about `slots` players on the field. If someone is out, the rest cover his snaps
    slots = group.snap_share.sum()
    here = group[~group.out]
    if here.snap_share.sum() <= 0:
        return pd.Series(0.0, index=group.index)
    scaled = (here.snap_share * slots / here.snap_share.sum()).clip(upper=1.0)
    return scaled.reindex(group.index).fillna(0.0)


def build_offense(team: str, lineup: pd.DataFrame, ratings: Ratings, out_names=(), play_names=()) -> Offense:
    # lineup is the table from build_lineup_inputs. out_names adds players the user marks out,
    # play_names puts players the injury report lists as out back on the field
    info = ratings.info
    t = lineup[lineup.team == team].set_index("gsis_id", drop=False)
    t["out"] = (t.out & ~t.full_name.isin(play_names)) | t.full_name.isin(out_names)

    qbs = t[(t.position == "QB") & ~t.out].sort_values("snap_share", ascending=False)
    if len(qbs) == 0:
        raise ValueError(f"{team} has no quarterback available")
    top = qbs.iloc[0]
    row = ratings.qb.loc[top.gsis_id] if top.gsis_id in ratings.qb.index else None
    qb = QB(top.full_name) if row is None else QB(top.full_name, row.comp_shift, row.int_shift, row.sack_shift,
                                                   row.scramble_shift, row.yards_effect)

    league_mix = {g: {d: info["depth_mix"][g]["league_mix"][d] for d in DEPTHS} for g in ("WR", "TE", "RB")}
    receivers, runners = [], []
    for pos in ("WR", "TE", "RB"):
        group = t[(t.position == pos) & (t.snap_share > 0.03)]
        fractions = _available_fractions(group)
        prior_rate = info["target_rate"][pos]["league_rate"]
        for pid, f in fractions.items():
            if f <= 0:
                continue
            r = ratings.receiver.loc[pid] if pid in ratings.receiver.index else None
            rate = prior_rate if r is None else r.target_rate
            share = league_mix[pos] if r is None else {d: r[f"share_{d}"] for d in DEPTHS}
            receivers.append(Receiver(group.full_name[pid], pos, rate * f, share,
                                      0.0 if r is None else r.comp_shift, 0.0 if r is None else r.yards_effect))
    carriers = t[t.position.isin(["QB", "RB", "WR", "TE"]) & (t.snap_share > 0.03)]
    for pos, group in carriers.groupby("position"):
        fractions = _available_fractions(group) if pos != "QB" else (group.snap_share * (group.index == top.name))
        prior_rate = info["carry_rate"].get(pos, {}).get("league_rate", 0.0)
        for pid, f in fractions.items():
            if f <= 0:
                continue
            r = ratings.runner.loc[pid] if pid in ratings.runner.index else None
            rate = prior_rate if r is None or pd.isna(r.carry_rate) else r.carry_rate
            if rate > 0:
                runners.append(Runner(group.full_name[pid], rate * f, 0.0 if r is None else r.yards_over_expected))
    if not receivers:
        raise ValueError(f"{team} has no receivers available")
    if not runners:
        raise ValueError(f"{team} has no runners available")
    skill = t[t.position.isin(["QB", "RB", "WR", "TE"]) & ~t.out]
    skill = skill[(skill.snap_share > 0.03) | (skill.position == "QB")]
    skill = skill.assign(order=skill.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3}))
    skill = skill.sort_values(["order", "snap_share"], ascending=[True, False])
    qb_rows = skill[skill.position == "QB"].head(3)
    skill = pd.concat([qb_rows, skill[skill.position != "QB"]])
    roster = [(r.full_name, r.position) for r in skill.itertuples()]
    return Offense(qb, receivers, runners, league_mix, info["completion_yards"]["yards_variance"],
                   info["run_yards"]["yards_variance"], roster)


def team_edge(ratings: Ratings, offense: str, defense: str, line: str = None) -> Edge:
    # unit strength of the offense (line, scheme) plus the defense it faces, all in the engine's own units.
    # line swaps in another team's blocking: how many yards its backs get and how often its passer is sacked
    o, d = ratings.team.loc[offense], ratings.team.loc[defense]
    blockers = o if line is None else ratings.team.loc[line]
    info = ratings.info
    return Edge(o.off_completion + d.def_completion, o.off_interception + d.def_interception,
                blockers.off_sack + d.def_sack,
                (o.off_comp_yards + d.def_comp_yards) / info["completion_yards"]["yards_variance"],
                (blockers.off_run_yards + d.def_run_yards) / info["run_yards"]["yards_variance"])


def line_grades(ratings: Ratings) -> list:
    # every team's offensive line as the engine sees it, best first. run is yards per carry over average,
    # sack is the change in sack rate in points (negative is better)
    t = ratings.team
    p = ratings.info["sack"]["league_rate"]
    score = (t.off_run_yards - t.off_run_yards.mean()) / t.off_run_yards.std() - (t.off_sack - t.off_sack.mean()) / t.off_sack.std()
    rows = [{"team": team, "run": float(t.off_run_yards[team]), "sack": float(100 * p * (1 - p) * t.off_sack[team]), "score": float(score[team])}
            for team in t.index]
    rows.sort(key=lambda r: -r["score"])
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return rows


def _difference(a: Edge, b: Edge) -> Edge:
    return Edge(a.completion - b.completion, a.interception - b.interception, a.sack - b.sack,
                a.completion_tilt - b.completion_tilt, a.run_tilt - b.run_tilt)


def home_bonus(model: TeamModel) -> tuple:
    # what the engine already knows about home field, as the change from a neutral game: (home offense, away offense)
    mean = Team("AVG", "AVG", 80, 80, 80)
    plain = model.edge(mean, mean)
    h = model.home_points
    return _difference(model.edge(mean, mean, off_bonus=h), plain), _difference(model.edge(mean, mean, def_bonus=h), plain)


def matchup(home: str, away: str, lineup: pd.DataFrame, ratings: Ratings, model: TeamModel,
            out_home=(), out_away=(), home_field=True, play_home=(), play_away=(), line_home=None, line_away=None) -> tuple:
    # returns (edges, units) ready to hand to Game
    bonus_home, bonus_away = home_bonus(model) if home_field else (Edge(), Edge())
    edges = (add_edges(team_edge(ratings, home, away, line_home), bonus_home),
             add_edges(team_edge(ratings, away, home, line_away), bonus_away))
    units = (build_offense(home, lineup, ratings, out_home, play_home), build_offense(away, lineup, ratings, out_away, play_away))
    return edges, units
