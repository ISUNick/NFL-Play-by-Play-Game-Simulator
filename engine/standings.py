import random
from dataclasses import dataclass, field

from engine.league import ALL_ABBRS, CONFERENCE_OF, CONFERENCES, DIVISION_OF, DIVISIONS


@dataclass
class GameResult:
    week: int
    home: str
    away: str
    home_score: int
    away_score: int
    ot: bool = False
    round: str = "regular"
    stats: list = field(default_factory=lambda: [{}, {}])
    log: list = None

    @property
    def winner(self):
        if self.home_score == self.away_score:
            return None
        return self.home if self.home_score > self.away_score else self.away

    def score_of(self, team: str) -> int:
        return self.home_score if team == self.home else self.away_score

    def other(self, team: str) -> str:
        return self.away if team == self.home else self.home


class Record:
    __slots__ = ("w", "l", "t", "pf", "pa")

    def __init__(self):
        self.w = self.l = self.t = self.pf = self.pa = 0

    def add(self, mine: int, theirs: int):
        self.pf += mine
        self.pa += theirs
        if mine > theirs:
            self.w += 1
        elif mine < theirs:
            self.l += 1
        else:
            self.t += 1

    @property
    def games(self) -> int:
        return self.w + self.l + self.t

    @property
    def pct(self) -> float:
        return (self.w + 0.5 * self.t) / self.games if self.games else 0.0

    def __str__(self):
        return f"{self.w}-{self.l}" + (f"-{self.t}" if self.t else "")


def regular_only(results: list) -> list:
    return [r for r in results if r.round == "regular"]


class Standings:
    def __init__(self, results: list, rng: random.Random = None):
        self.rng = rng or random.Random(0)
        self.results = regular_only(results)
        self.teams = sorted(set(ALL_ABBRS) | {t for r in self.results for t in (r.home, r.away)})
        self.games = {t: [] for t in self.teams}
        for r in self.results:
            self.games[r.home].append((r.away, r.home_score, r.away_score))
            self.games[r.away].append((r.home, r.away_score, r.home_score))
        self.overall = {t: self._record(t) for t in self.teams}
        self._coin = {t: self.rng.random() for t in self.teams}

    def _record(self, team: str, only=None) -> Record:
        rec = Record()
        for opp, mine, theirs in self.games[team]:
            if only is None or only(opp):
                rec.add(mine, theirs)
        return rec

    def division_record(self, team):
        return self._record(team, lambda o: DIVISION_OF[o] == DIVISION_OF[team])

    def conference_record(self, team):
        return self._record(team, lambda o: CONFERENCE_OF[o] == CONFERENCE_OF[team])

    def record_vs(self, team, group):
        group = set(group)
        return self._record(team, lambda o: o in group)

    def pct(self, team) -> float:
        return self.overall[team].pct

    def net_points(self, team) -> int:
        r = self.overall[team]
        return r.pf - r.pa

    def strength_of_victory(self, team) -> float:
        opps = [o for o, m, t in self.games[team] if m > t]
        return self._combined(opps)

    def strength_of_schedule(self, team) -> float:
        return self._combined([o for o, m, t in self.games[team]])

    def _combined(self, opps) -> float:
        w = sum(self.overall[o].w + 0.5 * self.overall[o].t for o in opps)
        g = sum(self.overall[o].games for o in opps)
        return w / g if g else 0.0

    def common_games(self, team, group):
        sets = [{o for o, m, t in self.games[x]} for x in group]
        common = set.intersection(*sets) - set(group)
        if not common:
            return None
        rec = self._record(team, lambda o: o in common)
        return rec if rec.games >= 4 else None

    def _head_to_head(self, group, need_complete: bool):
        group = list(group)
        recs = {t: self.record_vs(t, [o for o in group if o != t]) for t in group}
        if need_complete:
            for t in group:
                played = {o for o, m, th in self.games[t]}
                if any(o not in played for o in group if o != t):
                    return None
        if all(recs[t].games == 0 for t in group):
            return None
        return {t: recs[t].pct for t in group}

    def order(self, teams, kind: str) -> list:
        teams = list(teams)
        by_pct = {}
        for t in teams:
            by_pct.setdefault(round(self.pct(t), 9), []).append(t)
        out = []
        for key in sorted(by_pct, reverse=True):
            out += self.break_tie(by_pct[key], kind)
        return out

    def break_tie(self, group, kind: str) -> list:
        group = list(group)
        if len(group) == 1:
            return group
        if kind == "wild card":
            return self._wild_card_tie(group)
        return self._tie(group, kind)

    def _tie(self, group, kind):
        for step in self._steps(kind):
            values = step(group)
            if values is None:
                continue
            groups = {}
            for t in group:
                groups.setdefault(round(values[t], 9), []).append(t)
            if len(groups) == 1:
                continue
            out = []
            for key in sorted(groups, reverse=True):
                sub = groups[key]
                out += self._tie(sub, kind) if len(sub) > 1 else sub
            return out
        return sorted(group, key=lambda t: self._coin[t])

    def _steps(self, kind):
        def h2h(need_complete):
            return lambda g: self._head_to_head(g, need_complete)

        def per_team(fn):
            return lambda g: {t: fn(t) for t in g}

        def common(g):
            recs = {t: self.common_games(t, g) for t in g}
            return None if any(r is None for r in recs.values()) else {t: r.pct for t, r in recs.items()}

        division = per_team(lambda t: self.division_record(t).pct)
        conference = per_team(lambda t: self.conference_record(t).pct)
        sov = per_team(self.strength_of_victory)
        sos = per_team(self.strength_of_schedule)
        net = per_team(self.net_points)
        if kind == "division":
            return [h2h(False), division, common, conference, sov, sos, net]
        return [h2h(True), conference, common, sov, sos, net]

    def _wild_card_tie(self, group):
        left = list(group)
        out = []
        while left:
            reps = []
            for d in {DIVISION_OF[t] for t in left}:
                same = [t for t in left if DIVISION_OF[t] == d]
                reps.append(self._tie(same, "division")[0] if len(same) > 1 else same[0])
            best = self._tie(reps, "wild card")[0] if len(reps) > 1 else reps[0]
            out.append(best)
            left.remove(best)
        return out

    def division_order(self, division: str) -> list:
        return self.order(DIVISIONS[division], "division")

    def places(self) -> dict:
        out = {}
        for d in DIVISIONS:
            for place, t in enumerate(self.division_order(d), 1):
                out[t] = place
        return out

    def seeds(self, conference: str) -> list:
        winners = [self.division_order(d)[0] for d in CONFERENCES[conference]]
        seeded = self.order(winners, "wild card")
        rest = [t for d in CONFERENCES[conference] for t in DIVISIONS[d] if t not in winners]
        wild = self.order(rest, "wild card")[:3]
        return seeded + wild


def bracket_round(seeds: list, round_name: str) -> list:
    if round_name == "wild card":
        return [(seeds[1], seeds[6]), (seeds[2], seeds[5]), (seeds[3], seeds[4])]
    if round_name == "divisional":
        return [(seeds[0], seeds[3]), (seeds[1], seeds[2])]
    return [(seeds[0], seeds[1])]
