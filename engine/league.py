import random
from dataclasses import dataclass

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SF", "SEA"],
}
CONFERENCES = {"AFC": ["AFC East", "AFC North", "AFC South", "AFC West"],
               "NFC": ["NFC East", "NFC North", "NFC South", "NFC West"]}


DIVISION_OF = {t: d for d, ts in DIVISIONS.items() for t in ts}
CONFERENCE_OF = {t: d[:3] for t, d in DIVISION_OF.items()}
ALL_ABBRS = [t for ts in DIVISIONS.values() for t in ts]

WEEKS = 18
BYE_WEEKS = range(4, 14)
PAIRINGS = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]


@dataclass(frozen=True)
class Slot:
    home: str
    away: str
    kind: str


def _latin_host(i: int, j: int, shift: int) -> bool:
    return (i + j + shift) % 4 < 2


def opponents(year: int, places: dict) -> list:
    games = []
    # division rivals, home and away
    for ts in DIVISIONS.values():
        for a in ts:
            for b in ts:
                if a != b:
                    games.append(Slot(a, b, "division"))
    for conf, divs in CONFERENCES.items():
        # play one division from the same conference, rotates every 3 years
        pairing = PAIRINGS[year % 3]
        for x, y in pairing:
            for i, a in enumerate(DIVISIONS[divs[x]]):
                for j, b in enumerate(DIVISIONS[divs[y]]):
                    host = _latin_host(i, j, year)
                    games.append(Slot(a, b, "conference") if host else Slot(b, a, "conference"))
        # the other two divisions: same finishing place only
        (a_, b_), (c_, d_) = pairing
        cycle = [(a_, c_), (c_, b_), (b_, d_), (d_, a_)]
        if year % 2:
            cycle = [(v, h) for h, v in cycle]
        for h, v in cycle:
            for place in (1, 2, 3, 4):
                home = _team_in_place(DIVISIONS[divs[h]], place, places)
                away = _team_in_place(DIVISIONS[divs[v]], place, places)
                games.append(Slot(home, away, "place"))
    for i, afc in enumerate(CONFERENCES["AFC"]):
        # one division from the other conference, rotates every 4 years
        nfc = CONFERENCES["NFC"][(i + year) % 4]
        for p, a in enumerate(DIVISIONS[afc]):
            for q, b in enumerate(DIVISIONS[nfc]):
                host = _latin_host(p, q, year // 4)
                games.append(Slot(a, b, "interconference") if host else Slot(b, a, "interconference"))
        # the 17th game, another interconference division at the same place
        nfc17 = CONFERENCES["NFC"][(i + year + 1) % 4]
        for place in (1, 2, 3, 4):
            a = _team_in_place(DIVISIONS[afc], place, places)
            b = _team_in_place(DIVISIONS[nfc17], place, places)
            games.append(Slot(a, b, "seventeenth") if year % 2 == 0 else Slot(b, a, "seventeenth"))
    return games


def _team_in_place(division_teams: list, place: int, places: dict) -> str:
    for t in division_teams:
        if places[t] == place:
            return t
    raise ValueError(f"nobody finished {place} in {division_teams}")


def initial_places(ratings_order: list = None) -> dict:
    order = ratings_order or ALL_ABBRS
    rank = {t: n for n, t in enumerate(order)}
    out = {}
    for ts in DIVISIONS.values():
        for place, t in enumerate(sorted(ts, key=lambda t: rank.get(t, 99)), 1):
            out[t] = place
    return out


class ScheduleError(Exception):
    pass


def _assign_byes(rng: random.Random) -> dict:
    # 2 teams per bye week plus 2 extra in 6 random weeks = 32
    weeks = list(BYE_WEEKS)
    counts = {w: 2 for w in weeks}
    for w in rng.sample(weeks, 6):
        counts[w] += 2
    slots = [w for w in weeks for _ in range(counts[w])]
    teams = ALL_ABBRS[:]
    rng.shuffle(teams)
    return dict(zip(teams, slots))


def _fill_weeks(games: list, byes: dict, rng: random.Random):
    remaining = {}
    for g in games:
        remaining.setdefault(frozenset((g.home, g.away)), []).append(g)
    weeks = [[] for _ in range(WEEKS)]

    # week 18 is all division games
    for ts in DIVISIONS.values():
        order = ts[:]
        rng.shuffle(order)
        for a, b in ((order[0], order[1]), (order[2], order[3])):
            pair = remaining[frozenset((a, b))]
            weeks[WEEKS - 1].append(pair.pop(rng.randrange(len(pair))))

    left = {t: [] for t in ALL_ABBRS}
    for pair, lst in remaining.items():
        a, b = tuple(pair)
        for _ in lst:
            left[a].append(b)
            left[b].append(a)

    def take(a, b):
        left[a].remove(b)
        left[b].remove(a)
        lst = remaining[frozenset((a, b))]
        return lst.pop(rng.randrange(len(lst)))

    for w in range(WEEKS - 1):
        active = [t for t in ALL_ABBRS if byes[t] != w]
        for attempt in range(60):
            free = set(active)
            chosen = []
            ok = True
            while free and ok:
                # sorted, so the same seed gives the same schedule in every run (sets change order between runs)
                options = {t: sorted(o for o in set(left[t]) if o in free) for t in sorted(free)}
                t = min(options, key=lambda x: (len(options[x]), rng.random()))
                if not options[t]:
                    ok = False
                    break
                opp = max(options[t], key=lambda o: (left[t].count(o), rng.random()))
                chosen.append((t, opp))
                free.discard(t)
                free.discard(opp)
            if ok:
                for a, b in chosen:
                    weeks[w].append(take(a, b))
                break
        else:
            return None
    if any(remaining[p] for p in remaining):
        return None
    return weeks


def make_schedule(year: int, places: dict, rng: random.Random) -> list:
    games = opponents(year, places)
    for _ in range(200):
        byes = _assign_byes(rng)
        weeks = _fill_weeks(games, byes, rng)
        if weeks:
            return weeks
    raise ScheduleError("could not build a schedule")
