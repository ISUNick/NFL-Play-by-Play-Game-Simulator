import multiprocessing as mp
import random
from collections import Counter
from dataclasses import dataclass, field

from engine.game import Game
from engine.league import ALL_ABBRS, CONFERENCES, DIVISIONS, initial_places, make_schedule
from engine.standings import GameResult, Standings, bracket_round
from engine.tables import Tables
from engine.teams import ALL_TEAMS, TeamModel, find_team

PLAYOFF_ROUNDS = ("wild card", "divisional", "conference", "super bowl")
REGULAR_WEEKS = 18


def play_game(tables, model, home, away, seed, week=1, round_name="regular", keep_log=False) -> GameResult:
    rng = random.Random(seed)
    home_field = round_name != "super bowl"
    edges = model.edge_pair(find_team(home), find_team(away), home_field=home_field)
    g = Game(tables, rng, names=(home, away), edges=edges, playoff=round_name != "regular", home_field=home_field)
    log = [] if keep_log else None
    while not (g.over and not g._queue):
        ev = g.step()
        if log is not None:
            log.append(f"{ev.scores[0]}-{ev.scores[1]} {ev.text}")
    return GameResult(week=week, home=home, away=away, home_score=g.scores[0], away_score=g.scores[1],
                      ot=g.went_to_ot, round=round_name, stats=[dict(g.stats[0]), dict(g.stats[1])], log=log)


# each worker process builds its own tables once
_worker = {}


def _init_worker(keep_log):
    _worker["tables"], _worker["model"], _worker["keep_log"] = Tables(), TeamModel(), keep_log


def _work(job):
    home, away, seed, week = job
    return play_game(_worker["tables"], _worker["model"], home, away, seed, week, "regular", _worker["keep_log"])


def game_seed(season_seed, year, week, home, away) -> str:
    # the seed only depends on who plays when, so results don't change with the number of workers
    return f"{season_seed}|{year}|{week}|{home}|{away}"


@dataclass
class Season:
    year: int
    seed: int
    weeks: list
    results: list
    places: dict
    seeds: dict
    champion: str = ""
    runner_up: str = ""
    division_winners: dict = field(default_factory=dict)

    @property
    def regular(self):
        return [r for r in self.results if r.round == "regular"]

    @property
    def playoffs(self):
        return [r for r in self.results if r.round != "regular"]

    def standings(self) -> Standings:
        return Standings(self.results, random.Random(f"tiebreak|{self.seed}|{self.year}"))


def run_season(year=0, places=None, seed=1, play=None, workers=1, keep_logs=False, tables=None,
               progress=None) -> Season:
    places = places or initial_places([t.abbr for t in ALL_TEAMS])
    weeks = make_schedule(year, places, random.Random(f"schedule|{seed}|{year}"))
    jobs = [(s.home, s.away, game_seed(seed, year, w + 1, s.home, s.away), w + 1)
            for w, wk in enumerate(weeks) for s in wk]

    # tests pass in a fake play function, the real engine is the default
    custom_play = play is not None
    if play is None:
        tables = tables or Tables()
        model = TeamModel()

        def play(home, away, sd, week, round_name):
            return play_game(tables, model, home, away, sd, week, round_name, keep_logs)

    regular = []
    if workers > 1 and not custom_play:
        with mp.Pool(workers, initializer=_init_worker, initargs=(keep_logs,)) as pool:
            for n, res in enumerate(pool.imap(_work, jobs, chunksize=8), 1):
                regular.append(res)
                if progress:
                    progress(n, len(jobs))
    else:
        for n, (home, away, sd, week) in enumerate(jobs, 1):
            regular.append(play(home, away, sd, week, "regular"))
            if progress:
                progress(n, len(jobs))

    standings = Standings(regular, random.Random(f"tiebreak|{seed}|{year}"))
    seeds = {conf: standings.seeds(conf) for conf in CONFERENCES}
    playoffs = run_playoffs(seeds, standings, play, seed, year)
    champion = playoffs[-1].winner
    runner_up = playoffs[-1].other(champion)
    winners = {d: standings.division_order(d)[0] for d in DIVISIONS}
    return Season(year, seed, [[(s.home, s.away, s.kind) for s in wk] for wk in weeks], regular + playoffs,
                  standings.places(), seeds, champion, runner_up, winners)


def round_pairs(round_name: str, alive: dict, standings: Standings) -> list:
    if round_name == "super bowl":
        afc, nfc = alive["AFC"][0], alive["NFC"][0]
        # neutral site, the better record is listed as home
        return [(nfc, afc) if standings.pct(nfc) > standings.pct(afc) else (afc, nfc)]
    return [pair for conf in CONFERENCES for pair in bracket_round(alive[conf], round_name)]


def advance_alive(round_name: str, alive: dict, results: list) -> dict:
    if round_name == "super bowl":
        winner = results[0].winner
        return {conf: [winner] if winner in teams else [] for conf, teams in alive.items()}
    new_alive = {}
    for conf in CONFERENCES:
        order = alive[conf]
        winners = [r.winner for r in results if r.home in order]
        if round_name == "wild card":
            # the 1 seed had the bye
            new_alive[conf] = [order[0]] + sorted(winners, key=order.index)
        else:
            new_alive[conf] = sorted(winners, key=order.index)
    return new_alive


def playoff_round(round_name: str, alive: dict, standings: Standings, play, seed, year, week) -> tuple:
    results = [play(hi, lo, game_seed(seed, year, week, hi, lo), week, round_name)
               for hi, lo in round_pairs(round_name, alive, standings)]
    return results, advance_alive(round_name, alive, results)


def run_playoffs(seeds: dict, standings: Standings, play, seed, year) -> list:
    out = []
    alive = {conf: list(s) for conf, s in seeds.items()}
    for n, round_name in enumerate(PLAYOFF_ROUNDS):
        results, alive = playoff_round(round_name, alive, standings, play, seed, year, REGULAR_WEEKS + 1 + n)
        out += results
    return out


TOTAL_KEYS = ("plays", "yards", "first_downs", "pass_att", "pass_comp", "pass_yards", "pass_td", "sack", "sack_yards",
              "rush_att", "rush_yards", "rush_td", "third_att", "third_conv", "fourth_att", "fourth_conv", "rz_trips",
              "rz_td", "big_plays", "top", "drives", "punt", "penalty", "penalty_yards", "interceptions_thrown",
              "fumbles_lost", "fg_att", "fg_made", "td")


def team_totals(results: list) -> dict:
    # per team: games, w, l, t, pf, pa, then every stat as "x" (for the team) and "opp_x" (against it)
    out = {t: Counter() for t in ALL_ABBRS}
    for r in results:
        if r.round != "regular":
            continue
        for me, i in ((r.home, 0), (r.away, 1)):
            c = out[me]
            mine, theirs = (r.home_score, r.away_score) if i == 0 else (r.away_score, r.home_score)
            c["games"] += 1
            c["w" if mine > theirs else "l" if mine < theirs else "t"] += 1
            c["pf"] += mine
            c["pa"] += theirs
            for k in TOTAL_KEYS:
                c[k] += r.stats[i].get(k, 0)
                c["opp_" + k] += r.stats[1 - i].get(k, 0)
    return out


def averages(c: Counter) -> dict:
    g = c["games"] or 1

    def ratio(a, b):
        return a / b if b else 0.0

    return {
        "pts": c["pf"] / g, "pts_allowed": c["pa"] / g,
        "yards": c["yards"] / g, "yards_allowed": c["opp_yards"] / g,
        "pass_yards": c["pass_yards"] / g, "rush_yards": c["rush_yards"] / g,
        "pass_yards_allowed": c["opp_pass_yards"] / g, "rush_yards_allowed": c["opp_rush_yards"] / g,
        "comp_pct": ratio(c["pass_comp"], c["pass_att"]), "third_pct": ratio(c["third_conv"], c["third_att"]),
        "turnovers": (c["interceptions_thrown"] + c["fumbles_lost"]) / g,
        "takeaways": (c["opp_interceptions_thrown"] + c["opp_fumbles_lost"]) / g,
        "sacks_taken": c["sack"] / g, "sacks": c["opp_sack"] / g,
        "penalty_yards": c["penalty_yards"] / g, "top": c["top"] / g,
        "first_downs": c["first_downs"] / g, "rz_pct": ratio(c["rz_td"], c["rz_trips"]),
    }


def team_table(results: list) -> list:
    totals = team_totals(results)
    rows = [(t, c["w"], c["l"], c["t"], c["pf"], c["pa"], averages(c)) for t, c in totals.items()]
    rows.sort(key=lambda r: (-(r[1] + 0.5 * r[3]), -(r[4] - r[5])))
    return rows
