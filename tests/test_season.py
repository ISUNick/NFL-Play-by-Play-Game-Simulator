import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.game import Game
from engine.league import (ALL_ABBRS, CONFERENCE_OF, CONFERENCES, DIVISION_OF, DIVISIONS, initial_places,
                           make_schedule)


from engine.season import run_season, team_table, team_totals
from engine.standings import GameResult, Standings, bracket_round
from engine.teams import ALL_TEAMS
from test_engine_tables import tables as make_tables

RANK = {t.abbr: n for n, t in enumerate(ALL_TEAMS)}


def fake_play(home, away, seed, week, round_name):
    hs, as_ = (24, 17) if RANK[home] < RANK[away] else (17, 24)
    stat = {"yards": 300, "plays": 60, "pass_yards": 200, "rush_yards": 100, "pass_att": 30, "pass_comp": 20,
            "third_att": 10, "third_conv": 4, "top": 1800, "interceptions_thrown": 1, "sack": 2}
    return GameResult(week, home, away, hs, as_, False, round_name, [dict(stat), dict(stat)])


def r(week, home, away, hs, as_):
    return GameResult(week, home, away, hs, as_)


def test_league_matches_the_team_list():
    assert sorted(ALL_ABBRS) == sorted(t.abbr for t in ALL_TEAMS)
    assert len(DIVISIONS) == 8 and all(len(v) == 4 for v in DIVISIONS.values())
    assert Counter(CONFERENCE_OF.values()) == {"AFC": 16, "NFC": 16}


def check_schedule(weeks, year):
    games = [g for wk in weeks for g in wk]
    assert len(weeks) == 18 and len(games) == 272, (len(weeks), len(games))
    games_of, home_of, opps, byes = Counter(), Counter(), {t: Counter() for t in ALL_ABBRS}, {}
    for w, wk in enumerate(weeks):
        seen = set()
        for g in wk:
            assert g.home != g.away and g.home not in seen and g.away not in seen, (year, w, g)
            seen |= {g.home, g.away}
            games_of[g.home] += 1
            games_of[g.away] += 1
            home_of[g.home] += 1
            opps[g.home][g.away] += 1
            opps[g.away][g.home] += 1
        for t in set(ALL_ABBRS) - seen:
            assert t not in byes, f"{t} has two byes"
            byes[t] = w
    assert all(games_of[t] == 17 for t in ALL_ABBRS) and len(byes) == 32
    assert all(4 <= b <= 13 for b in byes.values()), "byes are in weeks 5 to 14"
    assert all(home_of[t] in (8, 9) for t in ALL_ABBRS), home_of
    assert sum(1 for t in ALL_ABBRS if home_of[t] == 9) == 16, "half the league hosts 9 games"
    for t in ALL_ABBRS:
        mine = DIVISION_OF[t]
        div = [o for o in opps[t] if DIVISION_OF[o] == mine]
        assert sorted(opps[t][o] for o in div) == [2, 2, 2], "each division rival twice"
        assert all(n == 1 for o, n in opps[t].items() if DIVISION_OF[o] != mine)
        by_div = Counter(DIVISION_OF[o] for o in opps[t] if DIVISION_OF[o] != mine)
        same_conf = {d: n for d, n in by_div.items() if d[:3] == mine[:3]}
        other_conf = {d: n for d, n in by_div.items() if d[:3] != mine[:3]}
        assert sorted(same_conf.values()) == [1, 1, 4], (t, same_conf)
        assert sorted(other_conf.values()) == [1, 4], (t, other_conf)
    assert all(DIVISION_OF[g.home] == DIVISION_OF[g.away] for g in weeks[17]), "week 18 is all division games"
    assert len(weeks[17]) == 16


def test_schedule_is_valid_for_many_years_and_random_places():
    for year in range(15):
        rng = random.Random(year * 31)
        places = {}
        for ts in DIVISIONS.values():
            order = ts[:]
            rng.shuffle(order)
            places.update({t: p for p, t in enumerate(order, 1)})
        check_schedule(make_schedule(year, places, rng), year)


def test_schedule_is_repeatable_and_changes_by_year():
    places = initial_places()
    a = make_schedule(0, places, random.Random(5))
    b = make_schedule(0, places, random.Random(5))
    assert a == b
    sets = [frozenset((g.home, g.away, g.kind) for wk in make_schedule(y, places, random.Random(5)) for g in wk)
            for y in range(12)]
    assert len(set(sets)) == 12, "every year has different opponents"


def test_schedule_is_the_same_in_every_python_run():
    code = ("import random\n"
            "from engine.league import initial_places, make_schedule\n"
            "weeks = make_schedule(3, initial_places(), random.Random(5))\n"
            "print([(g.home, g.away) for wk in weeks for g in wk])")
    outs = set()
    for hash_seed in ("1", "2", "3"):
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert run.returncode == 0, run.stderr
        outs.add(run.stdout)
    assert len(outs) == 1, "the schedule changed between runs, something depends on set order"


def test_rotation_cycles():
    places = initial_places()

    def partner(year, team, kind):
        games = [g for wk in make_schedule(year, places, random.Random(1)) for g in wk]
        return {DIVISION_OF[g.away if g.home == team else g.home] for g in games
                if team in (g.home, g.away) and g.kind == kind}
    for kind, cycle in (("conference", 3), ("interconference", 4)):
        first = partner(0, "CHI", kind)
        assert len(first) == 1
        assert partner(cycle, "CHI", kind) == first
        assert all(partner(y, "CHI", kind) != first for y in range(1, cycle))


def order(results, teams, kind, seed=0):
    return Standings(results, random.Random(seed)).order(teams, kind)


def test_head_to_head_beats_net_points():
    res = [r(1, "BUF", "NE", 21, 14), r(2, "NE", "BUF", 14, 21), r(3, "DEN", "BUF", 30, 0), r(4, "BUF", "DEN", 0, 30),
           r(5, "NE", "KC", 50, 0), r(6, "KC", "NE", 0, 50)]
    st = Standings(res)
    assert st.pct("BUF") == st.pct("NE") == 0.5 and st.net_points("NE") > st.net_points("BUF")
    assert order(res, ["NE", "BUF"], "division") == ["BUF", "NE"]
    assert order(res, ["BUF", "NE"], "division") == ["BUF", "NE"]


def test_division_record_is_the_second_step():
    res = [r(1, "BUF", "NE", 20, 10), r(2, "NE", "BUF", 20, 10),
           r(3, "BUF", "MIA", 20, 10), r(4, "MIA", "BUF", 10, 20),
           r(5, "DEN", "BUF", 20, 10), r(6, "BUF", "DEN", 10, 20),
           r(7, "NE", "KC", 20, 10), r(8, "KC", "NE", 10, 20),
           r(9, "DEN", "NE", 20, 10), r(10, "NE", "DEN", 10, 20)]
    st = Standings(res)
    assert st.pct("BUF") == st.pct("NE") == 0.5
    assert str(st.division_record("BUF")) == "3-1" and str(st.division_record("NE")) == "1-1"
    assert order(res, ["NE", "BUF"], "division") == ["BUF", "NE"]


def test_wild_card_uses_conference_record_when_teams_never_met():
    res = [r(1, "BUF", "CIN", 10, 7), r(2, "DAL", "BUF", 10, 7),
           r(3, "BAL", "DAL", 40, 0), r(4, "CIN", "BAL", 10, 7)]
    st = Standings(res)
    assert st.pct("BUF") == st.pct("BAL") == 0.5 and st.net_points("BAL") > st.net_points("BUF")
    assert order(res, ["BAL", "BUF"], "wild card") == ["BUF", "BAL"]


def test_wild_card_takes_the_best_of_each_division_first():
    res = [r(1, "BUF", "NE", 20, 10), r(2, "NE", "BUF", 10, 20)]
    res += [r(3, "NE", "KC", 20, 10), r(4, "KC", "NE", 10, 20), r(5, "BUF", "DEN", 10, 20), r(6, "DEN", "BUF", 20, 10),
            r(7, "BAL", "KC", 20, 10), r(8, "BAL", "DEN", 20, 10), r(9, "KC", "BAL", 20, 10), r(10, "DEN", "BAL", 20, 10)]
    out = order(res, ["BUF", "NE", "BAL"], "wild card")
    assert sorted(out) == ["BAL", "BUF", "NE"] and out.index("BUF") < out.index("NE")


def test_coin_flip_is_repeatable_and_a_real_flip():
    res = [r(1, "BUF", "CIN", 10, 7), r(2, "CIN", "BUF", 7, 10), r(3, "BAL", "CIN", 10, 7), r(4, "CIN", "BAL", 7, 10)]
    a = order(res, ["BUF", "BAL"], "wild card", seed=3)
    assert a == order(res, ["BUF", "BAL"], "wild card", seed=3)
    seen = {tuple(order(res, ["BUF", "BAL"], "wild card", seed=s)) for s in range(40)}
    assert len(seen) == 2, "with everything equal both orders must happen"


def test_three_team_tie_resolves_in_one_ordering():
    res = [r(1, "BUF", "MIA", 30, 0), r(2, "MIA", "NE", 10, 7), r(3, "NE", "BUF", 10, 7)]
    out = order(res, ["BUF", "MIA", "NE"], "division")
    assert sorted(out) == ["BUF", "MIA", "NE"] and out[0] == "BUF"


def test_common_games_need_four():
    st = Standings([r(1, "BUF", "DEN", 10, 0), r(2, "NE", "DEN", 10, 0)])
    assert st.common_games("BUF", ["BUF", "NE"]) is None


def test_bracket_pairs():
    seeds = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    assert bracket_round(seeds, "wild card") == [("s2", "s7"), ("s3", "s6"), ("s4", "s5")]
    assert bracket_round(["s1", "s2", "s5", "s6"], "divisional") == [("s1", "s6"), ("s2", "s5")]
    assert bracket_round(["s1", "s4"], "conference") == [("s1", "s4")]


def test_full_season_structure():
    s = run_season(0, None, 3, play=fake_play)
    assert len(s.regular) == 272 and len(s.playoffs) == 13
    assert Counter(p.round for p in s.playoffs) == {"wild card": 6, "divisional": 4, "conference": 2, "super bowl": 1}
    wins = Counter()
    for g in s.regular:
        wins[g.winner] += 1
    assert sum(wins.values()) == 272, "no ties with the fake game"
    st = s.standings()
    for conf in CONFERENCES:
        seeds = s.seeds[conf]
        assert len(seeds) == 7 and len(set(seeds)) == 7 and all(CONFERENCE_OF[t] == conf for t in seeds)
        winners = [st.division_order(d)[0] for d in CONFERENCES[conf]]
        assert set(winners) == set(seeds[:4]), "the four division winners are seeds 1 to 4"
        pcts = [st.pct(t) for t in seeds[:4]]
        assert pcts == sorted(pcts, reverse=True)
        wild = [st.pct(t) for t in seeds[4:]]
        assert wild == sorted(wild, reverse=True)
        others = [t for t in CONFERENCES[conf] for t in DIVISIONS[t] if t not in seeds]
        assert all(st.pct(o) <= wild[-1] for o in others), "wild cards are the best of the rest"
    first_round = [p for p in s.playoffs if p.round == "wild card"]
    assert all(p.home != s.seeds[CONFERENCE_OF[p.home]][0] for p in first_round), "seed 1 has the bye"
    for p in s.playoffs:
        seeds = {t: i for conf in s.seeds.values() for i, t in enumerate(conf, 1)}
        if p.round != "super bowl":
            assert seeds[p.home] < seeds[p.away], "higher seed hosts"
    assert s.champion == s.playoffs[-1].winner and s.champion != s.runner_up
    assert s.places == st.places() and all(sorted(s.places[t] for t in ts) == [1, 2, 3, 4] for ts in DIVISIONS.values())


def test_division_winner_with_a_poor_record_still_gets_in():
    teams = ALL_ABBRS
    results = []
    target = {}
    for ts in CONFERENCES["AFC"]:
        for n, t in enumerate(DIVISIONS[ts]):
            target[t] = [12, 11, 10, 9][n]
    target["MIA"], target["BUF"], target["NE"], target["NYJ"] = 6, 5, 4, 3
    week = 1
    for t in teams:
        for k in range(17):
            if CONFERENCE_OF[t] == "AFC":
                win = k < target[t]
                results.append(r(week, t, "ZZZ", 10 if win else 0, 0 if win else 10))
        week += 1
    DIVISION_OF["ZZZ"], CONFERENCE_OF["ZZZ"] = "NFC West", "NFC"
    try:
        st = Standings(results)
        seeds = st.seeds("AFC")
        assert "MIA" in seeds[:4], seeds
        assert seeds.index("MIA") == 3, "the 6-11 division winner is seed 4, below every better record's division winner"
        assert len(seeds) == 7
    finally:
        del DIVISION_OF["ZZZ"], CONFERENCE_OF["ZZZ"]


def test_no_team_plays_twice_in_a_playoff_round():
    s = run_season(1, None, 9, play=fake_play)
    for rnd in ("wild card", "divisional", "conference"):
        teams = [t for p in s.playoffs if p.round == rnd for t in (p.home, p.away)]
        assert len(teams) == len(set(teams))


def test_season_is_repeatable_and_seeds_differ():
    a = run_season(0, None, 4, play=fake_play)
    b = run_season(0, None, 4, play=fake_play)
    c = run_season(0, None, 5, play=fake_play)
    assert [(g.week, g.home, g.away) for g in a.results] == [(g.week, g.home, g.away) for g in b.results]
    assert [(g.week, g.home, g.away) for g in a.results] != [(g.week, g.home, g.away) for g in c.results]


def test_many_seasons_new_schedule_each_year():
    seasons, places = [], None
    for year in range(6):
        seasons.append(run_season(year, places, 2, play=fake_play))
        places = seasons[-1].places
    keys = [frozenset((g.home, g.away) for g in s.regular) for s in seasons]
    assert len(set(keys)) == 6 and [s.year for s in seasons] == list(range(6))
    rebuilt = make_schedule(1, seasons[0].places, random.Random("schedule|2|1"))
    assert [(s.home, s.away) for wk in rebuilt for s in wk] == [(h, a) for wk in seasons[1].weeks for h, a, _ in wk]


def test_team_totals_and_table():
    s = run_season(0, None, 3, play=fake_play)
    totals = team_totals(s.results)
    assert all(c["games"] == 17 for c in totals.values())
    assert sum(c["w"] for c in totals.values()) == sum(c["l"] for c in totals.values()) == 272
    assert sum(c["pf"] for c in totals.values()) == sum(c["pa"] for c in totals.values())
    assert all(c["yards"] == 300 * 17 and c["opp_yards"] == 300 * 17 for c in totals.values())
    rows = team_table(s.results)
    assert len(rows) == 32 and rows[0][1] >= rows[-1][1]


def test_playoff_games_never_tie_and_regular_games_can():
    tables = make_tables()
    for seed in range(400):
        g = Game(tables, random.Random(seed), playoff=True)
        while not (g.over and not g._queue):
            g.step()
        assert g.scores[0] != g.scores[1], seed
    tied = 0
    for seed in range(1500):
        g = Game(tables, random.Random(seed))
        while not (g.over and not g._queue):
            g.step()
        tied += g.scores[0] == g.scores[1]
    assert tied > 0, "regular season ties still happen"


def test_real_engine_season_runs_and_is_repeatable():
    tables = make_tables()
    a = run_season(0, None, 11, tables=tables)
    assert len(a.regular) == 272 and len(a.playoffs) == 13
    assert all(p.home_score != p.away_score for p in a.playoffs)
    ties = sum(1 for g in a.regular if g.home_score == g.away_score)
    assert ties <= 8
    for g in a.results:
        assert g.stats[0]["yards"] > 0 and g.stats[0]["plays"] > 20
        assert sum(g.stats[i].get("top", 0) for i in (0, 1)) >= 3600 - 1
    b = run_season(0, None, 11, tables=tables)
    assert [(g.home_score, g.away_score) for g in a.results] == [(g.home_score, g.away_score) for g in b.results]
    c = run_season(0, None, 11, tables=tables, workers=2)
    assert [(g.home_score, g.away_score) for g in a.regular] == [(g.home_score, g.away_score) for g in c.regular]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All season checks passed.")
