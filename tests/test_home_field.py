import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine import home_field
from engine.game import Game
from engine.penalty import _roll, resolve_penalty
from engine.season import play_game
from engine.state import GameState
from engine.teams import Team, TeamModel, find_team
from test_engine_tables import tables as make_tables


class FixedRandom:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def test_the_numbers_are_there_and_make_sense():
    d = home_field.data()
    assert 1.5 < d["real"]["home_margin"] < 3.2 and 0.53 < d["real"]["home_win_pct"] < 0.6 and d["real"]["games"] > 3000
    assert 0.8 < home_field.rating_points() < 2.5
    for key in ("pre_offense", "pre_defense", "post_offense", "post_defense"):
        m = d["penalties"][key]["multiplier"]
        assert abs((m["home"] + m["away"]) / 2 - 1) < 1e-6, "the two multipliers average to 1, the table's chance stays the league average"
    assert home_field.multiplier("pre", "defense", True) < 1 < home_field.multiplier("pre", "defense", False)
    assert home_field.multiplier("post", "defense", True) < 1 < home_field.multiplier("post", "defense", False)
    assert home_field.foul_weight("offense", "Delay of Game", False) > 1.1 > 0.9 > home_field.foul_weight("offense", "Delay of Game", True)
    for side in ("offense", "defense"):
        for foul in ("Delay of Game", "False Start", "Defensive Offside", "A Foul Never Seen"):
            assert home_field.foul_weight(side, foul, True) > 0 and home_field.foul_weight(side, foul, False) > 0


def test_away_teams_do_not_get_more_false_starts_in_the_real_data():
    assert home_field.foul_weight("offense", "False Start", True) >= home_field.foul_weight("offense", "False Start", False)


def test_the_chance_of_a_flag_is_scaled_exactly_for_each_team():
    row = {"p_off": 0.02, "p_def": 0.01}
    assert _roll(row, FixedRandom(0.02), "pre", home_offense=False) == "offense"
    assert _roll(row, FixedRandom(0.02), "pre", home_offense=True) != "offense"
    lo = row["p_def"] * home_field.multiplier("pre", "defense", True)
    hi = row["p_def"] * home_field.multiplier("pre", "defense", False)
    p_off_away = row["p_off"] * home_field.multiplier("pre", "offense", False)
    u = p_off_away + (lo + hi) / 2
    assert _roll(row, FixedRandom(u), "pre", home_offense=True) == "defense", "away defense is flagged more"
    p_off_home = row["p_off"] * home_field.multiplier("pre", "offense", True)
    assert _roll(row, FixedRandom(p_off_home + (lo + hi) / 2), "pre", home_offense=False) is None, "home defense is flagged less"


def test_no_home_field_means_the_table_unchanged():
    row = {"p_off": 0.02, "p_def": 0.01}
    assert _roll(row, FixedRandom(0.0199)) == "offense" and _roll(row, FixedRandom(0.0201)) == "defense"
    assert _roll(row, FixedRandom(0.0299)) == "defense" and _roll(row, FixedRandom(0.0301)) is None


def test_delay_of_game_is_called_more_on_the_away_offense():
    tables = make_tables()
    st = GameState(quarter=2, clock=600, down=1, distance=10, yardline_100=60, offense_score=7, defense_score=7,
                   offense_timeouts=3, defense_timeouts=3)
    counts = {}
    for home in (True, False):
        rng, delay, n = random.Random(8), 0, 12000
        for _ in range(n):
            r = resolve_penalty(st, "offense", tables, rng, "pre", home_offense=home)
            delay += r.foul == "Delay of Game"
        counts[home] = delay / n
    assert counts[False] > 1.25 * counts[True], counts
    league = 0.5 * (counts[True] + counts[False])
    plain = random.Random(8)
    base = sum(resolve_penalty(st, "offense", tables, plain, "pre").foul == "Delay of Game" for _ in range(12000)) / 12000
    assert abs(league - base) < 0.025, "away and home together keep the league's mix of fouls"


def test_home_bonus_goes_to_the_home_team_on_both_sides_of_the_ball():
    model = TeamModel()
    avg = Team("Avg", "AVG", 80, 80, 80)
    even = model.edge_pair(avg, avg)
    home, away = model.edge_pair(avg, avg, home_field=True)
    assert model.edge_pair(avg, avg, home_field=False) == even
    assert home.completion > even[0].completion > away.completion, "the home offense gains, the away offense faces the home defense"
    assert home.completion > 0 > away.completion
    assert abs(home.completion + away.completion) < abs(home.completion) * 0.2, "about equal and opposite"


def replay(tables, model, seed, home_field):
    edges = model.edge_pair(find_team("KC"), find_team("BUF"), home_field=home_field)
    g = Game(tables, random.Random(seed), names=("KC", "BUF"), edges=edges, playoff=True, home_field=home_field)
    while not (g.over and not g._queue):
        g.step()
    return g.scores[0], g.scores[1]


def test_super_bowl_is_at_a_neutral_site():
    tables, model = make_tables(), TeamModel()
    for seed in ("n1", "n2", "n3"):
        sb = play_game(tables, model, "KC", "BUF", seed, 22, "super bowl")
        assert (sb.home_score, sb.away_score) == replay(tables, model, seed, False)
        wk = play_game(tables, model, "KC", "BUF", seed, 20, "divisional")
        assert (wk.home_score, wk.away_score) == replay(tables, model, seed, True)


def test_the_engine_reproduces_the_real_home_margin_and_flags():
    tables, model = make_tables(), TeamModel()
    avg = Team("Avg", "AVG", round(model._stats["offense"][0]), round(model._stats["defense"][0]), 80)
    edges = model.edge_pair(avg, avg, home_field=True)
    margin, flags = [], [0, 0]
    n = 3000
    for i in range(n):
        g = Game(tables, random.Random(f"hf{i}"), names=("H", "A"), edges=edges, home_field=True)
        while not (g.over and not g._queue):
            g.step()
        margin.append(g.scores[0] - g.scores[1])
        flags[0] += g.stats[0].get("penalty", 0)
        flags[1] += g.stats[1].get("penalty", 0)
    mean = sum(margin) / n
    real = home_field.data()["real"]["home_margin"]
    assert abs(mean - real) < 1.0, (mean, real)
    assert flags[1] > flags[0] + 0.2 * n, "the away team draws clearly more flags a game"
    assert 3 < flags[0] / n < 9


def test_games_without_home_field_are_unchanged():
    tables = make_tables()
    a = Game(tables, random.Random(5), names=("A", "B"))
    b = Game(tables, random.Random(5), names=("A", "B"), home_field=False)
    for g in (a, b):
        while not (g.over and not g._queue):
            g.step()
    assert a.scores == b.scores and a.stats == b.stats


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All home field checks passed.")
