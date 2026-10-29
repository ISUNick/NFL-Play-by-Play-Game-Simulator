import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.game import Game
from engine.teams import TeamModel
from players.lineup import Ratings, build_offense, home_bonus, matchup, team_edge, unit_grades
from test_engine_tables import tables as make_tables

RATINGS = Ratings.load()
LINEUP = pd.read_csv(ROOT / "data" / "players" / "current_roster.csv")
MODEL = TeamModel()


def play(units, edges, seed):
    g = Game(make_tables(), random.Random(seed), names=("CHI", "GB"), edges=edges, home_field=True, units=units)
    while not (g.over and not g._queue):
        g.step()
    return g


def test_weights_cover_the_whole_offense():
    o = build_offense("CHI", LINEUP, RATINGS)
    assert o.qb.name and len(o.receivers) >= 5 and len(o.runners) >= 2
    assert all(r.weight > 0 for r in o.receivers + o.runners)


def test_a_player_who_is_out_is_never_targeted_and_the_others_pick_up_his_share():
    full = build_offense("CHI", LINEUP, RATINGS)
    star = max(full.receivers, key=lambda r: r.weight)
    short = build_offense("CHI", LINEUP, RATINGS, out_names=[star.name])
    assert star.name not in [r.name for r in short.receivers]
    rng = random.Random(1)
    assert all(short.pick_receiver(rng).name != star.name for _ in range(3000))
    same_group = [r for r in full.receivers if r.position == star.position and r.name != star.name]
    after = {r.name: r.weight for r in short.receivers}
    assert sum(after[r.name] for r in same_group) > sum(r.weight for r in same_group)


def test_marking_the_starting_quarterback_out_starts_the_backup():
    starter = build_offense("CHI", LINEUP, RATINGS).qb.name
    backup = build_offense("CHI", LINEUP, RATINGS, out_names=[starter]).qb.name
    assert backup != starter


def test_no_quarterback_is_an_error():
    qbs = LINEUP[(LINEUP.team == "CHI") & (LINEUP.position == "QB")].full_name.tolist()
    try:
        build_offense("CHI", LINEUP, RATINGS, out_names=qbs)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_same_seed_same_game_and_named_stats_add_up():
    edges, units = matchup("CHI", "GB", LINEUP, RATINGS, MODEL)
    a, b = play(units, edges, 11), play(units, edges, 11)
    assert a.scores == b.scores and a.players == b.players
    for side in (0, 1):
        lines = a.players[side].values()
        assert sum(p["targets"] for p in lines) == a.stats[side]["pass_att"]
        assert sum(p["att"] for p in lines) == a.stats[side]["pass_att"]
        assert sum(p["rec"] for p in lines) == a.stats[side]["pass_comp"]
        assert sum(p["rush_yards"] for p in lines) == a.stats[side]["rush_yards"]


def test_a_receiver_who_is_out_has_no_stats_in_the_game():
    star = max(build_offense("CHI", LINEUP, RATINGS).receivers, key=lambda r: r.weight).name
    edges, units = matchup("CHI", "GB", LINEUP, RATINGS, MODEL, out_home=[star])
    g = play(units, edges, 5)
    assert star not in g.players[0]


def test_home_field_helps_the_home_offense_and_hurts_the_away_offense():
    home, away = home_bonus(MODEL)
    assert home.completion > 0 and away.completion < 0


def test_scaling_stretches_every_effect():
    big = RATINGS.scaled(2.0, 3.0)
    assert abs(big.team.off_completion.iloc[0] - 2 * RATINGS.team.off_completion.iloc[0]) < 1e-12
    assert abs(big.qb.comp_shift.iloc[0] - 3 * RATINGS.qb.comp_shift.iloc[0]) < 1e-12


def test_every_team_can_field_an_offense():
    for team in RATINGS.team.index:
        build_offense(team, LINEUP, RATINGS)


def test_a_player_listed_out_plays_when_he_is_put_back():
    out = LINEUP[LINEUP.out & LINEUP.position.isin(["WR", "TE"]) & (LINEUP.snap_share > 0.2)]
    assert len(out), "the injury report should list some receivers out"
    row = out.iloc[0]
    assert row.full_name not in [r.name for r in build_offense(row.team, LINEUP, RATINGS).receivers]
    assert row.full_name in [r.name for r in build_offense(row.team, LINEUP, RATINGS, play_names=[row.full_name]).receivers]


def best_and_worst(unit):
    rows = unit_grades(RATINGS)[unit]
    return rows[0]["team"], rows[-1]["team"]


def test_a_borrowed_line_brings_its_run_and_sack_numbers_but_not_its_passing():
    best, worst = best_and_worst("line")
    own = team_edge(RATINGS, worst, "GB")
    swapped = team_edge(RATINGS, worst, "GB", {"line": best})
    assert swapped.run_tilt > own.run_tilt and swapped.sack < own.sack
    assert swapped.completion == own.completion and swapped.interception == own.interception
    assert team_edge(RATINGS, worst, "GB", {"line": worst}) == own


def test_a_borrowed_pass_rush_only_changes_sacks():
    best, worst = best_and_worst("rush")
    own = team_edge(RATINGS, "CHI", worst)
    swapped = team_edge(RATINGS, "CHI", worst, defense_swap={"rush": best})
    assert swapped.sack > own.sack
    assert (swapped.run_tilt, swapped.completion, swapped.interception) == (own.run_tilt, own.completion, own.interception)


def test_a_borrowed_run_defense_only_changes_the_run():
    best, worst = best_and_worst("run_defense")
    own = team_edge(RATINGS, "CHI", worst)
    swapped = team_edge(RATINGS, "CHI", worst, defense_swap={"run_defense": best})
    assert swapped.run_tilt < own.run_tilt
    assert (swapped.sack, swapped.completion, swapped.completion_tilt) == (own.sack, own.completion, own.completion_tilt)


def test_a_borrowed_coverage_changes_the_passing_game_and_not_the_run():
    best, worst = best_and_worst("coverage")
    own = team_edge(RATINGS, "CHI", worst)
    swapped = team_edge(RATINGS, "CHI", worst, defense_swap={"coverage": best})
    assert swapped.completion < own.completion and swapped.interception > own.interception
    assert swapped.completion_tilt < own.completion_tilt
    assert (swapped.sack, swapped.run_tilt) == (own.sack, own.run_tilt)


def test_unit_grades_rank_every_team_best_first():
    for unit, rows in unit_grades(RATINGS).items():
        assert len(rows) == len(RATINGS.team) and [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
        assert all(a["score"] >= b["score"] for a, b in zip(rows, rows[1:]))


def test_a_good_defense_unit_cancels_a_good_offense_unit():
    best_line = best_and_worst("line")[0]
    best_run_defense = best_and_worst("run_defense")[0]
    both = team_edge(RATINGS, "CHI", "GB", {"line": best_line}, {"run_defense": best_run_defense})
    line_only = team_edge(RATINGS, "CHI", "GB", {"line": best_line}, {})
    defense_only = team_edge(RATINGS, "CHI", "GB", {}, {"run_defense": best_run_defense})
    assert defense_only.run_tilt < both.run_tilt < line_only.run_tilt


def test_matchup_applies_each_sides_swaps_to_the_right_offense():
    best_line, worst_line = best_and_worst("line")
    base = matchup("CHI", "GB", LINEUP, RATINGS, MODEL)[0]
    home = matchup("CHI", "GB", LINEUP, RATINGS, MODEL, swap_home={"line": best_line})[0]
    away = matchup("CHI", "GB", LINEUP, RATINGS, MODEL, swap_away={"line": best_line})[0]
    assert home[0] != base[0] and home[1] == base[1]
    assert away[1] != base[1] and away[0] == base[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All lineup checks passed.")
