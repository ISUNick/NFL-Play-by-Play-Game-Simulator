import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from engine.game import Game
from engine.teams import TeamModel
from players.lineup import Ratings, build_offense, home_bonus, line_grades, matchup, team_edge
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


def test_a_borrowed_line_brings_its_run_and_sack_numbers_but_not_its_passing():
    t = RATINGS.team
    best, worst = t.off_run_yards.idxmax(), t.off_run_yards.idxmin()
    own = team_edge(RATINGS, worst, "GB")
    swapped = team_edge(RATINGS, worst, "GB", line=best)
    assert swapped.run_tilt > own.run_tilt
    assert swapped.completion == own.completion and swapped.interception == own.interception
    assert team_edge(RATINGS, worst, "GB", line=worst) == own


def test_line_grades_rank_every_team_best_first():
    rows = line_grades(RATINGS)
    assert len(rows) == len(RATINGS.team) and [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert all(a["score"] >= b["score"] for a, b in zip(rows, rows[1:]))


def test_matchup_takes_a_line_for_either_side():
    best = line_grades(RATINGS)[0]["team"]
    base = matchup("CHI", "GB", LINEUP, RATINGS, MODEL)
    plugged = matchup("CHI", "GB", LINEUP, RATINGS, MODEL, line_home=best, line_away=best)
    assert plugged[0][0] == team_edge(RATINGS, "CHI", "GB", best) or plugged[0][0] != base[0][0]
    assert plugged[0][1] == matchup("CHI", "GB", LINEUP, RATINGS, MODEL, line_away=best)[0][1]
    assert plugged[0][0] != base[0][0] or best == "CHI"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All lineup checks passed.")
