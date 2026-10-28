import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import boxscore
from engine.game import Game
from engine.tables import Tables
from engine.text import clock_text

_T = None


def tables():
    global _T
    if _T is None:
        _T = Tables()
    return _T


def play_games(n, seed=1):
    out = []
    for i in range(n):
        g = Game(tables(), random.Random(seed + i), names=("A", "B"))
        while not g.over:
            g.step()
        out.append(g)
    return out


def test_the_counters_add_up_in_every_game():
    for g in play_games(30):
        for t in (0, 1):
            s = g.stats[t]
            assert s["plays"] == s["pass_att"] + s["rush_att"] + s["sack"], dict(s)
            assert s["pass_att"] == s["complete"] + s["incomplete"] + s["interception"]
            assert s["yards"] == s["rush_yards"] + s["pass_yards"] - s["sack_yards"], dict(s)
            assert s["pass_comp"] == s["complete"] and s["pass_td"] <= s["pass_comp"]
            assert s["third_conv"] <= s["third_att"] and s["fourth_conv"] <= s["fourth_att"]
            assert s["rz_td"] <= s["rz_trips"] <= s["drives"]
            assert s["rush_td"] + s["pass_td"] <= s["td"]
            assert s["big_plays"] <= s["plays"]


def test_time_of_possession_and_drives():
    for g in play_games(20, 100):
        total = g.stats[0]["top"] + g.stats[1]["top"]
        assert 2400 <= total <= 3600 + 15 * 60, total
        assert g.stats[0]["drives"] >= 4 and g.stats[1]["drives"] >= 4


def test_possession_time_adds_up_to_the_game():
    for g in play_games(40, 300):
        total = g.stats[0]["top"] + g.stats[1]["top"]
        if not g.went_to_ot:
            assert total == 3600, total
        else:
            assert 3600 < total <= 3600 + 600, total


def test_no_play_takes_zero_seconds_and_standard_fouls_cost_standard_yards():
    saw_flag = 0
    for seed in range(60):
        g = Game(tables(), random.Random(seed), names=("A", "B"))
        while not g.over:
            ev = g.step()
            if ev.kind == "play" and ev.seconds is not None:
                assert ev.seconds >= 3, (ev.text, ev.seconds)
            if ev.kind == "penalty":
                r = ev.detail
                far = 12 <= ev.state.yardline_100 <= 88
                if far and r.foul in ("False Start", "Delayed Start", "Neutral Zone Infraction", "Delay of Game", "Illegal Formation"):
                    saw_flag += 1
                    assert r.yards == 5, (r.foul, r.yards)
                if far and r.foul == "Offensive Holding" and ev.state.yardline_100 <= 80:
                    assert r.yards == 10, (r.foul, r.yards)
    assert saw_flag > 10


def test_league_numbers_look_like_the_nfl():
    games = play_games(300, 1000)
    n = 2 * len(games)
    tot = Counter()
    for g in games:
        for t in (0, 1):
            tot.update(g.stats[t])
    comp = tot["pass_comp"] / tot["pass_att"]
    third = tot["third_conv"] / tot["third_att"]
    rz = tot["rz_td"] / tot["rz_trips"]
    print(f"  per team per game: {tot['yards']/n:.0f} yds, {tot['pass_att']/n:.1f} att, {tot['rush_att']/n:.1f} carries, "
          f"comp {comp:.1%}, 3rd {third:.1%}, red zone TD {rz:.1%}, {tot['top']/n/60:.1f} min possession")
    assert 0.60 <= comp <= 0.68
    assert 0.35 <= third <= 0.46
    assert 0.48 <= rz <= 0.68
    assert 28 <= tot["pass_att"] / n <= 40 and 20 <= tot["rush_att"] / n <= 32
    assert 280 <= tot["yards"] / n <= 400


def test_passer_rating_and_rows():
    assert abs(boxscore.passer_rating(25, 25, 400, 5, 0) - 158.33) < 0.01
    assert boxscore.passer_rating(0, 10, 0, 0, 10) == 0.0
    assert boxscore.passer_rating(0, 0, 0, 0, 0) == 0.0
    a, b = Counter(plays=10, yards=80, pass_att=4, pass_comp=2, pass_yards=30, rush_att=5, rush_yards=50, sack=1, sack_yards=0, third_att=2, third_conv=1), Counter()
    secs = dict(boxscore.sections(a, b))
    rows = {r.label: r for rows in secs.values() for r in rows}
    assert rows["Total yards"].text == ("80", "0") and rows["Comp / Att"].text == ("2/4", "0/0")
    assert rows["3rd down"].text == ("1/2 50%", "0/0 -") and rows["Time of poss"].text == ("0:00", "0:00")
    assert rows["Turnovers"].better == -1 and rows["Plays"].better == 0
    assert clock_text(754) == "12:34"


def test_the_stats_do_not_change_the_game():
    g1, g2 = play_games(1, 5)[0], play_games(1, 5)[0]
    assert g1.scores == g2.scores and dict(g1.stats[0]) == dict(g2.stats[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All boxscore tests passed")
