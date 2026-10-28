import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.box import build_box, passer_rating
from engine.play import PlayOutcome
from engine.text import describe_play

ROSTER = [("Starter", "QB"), ("Backup", "QB"), ("Back", "RB"), ("Slot", "WR"), ("Tight", "TE")]


def test_perfect_passer_rating_is_158_3():
    assert abs(passer_rating(10, 10, 200, 3, 0) - 158.3) < 0.1
    assert passer_rating(0, 0, 0, 0, 0) == 0.0


def test_only_players_with_a_stat_are_listed_and_the_best_yards_come_first():
    box = build_box({"Starter": {"att": 30, "comp": 20, "pass_yards": 250, "pass_td": 2, "int": 1, "sacked": 3},
                     "Back": {"carries": 10, "rush_yards": 40, "targets": 3, "rec": 2, "rec_yards": 20},
                     "Slot": {"targets": 6, "rec": 5, "rec_yards": 80}, "Tight": {"targets": 2, "rec": 1, "rec_yards": 5}}, ROSTER)
    assert [r["name"] for r in box["passing"]] == ["Starter"]
    assert [r["name"] for r in box["rushing"]] == ["Back"]
    assert [r["name"] for r in box["receiving"]] == ["Slot", "Back", "Tight"]
    assert box["passing"][0]["pct"] == 20 / 30 * 100


def test_a_backup_who_never_threw_is_not_listed():
    box = build_box({"Starter": {"att": 5, "comp": 3, "pass_yards": 30}}, ROSTER)
    assert "Backup" not in [r["name"] for r in box["passing"]]
    assert build_box({}, ROSTER) == {"passing": [], "rushing": [], "receiving": []}


def test_rushing_is_sorted_by_yards_and_a_quarterback_can_lead_it():
    box = build_box({"Starter": {"carries": 4, "rush_yards": 31}, "Back": {"carries": 12, "rush_yards": 20},
                     "Slot": {"carries": 1, "rush_yards": 9}}, ROSTER)
    assert [r["name"] for r in box["rushing"]] == ["Starter", "Back", "Slot"]


def test_a_receiver_who_throws_and_an_unlisted_player_are_added():
    box = build_box({"Slot": {"att": 1, "comp": 1, "pass_yards": 30}, "Gadget": {"att": 1, "comp": 0}}, ROSTER)
    assert [r["name"] for r in box["passing"]] == ["Slot", "Gadget"]


def test_play_text_uses_names_and_no_depth_words():
    done = PlayOutcome("complete", "complete_in", 23, 40, first_down=True, depth="deep", passer="Caleb Williams", target="Rome Odunze")
    assert describe_play(done, "CHI", "GB") == "Caleb Williams pass complete to Rome Odunze, 23 yards. First down"
    miss = PlayOutcome("incomplete", "incomplete", 0, 40, depth="short", passer="Caleb Williams", target="Rome Odunze")
    assert "short" not in describe_play(miss, "CHI", "GB") and "Rome Odunze" in describe_play(miss, "CHI", "GB")
    run = PlayOutcome("run", "run_in", 4, 40, rusher="D'Andre Swift")
    assert describe_play(run, "CHI", "GB") == "D'Andre Swift rush, 4 yards"
    sack = PlayOutcome("sack", "sack", -7, 40, passer="Jordan Love")
    assert describe_play(sack, "GB", "CHI") == "Jordan Love sacked, loss of 7 yards"


def test_plays_without_names_still_read_fine():
    run = PlayOutcome("run", "run_in", 4, 40)
    assert describe_play(run, "CHI", "GB") == "CHI rush, 4 yards"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All box score checks passed.")
