import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine.game import Game
from engine.tables import Tables
from engine.text import clock_text, down_text, spot_text

DRIVE_ENDERS = {"kickoff", "onside", "punt", "quarter_end", "halftime", "end_regulation", "game_end"}


def scoreboard(g: Game) -> str:
    return f"[{g.names[0]} {g.scores[0]}  {g.names[1]} {g.scores[1]}]"


def show_stats(g: Game) -> None:
    keys = ["plays", "yards", "first_downs", "td", "complete", "incomplete", "interception", "sack", "run",
            "scramble", "punt", "fg_att", "fg_made", "kneel", "interceptions_thrown", "fumbles_lost", "penalty", "penalty_yards"]
    print(f"\n{'':22}{g.names[0]:>8}{g.names[1]:>8}")
    for k in keys:
        print(f"{k:22}{g.stats[0][k]:>8}{g.stats[1][k]:>8}")
    print()


SNAP_KINDS = {"play", "penalty", "punt", "field_goal", "kneel", "spike"}


def score_bug(ev, offense: int, names) -> str:
    st = ev.state
    quarter = "OT" if st.quarter == 5 else f"Q{st.quarter}"
    situation = f"{down_text(st.down, st.distance, st.yardline_100)} at {spot_text(st.yardline_100)}"
    scores = (st.offense_score, st.defense_score) if offense == 0 else (st.defense_score, st.offense_score)
    return (f"{quarter} {clock_text(st.clock)} | {situation} | {names[offense]} ball | "
            f"{names[0]} {scores[0]}, {names[1]} {scores[1]}")


def play_only(text: str) -> str:
    m = re.match(r"^(?:Q\d|OT) \d+:\d\d\s+", text)
    rest = text[m.end():] if m else text
    if re.match(r"^\d(?:st|nd|rd|th) &", rest) and ": " in rest:
        rest = rest.split(": ", 1)[1]
    return rest


def finished(g: Game) -> bool:
    return g.over and not g._queue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--overtime", action="store_true",
                    help="start from the first seed (at or after --seed) that goes to overtime, to test overtime play")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print("Loading data tables...")
    tables = Tables()
    if args.overtime:
        while True:
            probe = Game(tables, random.Random(seed))
            while not finished(probe):
                probe.step()
            if probe.went_to_ot:
                break
            seed += 1
    print(f"(game seed {seed})")
    g = Game(tables, random.Random(seed))

    run_to = None
    while not finished(g):
        at_snap = g.phase == "scrimmage" and not g._queue and not g.over
        offense = g.offense
        ev = g.step()
        if at_snap and ev.state is not None and ev.kind in SNAP_KINDS:
            print(f"\n{score_bug(ev, offense, g.names)}")
        body = play_only(ev.text) if ev.state is not None and ev.kind in ("play", "penalty", "timeout") else ev.text
        print(f"   {body}   {scoreboard(g)}" if ev.scorer is not None else f"   {body}")
        if args.auto:
            continue
        if run_to == "game":
            continue
        if run_to == "drive" and ev.kind not in DRIVE_ENDERS:
            continue
        if run_to == "quarter" and ev.kind not in ("quarter_end", "halftime", "end_regulation", "game_end"):
            continue
        run_to = None
        if finished(g):
            break
        while True:
            try:
                cmd = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if cmd == "":
                break
            if cmd == "d":
                run_to = "drive"
                break
            if cmd == "q":
                run_to = "quarter"
                break
            if cmd == "g":
                run_to = "game"
                break
            if cmd == "s":
                show_stats(g)
            elif cmd == "x":
                return
            else:
                print("Enter = next event, d = drive, q = quarter, g = finish, s = stats, x = quit")
    show_stats(g)


if __name__ == "__main__":
    main()
