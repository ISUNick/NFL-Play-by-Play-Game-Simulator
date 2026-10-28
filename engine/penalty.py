from dataclasses import dataclass
from typing import Optional

from engine import home_field
from engine.state import GameState

SPOT_FOUL = "Defensive Pass Interference"  # ball goes to the spot of the foul, not half the distance
# rulebook yards, the sign (for or against the offense) still comes from the data
STANDARD_YARDS = {"False Start": 5, "Neutral Zone Infraction": 5, "Encroachment": 5, "Delay of Game": 5,
                  "Defensive Offside": 5, "Offensive Offside": 5, "Illegal Formation": 5, "Illegal Shift": 5,
                  "Illegal Motion": 5, "Illegal Substitution": 5, "Defensive Delay of Game": 5, "Defensive Holding": 5,
                  "Offensive Too Many Men on Field": 5, "Defensive Too Many Men on Field": 5, "Illegal Use of Hands": 5,
                  "Offensive Holding": 10, "Unnecessary Roughness": 15, "Face Mask": 15}


MIN_BANK = 20  # fewer fouls than this in a group and we use a wider group


@dataclass
class PenaltyResult:
    team: str
    foul: str
    yards: int
    new_yardline_100: int
    down: int
    distance: int
    first_down: bool
    seconds: int
    kind: str = "pre"
    halved: bool = False


def play_kind(outcome) -> str:
    if outcome.kind in ("run", "scramble", "sack"):
        return outcome.kind
    return "pass_deep" if outcome.depth == "deep" else "pass_short"


def _roll(row, rng, phase: str = "pre", home_offense: Optional[bool] = None) -> Optional[str]:
    p_off, p_def = row["p_off"], row["p_def"]
    if home_offense is not None:
        p_off *= home_field.multiplier(phase, "offense", home_offense)
        p_def *= home_field.multiplier(phase, "defense", not home_offense)
    u = rng.random()
    if u < p_off:
        return "offense"
    if u < p_off + p_def:
        return "defense"
    return None


def roll_pre_snap(state: GameState, tables, rng, home_offense: Optional[bool] = None) -> Optional[str]:
    return _roll(tables.lookup("penalty_pre", state.situation())[1], rng, "pre", home_offense)


def roll_post_snap(state: GameState, outcome, tables, rng, home_offense: Optional[bool] = None) -> Optional[str]:
    sit = state.situation()
    sit["kind"] = play_kind(outcome)
    return _roll(tables.lookup("penalty_post", sit)[1], rng, "post", home_offense)


def _draw_foul(bank: list, team: str, rng, kind: str, home_offense: Optional[bool]):
    if home_offense is None or kind != "pre":
        return rng.choice(bank)
    at_home = home_offense if team == "offense" else not home_offense
    cap = home_field.max_foul_weight(team, at_home)
    while True:
        row = rng.choice(bank)
        if rng.random() * cap < home_field.foul_weight(team, row[3], at_home):
            return row


def resolve_penalty(state: GameState, team: str, tables, rng, kind: str = "pre",
                    home_offense: Optional[bool] = None) -> PenaltyResult:
    side = "off" if team == "offense" else "def"
    zone = state.situation()["zone_b"]
    for key in ((side, kind, zone), (side, kind), (side,)):
        bank = tables.penalty.get(key)
        if bank and len(bank) >= MIN_BANK:
            break
    dy, automatic_first, seconds, foul = _draw_foul(bank, team, rng, kind, home_offense)
    if dy and foul in STANDARD_YARDS:
        dy = STANDARD_YARDS[foul] if dy > 0 else -STANDARD_YARDS[foul]

    yl = state.yardline_100
    before = dy
    # no penalty can take more than half the distance to the goal
    if dy > 0:
        dy = min(dy, (100 - yl) // 2)
    elif foul == SPOT_FOUL:
        dy = max(dy, -(yl - 1))
    else:
        dy = max(dy, -(yl // 2))
    new_yl = yl + dy
    new_distance = state.distance + dy
    if automatic_first or new_distance <= 0:
        down, distance, first = 1, min(10, new_yl), True
    else:
        down, distance, first = state.down, max(1, min(new_distance, new_yl)), False
    return PenaltyResult(team, foul, abs(dy), new_yl, down, distance, first, seconds, kind, halved=dy != before)
