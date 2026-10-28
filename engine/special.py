from dataclasses import dataclass

from engine.clock_rules import MAX_DEFICIT_FOR_FIELD_GOAL, kick_distance
from engine.state import GameState, late_score_bucket, score_bucket

TOUCHBACK_YARDLINE = 80  # the 20 yard line

ENDGAME_SECONDS = 120  # the guard rules below only apply in the last 2 minutes

OWN_HALF_YARDLINE = 50

MIN_PUNT_YARDLINE = 28  # closer to the end zone than this nobody punts

MAX_KICK_DISTANCE = 66  # longest field goal attempt in the data

GO, PUNT, FIELD_GOAL = "go", "punt", "field_goal"


@dataclass
class PuntResult:
    kind: str
    distance: int
    return_yards: int
    receiver_has_ball: bool
    yardline_100: int
    touchdown: bool


def _clamp(yard: int) -> int:
    return max(1, min(99, yard))


def apply_punt(yardline_100: int, sample: tuple) -> PuntResult:
    distance, result, ret, return_td, fumble, blocked_poss, blocked_delta = sample

    if result == "blocked":
        spot = _clamp(yardline_100 + blocked_delta)
        if blocked_poss:
            return PuntResult("blocked", 0, 0, True, 100 - spot, False)
        return PuntResult("blocked", 0, 0, False, spot, False)

    if result == "return_td":
        return PuntResult("return_td", distance, ret, True, 0, True)

    landing = yardline_100 - distance
    if result == "touchback" or landing <= 0:
        return PuntResult("touchback", distance, 0, True, TOUCHBACK_YARDLINE, False)

    returned = ret if result == "returned" else 0
    start = _clamp(100 - landing - returned)
    if fumble:
        return PuntResult("fumble_lost", distance, returned, False, _clamp(100 - start), False)
    return PuntResult(result, distance, returned, True, start, False)


def resolve_punt(yardline_100: int, tables, rng) -> PuntResult:
    return apply_punt(yardline_100, rng.choice(tables.punt[_clamp(yardline_100)]))


@dataclass
class KickoffResult:
    kind: str
    yardline_100: int
    touchdown: bool


def resolve_kickoff(tables, rng) -> KickoffResult:
    kind, start, td = rng.choice(tables.kickoff)
    return KickoffResult(kind, start, bool(td))


def resolve_free_kick(tables, rng) -> KickoffResult:
    kind, start, td = rng.choice(tables.free_kick)
    return KickoffResult(kind, start, bool(td))


@dataclass
class OnsideResult:
    kicking_team_recovers: bool
    yardline_100: int


def resolve_onside(tables, rng) -> OnsideResult:
    recovered, start = rng.choice(tables.onside)
    return OnsideResult(bool(recovered), start)


def decide_onside(kicker_lead: int, quarter: int, game_seconds_remaining: int, tables, rng) -> bool:
    if quarter != 4 or kicker_lead >= 0 or game_seconds_remaining <= 0:
        return False
    kwin = ("k120" if game_seconds_remaining <= 120 else "k300" if game_seconds_remaining <= 300
            else "k600" if game_seconds_remaining <= 600 else "k900")
    found = tables.lookup_or_none("onside_choice", {"late_score": late_score_bucket(kicker_lead), "kwin": kwin})
    return found is not None and rng.random() < found[1]["p_onside"]


@dataclass
class TryResult:
    points: int
    kind: str


def decide_two_point(score_diff_after_td: int, quarter: int, tables, rng) -> bool:
    situation = {"sd_key": max(-20, min(20, score_diff_after_td)),
                 "phase": "late" if quarter >= 4 else "early",
                 "score_b": score_bucket(score_diff_after_td)}
    _, row = tables.lookup("try_choice", situation)
    return rng.random() < row["p_two"]


def resolve_try(go_for_two: bool, tables, rng) -> TryResult:
    if go_for_two:
        if rng.random() < tables.two_point["p_success"]:
            return TryResult(2, "two_good")
        return TryResult(0, "two_failed")
    u = rng.random()
    xp = tables.extra_point
    if u < xp["p_good"]:
        return TryResult(1, "xp_good")
    if u < xp["p_good"] + xp["p_failed"]:
        return TryResult(0, "xp_failed")
    return TryResult(0, "xp_blocked")


def choose_fourth_down(state: GameState, tables, rng) -> str:
    found = tables.lookup_or_none("fourth_down_ot", state.late_situation()) if state.quarter == 5 else None
    row = found[1] if found else tables.lookup_late("fourth_down_late", "fourth_down", state)[1]
    u = rng.random() * (row["p_go"] + row["p_punt"] + row["p_fg"])
    call = GO if u < row["p_go"] else PUNT if u < row["p_go"] + row["p_punt"] else FIELD_GOAL
    end_of_half = tables.half_end_fourth_call(state)
    if end_of_half is not None:
        p_fg, p_go = end_of_half
        v = rng.random()
        call = FIELD_GOAL if v < p_fg else GO if v < p_fg + p_go else PUNT

    # the table picks the call, these guards fix the cases where it would do something dumb
    yl, seconds = state.yardline_100, state.game_seconds_remaining
    late = state.quarter >= 4 and seconds <= ENDGAME_SECONDS

    if call == FIELD_GOAL and kick_distance(state) > MAX_KICK_DISTANCE:
        call = PUNT if yl >= MIN_PUNT_YARDLINE else GO
    if late and -8 <= state.score_diff <= -4 and call == FIELD_GOAL:
        call = GO
    if late and state.score_diff < 0 and call == PUNT:
        call = GO
    if late and state.score_diff > 0 and yl >= OWN_HALF_YARDLINE:
        call = PUNT
    if call == PUNT and yl < MIN_PUNT_YARDLINE:
        call = FIELD_GOAL if kick_distance(state) <= MAX_KICK_DISTANCE else GO
    if state.quarter == 5 and state.score_diff < 0:
        if call == PUNT or (call == FIELD_GOAL and state.score_diff < -MAX_DEFICIT_FOR_FIELD_GOAL):
            call = GO
    return call
