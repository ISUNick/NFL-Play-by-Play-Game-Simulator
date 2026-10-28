from enum import Enum

from engine.state import GameState

KNEEL_SECONDS = 40  # a kneel uses the whole play clock

KNEEL_AFTER_TIMEOUT_SECONDS = 2  # defense timeout stops the clock, only the play burns

KNEEL_YARDS_LOST = 2  # too close to your own goal line a kneel is a safety

BLOWOUT_LEAD = 17
BLOWOUT_SECONDS = 60  # with a lead this big teams kneel even if the clock wouldn't run out

LAST_KNEEL_DOWN = 3  # on 4th down they punt

KICK_DISTANCE_OFFSET = 18  # kick distance = yard line + 18 (end zone plus snap and hold)

MAX_DEFICIT_FOR_FIELD_GOAL = 3  # a field goal ties or wins

MAX_DEFICIT_TO_MANAGE_CLOCK = 8  # down one score, spikes and timeouts matter

SURE_KICK_RANGE = 55  # real teams kick from here on 4th down when it's tied or they trail by 3 or less
LATE_KICK_RANGE = 60  # 56 to 60 yards only when time is almost gone
LATE_KICK_SECONDS = 30
FOURTH_DOWN_WINDOW_SECONDS = 120  # before 2:00 the normal 4th down logic decides

KICK_NOW_SECONDS_NO_TIMEOUTS = 10  # 1st to 3rd down, clock stopped: kick when there's no time for another play
KICK_NOW_SECONDS_WITH_TIMEOUTS = 6

TIMEOUT_SECONDS = 20  # offense calls timeout after an in bounds play with this little time left

SPIKE_SECONDS_IN_RANGE = 20  # spike costs a down so only on 1st or 2nd
SPIKE_SECONDS_NEED_TOUCHDOWN = 10
LAST_SPIKE_DOWN = 2

LATE_QUARTERS = (4, 5)  # 4th quarter and overtime


class Action(Enum):
    NORMAL = "normal"
    KNEEL = "kneel"
    FIELD_GOAL = "field_goal"
    SPIKE = "spike"
    TIMEOUT = "timeout"


def kick_distance(state: GameState) -> int:
    return state.yardline_100 + KICK_DISTANCE_OFFSET


def kick_would_tie_or_win(state: GameState) -> bool:
    return -MAX_DEFICIT_FOR_FIELD_GOAL <= state.score_diff <= 0


def must_score(state: GameState) -> bool:
    return -MAX_DEFICIT_TO_MANAGE_CLOCK <= state.score_diff <= 0


def kneel_clock_burn(downs_left: int, defense_timeouts: int) -> int:
    timeouts_that_matter = min(defense_timeouts, downs_left)
    full_kneels = downs_left - timeouts_that_matter
    return full_kneels * KNEEL_SECONDS + timeouts_that_matter * KNEEL_AFTER_TIMEOUT_SECONDS


def can_run_out_clock(state: GameState) -> bool:
    kneels_left = LAST_KNEEL_DOWN - state.down + 1
    return state.game_seconds_remaining <= kneel_clock_burn(kneels_left, state.defense_timeouts)


def is_blowout(state: GameState) -> bool:
    return (state.score_diff >= BLOWOUT_LEAD
            and state.game_seconds_remaining <= BLOWOUT_SECONDS)


def should_kneel(state: GameState) -> bool:
    if state.quarter != 4:
        return False
    if state.score_diff <= 0:
        return False
    if state.game_seconds_remaining <= 0:
        return False
    if state.down > LAST_KNEEL_DOWN:
        return False
    if state.yardline_100 + KNEEL_YARDS_LOST >= 100:
        return False
    return can_run_out_clock(state) or is_blowout(state)


def should_kick(state: GameState) -> bool:
    if state.quarter not in LATE_QUARTERS or not kick_would_tie_or_win(state):
        return False
    seconds = state.game_seconds_remaining
    if seconds <= 0:
        return False
    distance = kick_distance(state)

    if state.down == 4:
        if seconds > FOURTH_DOWN_WINDOW_SECONDS:
            return False
        return distance <= SURE_KICK_RANGE or (
            distance <= LATE_KICK_RANGE and seconds <= LATE_KICK_SECONDS)

    if state.clock_running or distance > SURE_KICK_RANGE:
        return False
    limit = (KICK_NOW_SECONDS_NO_TIMEOUTS if state.offense_timeouts == 0
             else KICK_NOW_SECONDS_WITH_TIMEOUTS)
    return seconds <= limit


def should_spike(state: GameState) -> bool:
    if state.quarter not in LATE_QUARTERS or not state.clock_running:
        return False
    if state.offense_timeouts > 0:
        return False
    if state.down > LAST_SPIKE_DOWN:
        return False
    if not must_score(state):
        return False
    in_range = kick_would_tie_or_win(state) and kick_distance(state) <= SURE_KICK_RANGE
    limit = SPIKE_SECONDS_IN_RANGE if in_range else SPIKE_SECONDS_NEED_TOUCHDOWN
    return 1 < state.game_seconds_remaining <= limit


def should_call_timeout(state: GameState) -> bool:
    if state.quarter not in LATE_QUARTERS or not state.clock_running:
        return False
    if state.offense_timeouts <= 0:
        return False
    if state.down > 3:
        return False
    if not must_score(state):
        return False
    return 0 < state.game_seconds_remaining <= TIMEOUT_SECONDS


DEFENSE_TIMEOUT_RATES = [(60, 0.906), (120, 0.785), (180, 0.689), (240, 0.462), (300, 0.216), (420, 0.057)]

DEFENSE_TIMEOUT_BLOWOUT_LEAD = 16
DEFENSE_TIMEOUT_BLOWOUT_RATE = 0.14
DEFENSE_TIMEOUT_BLOWOUT_SECONDS = 180


def defense_timeout_probability(state: GameState) -> float:
    if state.quarter != 4 or not state.clock_running or state.defense_timeouts <= 0:
        return 0.0
    if state.score_diff <= 0 or state.down > 3:
        return 0.0
    seconds = state.game_seconds_remaining
    if seconds <= 0:
        return 0.0
    if state.score_diff > DEFENSE_TIMEOUT_BLOWOUT_LEAD:
        return DEFENSE_TIMEOUT_BLOWOUT_RATE if seconds <= DEFENSE_TIMEOUT_BLOWOUT_SECONDS else 0.0
    for limit, chance in DEFENSE_TIMEOUT_RATES:
        if seconds <= limit:
            return chance
    return 0.0


KNEEL_TIMEOUT_BY_LEAD = [(3, 0.207), (8, 0.254), (16, 0.076)]
KNEEL_TIMEOUT_BLOWOUT = 0.008
KNEEL_TIMEOUT_HALF_LAST30 = 0.014
KNEEL_TIMEOUT_HALF_EARLIER = 0.07


def defense_kneel_timeout_probability(state: GameState) -> float:
    if state.defense_timeouts <= 0:
        return 0.0
    if state.quarter == 2:
        return KNEEL_TIMEOUT_HALF_LAST30 if state.half_seconds_remaining <= 30 else KNEEL_TIMEOUT_HALF_EARLIER
    if state.score_diff <= 0:
        return 0.0
    for limit, chance in KNEEL_TIMEOUT_BY_LEAD:
        if state.score_diff <= limit:
            return chance
    return KNEEL_TIMEOUT_BLOWOUT


def decide_after_play(state: GameState) -> Action:
    if should_call_timeout(state):
        return Action.TIMEOUT
    return Action.NORMAL


def decide_before_snap(state: GameState) -> Action:
    if should_kneel(state):
        return Action.KNEEL
    if should_kick(state):
        return Action.FIELD_GOAL
    if should_spike(state):
        return Action.SPIKE
    return Action.NORMAL
