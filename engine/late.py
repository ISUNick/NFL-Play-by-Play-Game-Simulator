from engine import clock_rules as rules
from engine.state import GameState

MAX_SECONDS = 240
MIN_SECONDS_FOR_DEFENSE_TIMEOUT = 20
MUST_STOP_CLOCK_SECONDS = 10  # in field goal range with this little time the offense stops the clock
HURRY_LEAVES_SECONDS = 3  # a team that must score rushes to the line and leaves itself 3 seconds


def _row(table: str, state: GameState, tables):
    if state.quarter != 4 or not state.clock_running or not 0 < state.clock <= MAX_SECONDS:
        return None
    found = tables.lookup_or_none(table, state.late_timeout_situation())
    return None if found is None else found[1]


def offense_timeout(state: GameState, tables, rng, runoff: int = 0) -> bool:
    if state.offense_timeouts <= 0 or state.score_diff > 0:
        return False
    if (state.clock_running and state.clock > 0 and state.down <= 3
            and rules.kick_would_tie_or_win(state) and rules.kick_distance(state) <= rules.SURE_KICK_RANGE
            and state.clock - hurry_runoff(state, runoff) <= MUST_STOP_CLOCK_SECONDS):
        return True
    row = _row("late_timeout_off", state, tables)
    return row is not None and rng.random() < row["p_timeout"]


def defense_timeout(state: GameState, tables, rng) -> bool:
    if state.defense_timeouts <= 0 or state.score_diff > 0:
        return False
    if state.clock < MIN_SECONDS_FOR_DEFENSE_TIMEOUT and state.down < 4:
        return False
    row = _row("late_timeout_def", state, tables)
    return row is not None and rng.random() < row["p_timeout"]


def hurry_runoff(state: GameState, runoff: int) -> int:
    if state.quarter == 4 and rules.must_score(state) and state.clock_running and runoff >= state.clock:
        return max(0, state.clock - HURRY_LEAVES_SECONDS)
    return runoff
