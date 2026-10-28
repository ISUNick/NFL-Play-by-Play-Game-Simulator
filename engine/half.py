from engine import clock_rules as rules
from engine.clock_rules import Action
from engine.special import MAX_KICK_DISTANCE
from engine.state import GameState, half_window_bucket, half_zone_bucket

SNAP_SECONDS = 120  # kneel / field goal decisions only in the last 2 minutes of the half
MIN_SECONDS_FOR_DEFENSE_TIMEOUT = 20
TIMEOUT_SECONDS = 240  # timeouts and spikes only in the last 4 minutes
MIN_KNEEL_YARDLINE = 56  # only kneel deep in your own territory


def in_snap_window(state: GameState) -> bool:
    return state.quarter == 2 and 0 < state.half_seconds_remaining <= SNAP_SECONDS


def can_kneel(state: GameState) -> bool:
    return (state.down <= rules.LAST_KNEEL_DOWN
            and state.yardline_100 >= MIN_KNEEL_YARDLINE
            and state.yardline_100 + rules.KNEEL_YARDS_LOST < 100)


def decide_before_snap(state: GameState, tables, rng, kneeling: bool) -> Action:
    if not in_snap_window(state) or state.down > rules.LAST_KNEEL_DOWN:
        return Action.NORMAL
    if kneeling and can_kneel(state):
        return Action.KNEEL
    found = tables.lookup_or_none("half_snap", state.half_situation())
    if found is None:
        return Action.NORMAL
    row = found[1]
    u = rng.random()
    if u < row["p_kneel"]:
        return Action.KNEEL if can_kneel(state) else Action.NORMAL
    u -= row["p_kneel"]
    if u < row["p_fg"]:
        return Action.FIELD_GOAL if rules.kick_distance(state) <= MAX_KICK_DISTANCE else Action.NORMAL
    return Action.NORMAL


def _timeout_row(table: str, state: GameState, tables):
    if state.quarter != 2 or not state.clock_running or not 0 < state.clock <= TIMEOUT_SECONDS:
        return None
    sit = {"hwin": half_window_bucket(state.clock), "hzone": half_zone_bucket(state.yardline_100),
           "hdown": min(state.down, 4)}
    found = tables.lookup_or_none(table, sit)
    return None if found is None else found[1]


def offense_spike(state: GameState, tables, rng):
    if state.offense_timeouts > 0 or state.down > 3:
        return None
    row = _timeout_row("half_spike", state, tables)
    if row is None or rng.random() >= row["p_spike"]:
        return None
    return min(rng.choice(tables.spike_runoff), max(0, state.clock - 2))


def offense_timeout(state: GameState, tables, rng) -> bool:
    if state.offense_timeouts <= 0:
        return False
    row = _timeout_row("half_timeout_off", state, tables)
    return row is not None and rng.random() < row["p_timeout"]


def defense_timeout(state: GameState, tables, rng) -> bool:
    if state.defense_timeouts <= 0:
        return False
    if state.clock < MIN_SECONDS_FOR_DEFENSE_TIMEOUT and state.down < 4:
        return False
    row = _timeout_row("half_timeout_def", state, tables)
    return row is not None and rng.random() < row["p_timeout"]
