from dataclasses import dataclass, replace
from typing import Optional

from engine.state import GameState, fourth_distance_bucket
from engine.teams import ZERO_EDGE, logit_shift, pick_tilted

TOUCHBACK_YARDLINE = 80


@dataclass
class PlayOutcome:
    kind: str
    timing_kind: str
    yards: int
    new_yardline_100: int
    first_down: bool = False
    touchdown: bool = False
    safety: bool = False
    turnover: bool = False
    fumble_lost: bool = False
    defensive_td: bool = False
    out_of_bounds: bool = False
    depth: Optional[str] = None
    passer: Optional[str] = None
    target: Optional[str] = None
    rusher: Optional[str] = None


def _feasible_depth(depth: str, yardline_100: int) -> str:
    # can't throw deep from inside the 20 or medium from inside the 10
    if depth == "deep" and yardline_100 < 20:
        depth = "medium"
    if depth == "medium" and yardline_100 < 10:
        depth = "short"
    return depth


def _finish(state: GameState, kind: str, yards: int, oob: bool, td: bool, fumble_key: str,
            tables, rng, depth: Optional[str] = None) -> PlayOutcome:
    yl = state.yardline_100
    timing = {"run": "run_in", "scramble": "scramble_in", "sack": "sack", "complete": "complete_in"}[kind]

    if td:
        timing = "pass_td" if kind == "complete" else "run_td"
        return PlayOutcome(kind, timing, yl, 0, first_down=True, touchdown=True, depth=depth)

    yards = min(yards, yl - 1)
    spot = yl - yards
    if spot >= 100:
        return PlayOutcome(kind, timing, yards, 0, safety=True, depth=depth)

    if rng.random() < tables.fumble_lost[fumble_key]:
        if kind in ("run", "scramble"):
            timing = "run_fumble"
        return PlayOutcome(kind, timing, yards, max(1, min(99, 100 - spot)), turnover=True,
                           fumble_lost=True, depth=depth)

    if oob and kind != "sack":
        timing = {"run": "run_oob", "scramble": "scramble_oob", "complete": "complete_oob"}[kind]
    return PlayOutcome(kind, timing, yards, spot, first_down=yards >= state.distance,
                       out_of_bounds=oob and kind != "sack", depth=depth)


def _interception(state: GameState, depth: str, tables, rng) -> PlayOutcome:
    air_yards, return_yards, return_td = rng.choice(tables.interception[depth])
    catch = min(99, state.yardline_100 - air_yards)
    base = dict(kind="interception", timing_kind="interception", yards=0, turnover=True, depth=depth)
    if catch <= 0:
        return PlayOutcome(new_yardline_100=TOUCHBACK_YARDLINE, **base)
    if return_td:
        return PlayOutcome(new_yardline_100=0, defensive_td=True, **base)
    spot = 100 - catch - return_yards
    return PlayOutcome(new_yardline_100=max(1, min(99, spot)), **base)


def _tilted_depth(p_short: float, p_medium: float, receiver, league_mix: dict) -> tuple:
    # the situation says how likely each depth is, the receiver's own mix says how far he leans from the league
    p = [p_short, p_medium, max(0.0, 1.0 - p_short - p_medium)]
    odds = [receiver.depth_share[d] / league_mix[d] for d in ("short", "medium", "deep")]
    mixed = [a * b for a, b in zip(p, odds)]
    total = sum(mixed)
    return mixed[0] / total, mixed[1] / total


def _throw(state: GameState, situation: dict, tables, rng, edge=ZERO_EDGE, unit=None) -> PlayOutcome:
    suffix = {3: "_third", 4: "_fourth"}.get(state.down, "")
    if suffix:
        situation = dict(situation, fdist=fourth_distance_bucket(state.distance))
    receiver = None
    if unit is not None:
        receiver = unit.pick_receiver(rng)
        edge = replace(edge, completion=edge.completion + unit.qb.comp_shift + receiver.comp_shift,
                       interception=edge.interception + unit.qb.int_shift,
                       completion_tilt=edge.completion_tilt + (unit.qb.yards_effect + receiver.yards_effect) / unit.comp_variance)
    _, d = tables.lookup("pass_depth" + suffix, situation)
    p_short, p_medium = d["p_short"], d["p_medium"]
    if receiver is not None:
        p_short, p_medium = _tilted_depth(p_short, p_medium, receiver, unit.league_mix[receiver.position])
    u = rng.random()
    depth = "short" if u < p_short else "medium" if u < p_short + p_medium else "deep"
    depth = _feasible_depth(depth, state.yardline_100)

    _, r = tables.lookup("pass_result" + suffix, dict(situation, depth=depth))
    p_complete, p_int = r["p_complete"], r["p_int"]
    if edge.completion or edge.interception:
        p_complete = logit_shift(p_complete, edge.completion)
        p_int = logit_shift(p_int, edge.interception)
        if p_complete + p_int > 1:
            total = p_complete + p_int
            p_complete, p_int = p_complete / total, p_int / total
    u = rng.random()
    if u < p_complete:
        yards, oob, td = pick_tilted(tables.completion[(depth, state.yardline_100)], edge.completion_tilt, rng, tables)
        out = _finish(state, "complete", yards, bool(oob), bool(td), "completion", tables, rng, depth)
    elif u < p_complete + p_int:
        out = _interception(state, depth, tables, rng)
    else:
        out = PlayOutcome("incomplete", "incomplete", 0, state.yardline_100, out_of_bounds=True, depth=depth)
    if unit is not None:
        out.passer, out.target = unit.qb.name, receiver.name
    return out


def run_scrimmage_play(state: GameState, tables, rng, edge=ZERO_EDGE, unit=None) -> PlayOutcome:
    situation = state.situation()
    yl = state.yardline_100

    if state.down == 4:
        sit = dict(situation, fdist=fourth_distance_bucket(state.distance))
        _, call = tables.lookup("play_call_fourth", sit)
    else:
        _, call = tables.lookup_late("play_call_late", "play_call", state)
    final = tables.final_minute_dropback(state)
    dropback = rng.random() < (call["p_dropback"] if final is None else final)
    if not dropback:
        runner = unit.pick_runner(rng) if unit is not None else None
        tilt = edge.run_tilt + (runner.yards_effect / unit.run_variance if runner is not None else 0.0)
        yards, oob, td = pick_tilted(tables.run[(situation["dist_b"], yl)], tilt, rng, tables)
        out = _finish(state, "run", yards, bool(oob), bool(td), "run", tables, rng)
        if runner is not None:
            out.rusher = runner.name
        return out

    _, split = tables.lookup("dropback_split", situation)
    shift_sack = edge.sack + (unit.qb.sack_shift if unit is not None else 0.0)
    p_sack = logit_shift(split["p_sack"], shift_sack) if shift_sack else split["p_sack"]
    p_scramble = split["p_scramble"]
    if unit is not None and unit.qb.scramble_shift:
        p_scramble = logit_shift(p_scramble, unit.qb.scramble_shift)
    qb_name = unit.qb.name if unit is not None else None
    u = rng.random()
    if u < p_sack:
        yards, _, _ = rng.choice(tables.sack[yl])
        out = _finish(state, "sack", yards, False, False, "sack", tables, rng)
        out.passer = qb_name
        return out
    if u < p_sack + p_scramble:
        yards, oob, td = rng.choice(tables.scramble[yl])
        out = _finish(state, "scramble", yards, bool(oob), bool(td), "scramble", tables, rng)
        out.rusher = qb_name
        return out
    return _throw(state, situation, tables, rng, edge, unit)
