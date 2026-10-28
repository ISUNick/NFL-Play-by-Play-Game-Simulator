from engine.play import PlayOutcome
from engine.special import KickoffResult, PuntResult, TryResult

ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def clock_text(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def spot_text(yardline_100: int) -> str:
    if yardline_100 > 50:
        return f"own {100 - yardline_100}"
    if yardline_100 < 50:
        return f"opp {yardline_100}"
    return "midfield"


def down_text(down: int, distance: int, yardline_100: int) -> str:
    if distance >= yardline_100:
        return f"{ORDINAL[down]} & Goal"
    return f"{ORDINAL[down]} & {distance}"


def _yards(y: int) -> str:
    if y == 0:
        return "no gain"
    word = "yard" if abs(y) == 1 else "yards"
    return f"{y} {word}" if y > 0 else f"loss of {abs(y)} {word}"


def describe_play(o: PlayOutcome, offense: str, defense: str) -> str:
    qb, target, rusher = o.passer or "QB", o.target or "a receiver", o.rusher or offense
    if o.kind == "run":
        text = f"{rusher} rush, {_yards(o.yards)}"
    elif o.kind == "scramble":
        text = f"{rusher} scrambles, {_yards(o.yards)}"
    elif o.kind == "sack":
        text = f"{qb} sacked, {_yards(o.yards)}" if o.yards < 0 else f"{qb} sacked at the line"
    elif o.kind == "complete":
        text = f"{qb} pass complete to {target}, {_yards(o.yards)}"
    elif o.kind == "incomplete":
        text = f"{qb} pass incomplete, intended for {target}"
    else:
        text = f"{qb} pass intended for {target} INTERCEPTED by {defense}"
    if o.touchdown:
        text += ". TOUCHDOWN!"
    elif o.safety:
        text += ". SAFETY!"
    elif o.defensive_td:
        text += ", returned for a TOUCHDOWN!"
    elif o.fumble_lost:
        text += f". FUMBLE, recovered by {defense}"
    elif o.first_down:
        text += ". First down"
    return text


def describe_kickoff(r: KickoffResult, kicker: str, receiver: str) -> str:
    if r.touchdown:
        return f"{kicker} kicks off, {receiver} return for a TOUCHDOWN!"
    how = {"touchback": "touchback", "fair_catch": "fair catch", "oob": "out of bounds"}.get(r.kind, "returned")
    return f"{kicker} kicks off, {how}. {receiver} ball at the {spot_text(r.yardline_100)}"


def describe_free_kick(r: KickoffResult, kicker: str, receiver: str) -> str:
    if r.touchdown:
        return f"{kicker} free kick after the safety, {receiver} return for a TOUCHDOWN!"
    how = {"touchback": "touchback", "fair_catch": "fair catch", "oob": "out of bounds"}.get(r.kind, "returned")
    return f"{kicker} free kick after the safety, {how}. {receiver} ball at the {spot_text(r.yardline_100)}"


def describe_onside(r, kicker: str, receiver: str) -> str:
    if r.kicking_team_recovers:
        return f"{kicker} onside kick, RECOVERED by {kicker}! {kicker} ball at the {spot_text(r.yardline_100)}"
    return f"{kicker} onside kick, recovered by {receiver}. {receiver} ball at the {spot_text(r.yardline_100)}"


def describe_punt(r: PuntResult, punter: str, receiver: str) -> str:
    if r.kind == "blocked":
        who = receiver if r.receiver_has_ball else punter
        return f"{punter} punt BLOCKED, {who} ball at the {spot_text(r.yardline_100)}"
    if r.kind == "return_td":
        return f"{punter} punts {r.distance}, {receiver} return for a TOUCHDOWN!"
    if r.kind == "fumble_lost":
        return f"{punter} punts {r.distance}, FUMBLE, recovered by {punter} at the {spot_text(r.yardline_100)}"
    how = {"touchback": "touchback", "fair_catch": "fair catch", "oob": "out of bounds", "downed": "downed",
           "returned": f"returned {r.return_yards}"}[r.kind]
    return f"{punter} punts {r.distance}, {how}. {receiver} ball at the {spot_text(r.yardline_100)}"


def describe_field_goal(made: bool, distance: int, team: str, defense: str, new_yardline_100: int) -> str:
    if made:
        return f"{team} {distance} yard field goal is GOOD"
    return f"{team} {distance} yard field goal is NO GOOD. {defense} ball at the {spot_text(new_yardline_100)}"


def describe_try(r: TryResult, team: str) -> str:
    return {"xp_good": f"{team} extra point is good",
            "xp_failed": f"{team} extra point is no good",
            "xp_blocked": f"{team} extra point is BLOCKED",
            "two_good": f"{team} two point try is GOOD",
            "two_failed": f"{team} two point try fails"}[r.kind]
