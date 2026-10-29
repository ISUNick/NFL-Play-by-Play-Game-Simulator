import copy
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from engine import clock_rules as rules
from engine import half
from engine import late
from engine import text
from engine.clock_rules import Action
from engine.penalty import play_kind, resolve_penalty, roll_post_snap, roll_pre_snap
from engine.play import PlayOutcome, run_scrimmage_play
from engine.teams import ZERO_EDGE, add_edges
from engine.special import (FIELD_GOAL, PUNT, choose_fourth_down, decide_onside, decide_two_point,
                            resolve_free_kick, resolve_kickoff, resolve_onside, resolve_punt, resolve_try)


from engine.state import GameState, score_bucket

QUARTER_SECONDS = 900
HALF_TIMEOUTS = 3
OT_SECONDS = 600
OT_TIMEOUTS = 2
TWO_MINUTE_WARNING = 120
PLAY_SECONDS = 6
SPIKE_SECONDS = 1
RUSHED_KICK_SECONDS = 30
MIN_PLAY_SECONDS = 3
MAX_STEPS = 3000


@dataclass
class Event:
    kind: str
    team: int
    state: GameState
    text: str
    detail: Any = None
    seconds: int = 0
    scorer: Optional[int] = None
    points: int = 0
    scores: tuple = (0, 0)


class Game:
    # team 0 is the home team. yardline_100 is always from the view of the team with the ball
    def __init__(self, tables, rng, names=("HOME", "AWAY"), edges=None, playoff=False, home_field=False, units=None):
        self.tables = tables
        self.rng = rng
        self.edges = edges
        era = getattr(tables, "era_edge", None)
        if era is not None:
            both = edges or (ZERO_EDGE, ZERO_EDGE)
            self.edges = (add_edges(both[0], era), add_edges(both[1], era))
        self.units = units
        self.playoff = playoff
        self.home_field = home_field
        self.names = list(names)
        self.scores = [0, 0]
        self.timeouts = [HALF_TIMEOUTS, HALF_TIMEOUTS]
        self.quarter = 1
        self.clock = QUARTER_SECONDS
        receiver = rng.randint(0, 1)
        self.kicker = 1 - receiver
        self.second_half_kicker = receiver
        self.offense = receiver
        self.phase = "kickoff"
        self.down, self.distance, self.yardline_100 = 1, 10, 75
        self.clock_running = False
        self.runoff = 0
        self.borrowed = 0  # the second put back on the clock for an untimed down, it is not anyone's possession time
        self.try_team: Optional[int] = None
        self.free_kick = False
        self.fresh = True
        self.kneeling_team: Optional[int] = None
        self.after_kneel = False
        # overtime phases: None, then "first", "second", "sudden"
        self.ot_phase: Optional[str] = None
        self.ot_first: Optional[int] = None
        self.ot_ended = False
        self.went_to_ot = False
        self.over = False
        self.steps = 0
        self.stats = [Counter(), Counter()]
        self.players = [{}, {}]
        self._rz_marked = False
        self._queue: list = []

    def fork(self, seed) -> "Game":
        # an independent copy of the game as it stands, with its own random stream. Tables and ratings are shared
        shared = {id(x): x for x in (self.tables, self.edges, self.units) if x is not None}
        twin = copy.deepcopy(self, shared)
        twin.rng = random.Random(seed)
        return twin

    def state(self) -> GameState:
        return self._state_for(self.offense)

    def _state_for(self, team: int) -> GameState:
        return GameState(quarter=self.quarter, clock=self.clock, down=self.down, distance=self.distance,
                         yardline_100=self.yardline_100, offense_score=self.scores[team],
                         defense_score=self.scores[1 - team], offense_timeouts=self.timeouts[team],
                         defense_timeouts=self.timeouts[1 - team], clock_running=self.clock_running,
                         new_possession=self.fresh and team == self.offense)

    def _event(self, kind, team, st, line, detail=None, seconds=0, scorer=None, points=0) -> Event:
        return Event(kind, team, st, line, detail, seconds, scorer, points, tuple(self.scores))

    def _header(self, st: GameState) -> str:
        label = "OT" if st.quarter >= 5 else f"Q{st.quarter}"
        return f"{label} {text.clock_text(st.clock)}"

    def _where(self, down: int, distance: int, yardline_100: int) -> str:
        return f"{text.down_text(down, distance, yardline_100)} at {text.spot_text(yardline_100)}"

    def _tick(self, seconds: int, running: bool) -> bool:
        # a running clock stops at the two minute warning, returns True when that happens
        warning = False
        if (running and self.quarter in (2, 4, 5) and self.clock > TWO_MINUTE_WARNING
                and self.clock - seconds <= TWO_MINUTE_WARNING):
            seconds = self.clock - TWO_MINUTE_WARNING
            warning = True
        # time of possession: the receiver on a kickoff, the scoring team on a try, otherwise the offense
        holder = (self.offense if self.phase == "scrimmage" else
                  1 - self.kicker if self.phase == "kickoff" and self.kicker is not None else
                  self.try_team if self.phase == "try" and self.try_team is not None else None)
        if holder is not None:
            self.stats[holder]["top"] += max(0, min(seconds, self.clock) - self.borrowed)
        if seconds > 0:
            self.borrowed = 0
        self.clock = max(0, self.clock - seconds)
        return warning

    def _new_series(self, yardline_100: int) -> None:
        self.yardline_100 = yardline_100
        self.down = 1
        self.distance = min(10, yardline_100)

    def _change_possession(self, new_offense_yardline: int) -> None:
        if self.quarter == 5:
            self._ot_possession_ended()
        self.offense = 1 - self.offense
        self._new_series(new_offense_yardline)
        self.clock_running = False
        self.fresh = True
        self.kneeling_team = None

    def _score_touchdown(self, team: int, defensive: bool = False) -> None:
        self.scores[team] += 6
        self.clock_running = False
        self.runoff = 0
        if self.quarter == 5 and not self._ot_try_needed(team, defensive):
            self._ot_end()
            return
        self.try_team = team
        self.phase = "try"

    def _score_then_kick_off(self, team: int, free_kick: bool = False) -> None:
        self.kicker = team
        self.free_kick = free_kick
        self.phase = "kickoff"
        self.clock_running = False
        self.runoff = 0
        self.kneeling_team = None

    def step(self) -> Event:
        if self.over and not self._queue:
            raise RuntimeError("the game is over")
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise RuntimeError(f"game did not finish in {MAX_STEPS} events, something is stuck")
        if self._queue:
            return self._queue.pop(0)
        handler = {"kickoff": self._kickoff, "scrimmage": self._scrimmage, "try": self._try}[self.phase]
        event = handler()
        self._end_of_period()
        return event

    def _end_of_period(self) -> None:
        if self.over or self.phase == "try":
            return
        if self.quarter == 5:
            if self.clock > 0 and not self.ot_ended:
                return
            if self.playoff and self.scores[0] == self.scores[1]:
                self._next_ot_period()
                return
            self._finish_game()
            return
        if self.clock > 0:
            return
        st = self.state()
        if self.quarter in (1, 3):
            self._queue.append(self._event("quarter_end", self.offense, st, f"End of quarter {self.quarter}"))
            self.quarter += 1
            self.clock = QUARTER_SECONDS
        elif self.quarter == 2:
            self._queue.append(self._event("halftime", self.offense, st, self._score_line("Halftime")))
            self.quarter, self.clock = 3, QUARTER_SECONDS
            self.timeouts = [HALF_TIMEOUTS, HALF_TIMEOUTS]
            self.kicker = self.second_half_kicker
            self.phase = "kickoff"
            self.clock_running = False
            self.runoff = 0
            self.try_team = None
        elif self.scores[0] == self.scores[1]:
            self._start_overtime(st)
        else:
            self._finish_game()

    def _finish_game(self) -> None:
        self.over = True
        label = "Final (OT)" if self.went_to_ot else "Final"
        line = self._score_line(label) + (", tie" if self.scores[0] == self.scores[1] else "")
        self._queue.append(self._event("game_end", self.offense, self.state(), line))

    def _score_line(self, label: str) -> str:
        return f"{label}: {self.names[0]} {self.scores[0]}, {self.names[1]} {self.scores[1]}"

    def _start_overtime(self, st: GameState) -> None:
        receiver = self.rng.randint(0, 1)
        self._queue.append(self._event("end_regulation", self.offense, st,
                                       self._score_line("End of regulation") + ", tied"))
        self.quarter, self.clock = 5, OT_SECONDS
        self.timeouts = [OT_TIMEOUTS, OT_TIMEOUTS]
        self.ot_phase, self.ot_first, self.went_to_ot = "first", receiver, True
        self.kicker, self.offense = 1 - receiver, receiver
        self.phase = "kickoff"
        self.clock_running, self.runoff, self.try_team, self.kneeling_team = False, 0, None, None
        self.after_kneel = False
        line = (f"Overtime. {self.names[receiver]} wins the coin toss and receives. "
                f"10:00, {OT_TIMEOUTS} timeouts each")
        self._queue.append(self._event("overtime", receiver, self.state(), line))

    def _next_ot_period(self) -> None:
        # playoff games only: another full period, next score wins
        self.clock, self.timeouts = 900, [3, 3]
        self.ot_phase, self.ot_ended = "sudden", False
        self.try_team = None
        self._queue.append(self._event("overtime", self.offense, self.state(),
                                       self._score_line("Another overtime period") + ", still tied"))

    def _ot_end(self) -> None:
        self.ot_ended = True
        self.try_team = None
        self.clock_running, self.runoff = False, 0

    def _ot_try_needed(self, team: int, defensive: bool) -> bool:
        if defensive or self.ot_phase == "sudden":
            return False
        if self.ot_phase == "first":
            return self.clock > 0
        return self.scores[team] <= self.scores[1 - team]

    def _ot_after_try(self, team: int) -> None:
        if self.ot_phase == "first":
            self.ot_phase = "second"
        elif self.ot_phase == "second":
            tied = self.scores[team] == self.scores[1 - team]
            if tied and self.clock > 0:
                self.ot_phase = "sudden"
            else:
                self._ot_end()

    def _ot_after_field_goal(self, team: int) -> None:
        if self.ot_phase == "sudden":
            self._ot_end()
        elif self.ot_phase == "first":
            if self.clock > 0:
                self.ot_phase = "second"
            else:
                self._ot_end()
        else:
            tied = self.scores[team] == self.scores[1 - team]
            if tied and self.clock > 0:
                self.ot_phase = "sudden"
            else:
                self._ot_end()

    def _ot_possession_ended(self) -> None:
        if self.ot_phase == "first":
            self.ot_phase = "sudden"
        elif self.ot_phase == "second":
            self._ot_end()

    def _kickoff(self) -> Event:
        kicker, receiver = self.kicker, 1 - self.kicker
        self.offense = receiver
        self.kneeling_team = None
        st = self._state_for(receiver)
        if self.free_kick:
            onside = False
        else:
            onside = decide_onside(-st.score_diff, st.quarter, st.game_seconds_remaining, self.tables, self.rng)
        if onside:
            return self._onside_kick(kicker, receiver, st)
        free_kick, self.free_kick = self.free_kick, False
        result = resolve_free_kick(self.tables, self.rng) if free_kick else resolve_kickoff(self.tables, self.rng)
        seconds = self.tables.draw_seconds("kickoff", st.tempo, score_bucket(st.score_diff), self.rng)
        self._tick(seconds, False)
        describe = text.describe_free_kick if free_kick else text.describe_kickoff
        line = f"{self._header(st)}  {describe(result, self.names[kicker], self.names[receiver])}"
        points, scorer = 0, None
        if result.touchdown:
            self._score_touchdown(receiver)
            points, scorer = 6, receiver
            self.stats[receiver]["td"] += 1
        else:
            self.phase = "scrimmage"
            self._new_series(result.yardline_100)
            self.clock_running = False
            self.fresh = True
        self.runoff = 0
        return self._event("kickoff", kicker, st, line, result, seconds, scorer, points)

    def _onside_kick(self, kicker: int, receiver: int, st: GameState) -> Event:
        result = resolve_onside(self.tables, self.rng)
        seconds = self.tables.draw_seconds("kickoff", st.tempo, score_bucket(st.score_diff), self.rng)
        self._tick(seconds, False)
        self.stats[kicker]["onside_att"] += 1
        if result.kicking_team_recovers:
            self.stats[kicker]["onside_rec"] += 1
            self.offense = kicker
        self.phase = "scrimmage"
        self._new_series(result.yardline_100)
        self.clock_running = False
        self.runoff = 0
        self.fresh = True
        line = f"{self._header(st)}  {text.describe_onside(result, self.names[kicker], self.names[receiver])}"
        return self._event("onside", kicker, st, line, result, seconds)

    def _try(self) -> Event:
        team = self.try_team
        st = self._state_for(team)
        two = decide_two_point(self.scores[team] - self.scores[1 - team], self.quarter, self.tables, self.rng)
        result = resolve_try(two, self.tables, self.rng)
        self.scores[team] += result.points
        self.stats[team]["two_att" if two else "xp_att"] += 1
        if result.points:
            self.stats[team]["two_made" if two else "xp_made"] += 1
        self.try_team = None
        if self.quarter == 5:
            self._ot_after_try(team)
        self._score_then_kick_off(team)
        line = f"{self._header(st)}  {text.describe_try(result, self.names[team])}"
        return self._event("try", team, st, line, result, 0, team if result.points else None, result.points)

    def _scrimmage(self) -> Event:
        o, d = self.offense, 1 - self.offense
        st = self.state()

        after_kneel, self.after_kneel = self.after_kneel, False
        spiking = False

        # order of things before a snap: timeouts, the runoff, kneel / spike / field goal, penalty, the play
        if self.clock_running and not after_kneel:
            if self.quarter == 2:
                offense_calls = half.offense_timeout(st, self.tables, self.rng)
            elif self.quarter == 4:
                offense_calls = late.offense_timeout(st, self.tables, self.rng, self.runoff)
            else:
                offense_calls = rules.decide_after_play(st) == Action.TIMEOUT
            if not offense_calls and st.tempo == "normal" and self.timeouts[o] > 0:
                offense_calls = self.rng.random() < self.tables.timeout_early["offense"]
            if offense_calls:
                return self._call_timeout(o, st)

            if self.quarter == 2:
                defense_calls = half.defense_timeout(st, self.tables, self.rng)
            elif self.quarter == 4 and st.score_diff <= 0:
                defense_calls = late.defense_timeout(st, self.tables, self.rng)
            else:
                defense_calls = self.rng.random() < rules.defense_timeout_probability(st)
                if self.quarter == 4 and st.clock < late.MIN_SECONDS_FOR_DEFENSE_TIMEOUT and st.down < 4:
                    defense_calls = False
            if not defense_calls and st.tempo == "normal" and self.timeouts[d] > 0:
                defense_calls = self.rng.random() < self.tables.timeout_early["defense"]
            if defense_calls:
                return self._call_timeout(d, st, detail="defense")

            if self.quarter == 2 and self.kneeling_team != o:
                spike_wait = half.offense_spike(st, self.tables, self.rng)
                if spike_wait is not None:
                    self.runoff, spiking = spike_wait, True

        if self.runoff:
            self.runoff = late.hurry_runoff(st, self.runoff)
            warned = self._tick(self.runoff, self.clock_running)
            self.runoff = 0
            if warned:
                self.clock_running = False
                return self._event("two_minute_warning", o, self.state(), "Two minute warning")
            if self.clock == 0:
                return self._event("time", o, st, "The clock runs out")
            st = self.state()

        if spiking and self.clock_running:
            action = Action.SPIKE
        elif self.quarter == 2:
            action = half.decide_before_snap(st, self.tables, self.rng, self.kneeling_team == o)
        else:
            action = rules.decide_before_snap(st)
        if self.fresh:
            self.stats[o]["drives"] += 1
            self._rz_marked = False
        self.fresh = False
        if action == Action.KNEEL:
            return self._kneel(st)
        if action == Action.SPIKE:
            return self._spike(st)
        if action == Action.FIELD_GOAL:
            return self._field_goal(st)

        flagged = roll_pre_snap(st, self.tables, self.rng, self._home_offense())
        if flagged:
            return self._penalty(st, flagged)

        if st.down == 4:
            call = choose_fourth_down(st, self.tables, self.rng)
            if call == PUNT:
                return self._punt(st)
            if call == FIELD_GOAL:
                return self._field_goal(st)
        return self._play(st)

    def _call_timeout(self, team: int, st: GameState, detail=None) -> Event:
        self.timeouts[team] -= 1
        # d1 is the clock used before the timeout, d2 the time from the timeout to the next snap
        d1, d2 = self.rng.choice(self.tables.timeout_clock[st.tempo])
        used = min(self.runoff, max(0, d1 - PLAY_SECONDS)) + d2
        used = min(used, max(0, self.clock - 1))
        self.runoff = 0
        warned = self._tick(used, True)
        self.clock_running = False
        if warned:
            self._queue.append(self._event("two_minute_warning", self.offense, self.state(), "Two minute warning"))
        line = f"{self._header(st)}  {self.names[team]} timeout ({self.timeouts[team]} left)"
        return self._event("timeout", team, st, line, detail=detail)

    def _kneel(self, st: GameState) -> Event:
        o, d = self.offense, 1 - self.offense
        self.kneeling_team = o
        self.after_kneel = True
        defense_stops = self.rng.random() < rules.defense_kneel_timeout_probability(st)
        if defense_stops:
            self.timeouts[d] -= 1
            seconds, self.clock_running = rules.KNEEL_AFTER_TIMEOUT_SECONDS, False
        else:
            seconds, self.clock_running = rules.KNEEL_SECONDS, True
        warned = self._tick(seconds, self.clock_running)
        if warned:
            self.clock_running = False
            self._queue.append(self._event("two_minute_warning", o, self.state(), "Two minute warning"))
        if self.clock == 0:
            self.clock_running = False
        self.yardline_100 += rules.KNEEL_YARDS_LOST
        self.distance += rules.KNEEL_YARDS_LOST
        self.down += 1
        self.stats[o]["kneel"] += 1
        line = f"{self._header(st)}  {self._where(st.down, st.distance, st.yardline_100)}: {self.names[o]} takes a knee"
        if defense_stops:
            line += f" ({self.names[d]} timeout)"
        return self._event("kneel", o, st, line, None, seconds)

    def _home_offense(self):
        return (self.offense == 0) if self.home_field else None

    def _penalty(self, st: GameState, side: str, kind: str = "pre") -> Event:
        o, d = self.offense, 1 - self.offense
        r = resolve_penalty(st, side, self.tables, self.rng, kind, self._home_offense())
        culprit = o if side == "offense" else d
        seconds = r.seconds
        if not self.clock_running and self.quarter in (2, 4, 5) and self.clock <= 120:
            seconds = 0 if kind == "pre" else min(seconds, PLAY_SECONDS)
        warned = self._tick(seconds, self.clock_running)
        # a defensive foul on the last play gives the offense one more snap
        untimed = side == "defense" and self.clock <= 0 and self.quarter in (2, 4, 5)
        if untimed:
            self.clock, self.borrowed = 1, 1
        self.clock_running = False
        self.runoff = 0
        self.yardline_100, self.down, self.distance = r.new_yardline_100, r.down, r.distance
        self.stats[culprit]["penalty"] += 1
        self.stats[culprit]["penalty_yards"] += r.yards
        outcome = "automatic first down" if r.first_down else "replay the down"
        line = (f"{self._header(st)}  {self._where(st.down, st.distance, st.yardline_100)}: "
                f"PENALTY on {self.names[culprit]}, {r.foul.lower()}, {r.yards} yard{'' if r.yards == 1 else 's'}. "
                f"{outcome.capitalize()}: {self._where(self.down, self.distance, self.yardline_100)}")
        if r.halved:
            line += " (half the distance to the goal)"
        if untimed:
            line += ". Time ran out, but a defensive foul gives the offense one more play"
        if warned:
            self._queue.append(self._event("two_minute_warning", self.offense, self.state(), "Two minute warning"))
        return self._event("penalty", o, st, line, r, seconds)

    def _spike(self, st: GameState) -> Event:
        o = self.offense
        self._tick(SPIKE_SECONDS, False)
        self.down += 1
        self.clock_running = False
        self.stats[o]["spike"] += 1
        line = f"{self._header(st)}  {self._where(st.down, st.distance, st.yardline_100)}: {self.names[o]} spikes the ball to stop the clock"
        return self._event("spike", o, st, line, None, SPIKE_SECONDS)

    def _field_goal(self, st: GameState) -> Event:
        o, d = self.offense, 1 - self.offense
        distance = rules.kick_distance(st)
        rushed = st.half_seconds_remaining <= RUSHED_KICK_SECONDS
        made = self.rng.random() < self.tables.fg_make_prob(distance, rushed)
        seconds = self.tables.draw_seconds("field_goal", st.tempo, score_bucket(st.score_diff), self.rng)
        self._tick(seconds, False)
        self.stats[o]["fg_att"] += 1
        points, scorer = 0, None
        if made:
            self.scores[o] += 3
            self.stats[o]["fg_made"] += 1
            points, scorer, new_yl = 3, o, 0
            if self.quarter == 5:
                self._ot_after_field_goal(o)
            self._score_then_kick_off(o)
        else:
            # missed kick, the other team gets it at the spot of the kick (7 yards behind the line, 80 at most)
            new_yl = max(1, min(80, 93 - st.yardline_100))
            self._change_possession(new_yl)
            self.runoff = 0
        line = (f"{self._header(st)}  {self._where(st.down, st.distance, st.yardline_100)}: "
                f"{text.describe_field_goal(made, distance, self.names[o], self.names[d], new_yl)}")
        return self._event("field_goal", o, st, line, made, seconds, scorer, points)

    def _punt(self, st: GameState) -> Event:
        o, d = self.offense, 1 - self.offense
        result = resolve_punt(st.yardline_100, self.tables, self.rng)
        seconds = self.tables.draw_seconds("punt", st.tempo, score_bucket(st.score_diff), self.rng)
        play_part = min(seconds, PLAY_SECONDS)
        self._tick(play_part, False)
        self.stats[o]["punt"] += 1
        points, scorer = 0, None
        if result.touchdown:
            self._score_touchdown(d, defensive=True)
            points, scorer = 6, d
            self.stats[d]["td"] += 1
        elif result.receiver_has_ball:
            self._change_possession(result.yardline_100)
            self._tick(seconds - play_part, False)
            self.runoff = 0
        else:
            # the kicking team got the ball back (muffed punt)
            self._new_series(result.yardline_100)
            self.clock_running = False
            self.fresh = True
            self.kneeling_team = None
            self.runoff = seconds - play_part
        line = (f"{self._header(st)}  {self._where(st.down, st.distance, st.yardline_100)}: "
                f"{text.describe_punt(result, self.names[o], self.names[d])}")
        return self._event("punt", o, st, line, result, seconds, scorer, points)

    def _play(self, st: GameState) -> Event:
        o, d = self.offense, 1 - self.offense
        edge = self.edges[o] if self.edges else ZERO_EDGE
        out = run_scrimmage_play(st, self.tables, self.rng, edge, self.units[o] if self.units else None)
        flagged = roll_post_snap(st, out, self.tables, self.rng, self._home_offense())
        if flagged:
            return self._penalty(st, flagged, play_kind(out))
        seconds = self.tables.hurry_seconds(st, out.timing_kind, self.rng)
        if seconds is None:
            seconds = self.tables.draw_seconds(out.timing_kind, st.tempo, score_bucket(st.score_diff), self.rng)
        seconds = max(seconds, MIN_PLAY_SECONDS)
        # the play itself is at most 6 seconds, the rest is runoff while the offense gets set
        play_part = min(seconds, PLAY_SECONDS)
        warned = self._tick(play_part, True)
        runoff = seconds - play_part
        self._record_play_stats(o, d, out, st)
        self._record_players(o, out)

        points, scorer = 0, None
        if out.touchdown:
            self._score_touchdown(o)
            points, scorer = 6, o
        elif out.safety:
            self.scores[d] += 2
            points, scorer = 2, d
            self.stats[d]["safety"] += 1
            if self.quarter == 5:
                self._ot_end()
            self._score_then_kick_off(o, free_kick=True)
        elif out.defensive_td:
            self._score_touchdown(d, defensive=True)
            points, scorer = 6, d
        elif out.turnover:
            self._change_possession(out.new_yardline_100)
            if not warned:
                self._tick(runoff, False)
            self.runoff = 0
        else:
            self.yardline_100 = out.new_yardline_100
            self.runoff = runoff
            if out.first_down:
                self._new_series(self.yardline_100)
                self.clock_running = not out.out_of_bounds
            else:
                self.down += 1
                self.distance = max(1, min(self.distance - out.yards, self.yardline_100))
                if self.down > 4:
                    self.stats[o]["turnover_on_downs"] += 1
                    self._change_possession(100 - self.yardline_100)
                    self.runoff = 0
                else:
                    self.clock_running = not out.out_of_bounds
        if warned:
            self.clock_running = False
            self.runoff = 0
        line = (f"{self._header(st)}  {self._where(st.down, st.distance, st.yardline_100)}: "
                f"{text.describe_play(out, self.names[o], self.names[d])}")
        if warned and not self.ot_ended:
            self._queue.append(self._event("two_minute_warning", self.offense, self.state(), "Two minute warning"))
        return self._event("play", o, st, line, out, seconds, scorer, points)

    def _record_players(self, o: int, out: PlayOutcome) -> None:
        def line(name):
            return self.players[o].setdefault(name, Counter())

        def longest(c, key, yards):
            c[key] = max(c[key], yards)

        if out.passer and out.kind in ("complete", "incomplete", "interception"):
            q = line(out.passer)
            q["att"] += 1
            q["comp"] += out.kind == "complete"
            q["pass_yards"] += out.yards if out.kind == "complete" else 0
            q["pass_td"] += out.touchdown
            q["int"] += out.kind == "interception"
            t = line(out.target)
            t["targets"] += 1
            if out.kind == "complete":
                t["rec"] += 1
                t["rec_yards"] += out.yards
                t["rec_td"] += out.touchdown
                longest(t, "long_rec", out.yards)
        elif out.passer and out.kind == "sack":
            q = line(out.passer)
            q["sacked"] += 1
            q["sack_yards"] += -out.yards
        elif out.rusher:
            r = line(out.rusher)
            r["carries"] += 1
            r["rush_yards"] += out.yards
            r["rush_td"] += out.touchdown
            longest(r, "long_rush", out.yards)

    def _record_play_stats(self, o: int, d: int, out: PlayOutcome, st: GameState) -> None:
        s = self.stats[o]
        s["plays"] += 1
        s[out.kind] += 1
        s["yards"] += out.yards
        if out.first_down:
            s["first_downs"] += 1
        if out.touchdown:
            s["td"] += 1
        if out.kind in ("complete", "incomplete", "interception"):
            s["pass_att"] += 1
            if out.kind == "complete":
                s["pass_comp"] += 1
                s["pass_yards"] += out.yards
                if out.touchdown:
                    s["pass_td"] += 1
        elif out.kind == "sack":
            s["sack_yards"] += -out.yards
        else:
            s["rush_att"] += 1
            s["rush_yards"] += out.yards
            if out.touchdown:
                s["rush_td"] += 1
        if out.kind in ("run", "scramble", "complete") and out.yards >= 20:
            s["big_plays"] += 1
        converted = out.first_down or out.touchdown
        if st.down == 3:
            s["third_att"] += 1
            s["third_conv"] += converted
        elif st.down == 4:
            s["fourth_att"] += 1
            s["fourth_conv"] += converted
        if st.yardline_100 <= 20 and not self._rz_marked:
            self._rz_marked = True
            s["rz_trips"] += 1
        if out.touchdown and self._rz_marked:
            s["rz_td"] += 1
        if out.kind == "interception":
            s["interceptions_thrown"] += 1
        if out.fumble_lost:
            s["fumbles_lost"] += 1
        if out.defensive_td:
            self.stats[d]["td"] += 1
