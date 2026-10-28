from dataclasses import dataclass

# every bucket function here has a twin in data_prep/buckets.py, test_state.py checks they match

QUARTER_SECONDS = 15 * 60


def distance_bucket(ydstogo: int) -> str:
    if ydstogo <= 2:
        return "short"
    if ydstogo <= 6:
        return "medium"
    if ydstogo <= 10:
        return "long"
    return "very_long"


def zone_bucket(yardline_100: int) -> str:
    if yardline_100 >= 90:
        return "own_10"
    if yardline_100 >= 80:
        return "backed_up"
    if yardline_100 >= 50:
        return "own_half"
    if yardline_100 > 20:
        return "opp_half"
    if yardline_100 > 5:
        return "red_zone"
    return "goal_line"


def score_bucket(score_diff: int) -> str:
    if score_diff <= -9:
        return "trailing_big"
    if score_diff <= -1:
        return "trailing_close"
    if score_diff == 0:
        return "tied"
    if score_diff <= 8:
        return "leading_close"
    return "leading_big"


def time_bucket(game_seconds_remaining: int) -> str:
    return "late" if game_seconds_remaining <= 300 else "normal"


def fourth_distance_bucket(ydstogo: int) -> str:
    if ydstogo <= 1:
        return "1"
    if ydstogo <= 2:
        return "2"
    if ydstogo <= 4:
        return "3_4"
    if ydstogo <= 7:
        return "5_7"
    if ydstogo <= 10:
        return "8_10"
    return "11_plus"


def fourth_zone_bucket(yardline_100: int) -> str:
    for limit, label in ((5, "z1_5"), (20, "z6_20"), (30, "z21_30"), (36, "z31_36"),
                         (40, "z37_40"), (50, "z41_50"), (65, "z51_65"), (80, "z66_80")):
        if yardline_100 <= limit:
            return label
    return "z81_99"


def tempo_bucket(quarter: int, half_seconds_remaining: int) -> str:
    if quarter in (2, 4, 5):
        if half_seconds_remaining <= 120:
            return "last2"
        if half_seconds_remaining <= 300:
            return "last5"
    return "normal"


def late_time_bucket(quarter: int, game_seconds_remaining: int) -> str:
    if quarter == 4:
        seconds, prefix = game_seconds_remaining, "q4"
    elif quarter == 2:
        seconds, prefix = game_seconds_remaining - 2 * QUARTER_SECONDS, "q2"
    else:
        return "none"
    if 0 < seconds <= 420:
        if seconds <= 120:
            return prefix + "_last2"
        if seconds <= 300:
            return prefix + "_last5"
        return prefix + "_last7"
    return "none"


def late_score_bucket(score_diff: int) -> str:
    if score_diff <= -17:
        return "trail_17p"
    if score_diff <= -9:
        return "trail_9_16"
    if score_diff <= -4:
        return "trail_4_8"
    if score_diff <= -1:
        return "trail_1_3"
    if score_diff == 0:
        return "tied"
    if score_diff <= 8:
        return "lead_1_8"
    return "lead_9p"


def late_zone_bucket(yardline_100: int) -> str:
    if yardline_100 <= 10:
        return "goal"
    if yardline_100 <= 35:
        return "fg"
    if yardline_100 <= 45:
        return "edge"
    if yardline_100 <= 65:
        return "mid"
    return "own"


def late_zone2_bucket(yardline_100: int) -> str:
    return "near" if yardline_100 <= 45 else "far"


def fourth_late_distance_bucket(ydstogo: int) -> str:
    if ydstogo <= 1:
        return "1"
    if ydstogo <= 2:
        return "2"
    if ydstogo <= 6:
        return "3_6"
    return "7p"


def late_quarter_bucket(late_time: str) -> str:
    return late_time[:2]


def late_distance_bucket(ydstogo: int) -> str:
    if ydstogo <= 2:
        return "short"
    if ydstogo <= 6:
        return "medium"
    return "long"


HALF_WINDOW_LIMITS = (3, 6, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240)


def half_window_bucket(seconds_left: int) -> str:
    if seconds_left <= 0:
        return "none"
    for limit in HALF_WINDOW_LIMITS:
        if seconds_left <= limit:
            return f"s{limit}"
    return "none"


def half_zone_bucket(yardline_100: int) -> str:
    if yardline_100 <= 37:
        return "h_fg"
    if yardline_100 <= 55:
        return "h_mid"
    if yardline_100 <= 75:
        return "h_own"
    return "h_deep"


def late_score_class(score_diff: int) -> str:
    if score_diff == 0:
        return "tied"
    if score_diff >= 1:
        return "lead"
    if score_diff >= -3:
        return "t1_3"
    if score_diff >= -8:
        return "t4_8"
    if score_diff >= -16:
        return "t9_16"
    return "t17p"


LATE_WINDOW2_LIMITS = (10, 20, 30, 45, 60, 90, 120, 180, 240)


def late_window2(seconds_left: int) -> str:
    if seconds_left <= 0:
        return "none"
    for limit in LATE_WINDOW2_LIMITS:
        if seconds_left <= limit:
            return f"w{limit}"
    return "none"


@dataclass
class GameState:
    quarter: int = 1
    clock: int = QUARTER_SECONDS
    down: int = 1
    distance: int = 10
    yardline_100: int = 75
    offense_score: int = 0
    defense_score: int = 0
    offense_timeouts: int = 3
    defense_timeouts: int = 3
    clock_running: bool = False
    new_possession: bool = False

    @property
    def score_diff(self) -> int:
        return self.offense_score - self.defense_score

    @property
    def game_seconds_remaining(self) -> int:
        return max(0, 4 - self.quarter) * QUARTER_SECONDS + self.clock

    @property
    def half_seconds_remaining(self) -> int:
        if self.quarter >= 5:
            return self.clock
        if self.quarter <= 2:
            return (2 - self.quarter) * QUARTER_SECONDS + self.clock
        return (4 - self.quarter) * QUARTER_SECONDS + self.clock

    @property
    def tempo(self) -> str:  # normal, last5 or last2 minutes of a half, picks the clock tables

        return tempo_bucket(self.quarter, self.half_seconds_remaining)

    def fourth_down_situation(self) -> dict:
        return {
            "fdist": fourth_distance_bucket(self.distance),
            "fzone": fourth_zone_bucket(self.yardline_100),
            "score_b": score_bucket(self.score_diff),
            "time_b": time_bucket(self.game_seconds_remaining),
        }

    def late_situation(self) -> dict:
        late_time = late_time_bucket(self.quarter, self.game_seconds_remaining)
        return {
            "down": self.down,
            "dist_b": distance_bucket(self.distance),
            "late_score": late_score_bucket(self.score_diff),
            "late_time": late_time,
            "late_q": late_quarter_bucket(late_time),
            "late_zone": late_zone_bucket(self.yardline_100),
            "late_zone2": late_zone2_bucket(self.yardline_100),
            "late_dist": late_distance_bucket(self.distance),
            "late_fdist": fourth_late_distance_bucket(self.distance),
        }

    def half_situation(self) -> dict:
        return {
            "hwin": half_window_bucket(self.half_seconds_remaining) if self.quarter == 2 else "none",
            "rs": "newposs" if self.new_possession else ("running" if self.clock_running else "stopped"),
            "hzone": half_zone_bucket(self.yardline_100),
            "dto": min(self.defense_timeouts, 2),
            "has_to": 1 if self.offense_timeouts > 0 else 0,
            "hdown": min(self.down, 4),
        }

    def late_timeout_situation(self) -> dict:
        in_q4 = self.quarter == 4
        sd = late_score_class(self.score_diff)
        return {
            "lwin": half_window_bucket(self.clock) if in_q4 else "none",
            "lwin2": late_window2(self.clock) if in_q4 else "none",
            "lzone": half_zone_bucket(self.yardline_100),
            "lsd": sd,
            "ldown": min(self.down, 4),
        }

    def situation(self) -> dict:
        return {
            "down": self.down,
            "dist_b": distance_bucket(self.distance),
            "zone_b": zone_bucket(self.yardline_100),
            "score_b": score_bucket(self.score_diff),
            "time_b": time_bucket(self.game_seconds_remaining),
        }
