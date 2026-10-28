import json
import math
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "tables" / "winprob.json"
QUARTER_SECONDS = 900
FEATURES = ["lead_per_sqrt", "lead", "ball_per_sqrt", "field_per_sqrt", "field", "lead_late", "ball_late",
            "timeouts_per_sqrt", "overtime_ball", "overtime_lead", "first_down_late"]


def seconds_left(game) -> float:
    if game.quarter >= 5:
        return float(game.clock)
    return (4 - game.quarter) * QUARTER_SECONDS + float(game.clock)


def snapshot(game) -> dict:
    # everything the bar needs, seen from the home team (team 0)
    return {"lead": game.scores[0] - game.scores[1], "left": seconds_left(game), "ot": game.quarter >= 5,
            "home_ball": 1.0 if game.offense == 0 else -1.0, "yardline_100": game.yardline_100,
            "down": game.down, "timeouts": game.timeouts[0] - game.timeouts[1]}


def features(snap: dict) -> np.ndarray:
    left, lead, ball = snap["left"], snap["lead"], snap["home_ball"]
    root = math.sqrt(left / 60.0 + 1.0)
    field = ball * (100 - snap["yardline_100"]) / 100.0
    late = 1.0 if left <= 180 and not snap["ot"] else 0.0
    ot = 1.0 if snap["ot"] else 0.0
    return np.array([lead / root, lead, ball / root, field / root, field, lead * late, ball * late,
                     snap["timeouts"] / root, ot * ball, ot * lead, late * ball * (1.0 if snap["down"] == 1 else 0.0)])


def fit(rows: np.ndarray, won: np.ndarray, ridge: float = 1e-3, iters: int = 25) -> np.ndarray:
    # plain logistic regression by Newton steps. The first column is the intercept
    x = np.column_stack([np.ones(len(rows)), rows])
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-x @ beta))
        grad = x.T @ (won - p) - ridge * beta
        hess = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(x.shape[1])
        beta += np.linalg.solve(hess, grad)
    return beta


class WinProb:
    def __init__(self, path=MODEL_PATH):
        data = json.loads(Path(path).read_text())
        self.beta = np.array(data["beta"])
        self.prior_weight = data["prior_weight"]

    def logit(self, snap: dict) -> float:
        return float(self.beta[0] + features(snap) @ self.beta[1:])

    def home(self, game, pregame: float = 0.5) -> float:
        # pregame is the home team's chance before kickoff. Its pull fades as the clock runs down
        snap = snapshot(game)
        z = self.logit(snap)
        edge = math.log(pregame / (1 - pregame))
        z += self.prior_weight * edge * math.sqrt(max(snap["left"], 0.0) / 3600.0)
        return 1.0 / (1.0 + math.exp(-z))


def rollout(game, n: int, seed) -> float:
    # play the rest of this exact game n times and count how often the home team wins (a tie counts half)
    wins = 0.0
    for i in range(n):
        twin = game.fork(f"{seed}|{i}")
        while not (twin.over and not twin._queue):
            twin.step()
        a, b = twin.scores
        wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / n


def track(game, model: WinProb, pregame: float = 0.5, rollouts: int = 0, seed="track") -> list:
    # play the game out and record the home win chance after every event: [(text, scores, left, prob)]
    out = [("Kickoff", tuple(game.scores), seconds_left(game), model.home(game, pregame) if not rollouts
            else rollout(game, rollouts, f"{seed}|0"))]
    i = 0
    while not (game.over and not game._queue):
        ev = game.step()
        i += 1
        if game.over and not game._queue:
            p = 1.0 if game.scores[0] > game.scores[1] else 0.5 if game.scores[0] == game.scores[1] else 0.0
        elif rollouts:
            p = rollout(game, rollouts, f"{seed}|{i}")
        else:
            p = model.home(game, pregame)
        out.append((ev.text, tuple(game.scores), seconds_left(game), p))
    return out
