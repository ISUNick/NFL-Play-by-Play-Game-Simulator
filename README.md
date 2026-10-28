# NFL Play by Play Game Simulator

An NFL game simulator that plays every snap using real play by play data and real player ratings. Pick two teams, mark who's injured, and run a million games in the browser, or watch one game live with a win probability chart.

## Features

- **Million game run:** win probability with a 95% range, average box score, and a plot of every final score. You can replay the most typical game of the run.
- **Live game:** plays out snap by snap with a win probability bar. Big swings are marked on the chart, and you can hover them to see the play.
- **Injuries:** the current injury report is loaded automatically. Check or uncheck any player to see how much he matters.
- **Backtesting:** every game is predicted using only data from before it, and the results are compared to the Vegas closing line.

## How it works

The engine samples each play from real NFL plays in the same situation (down, distance, field position, score, time left). That covers everything from the clock and timeouts to fourth down decisions, penalties, special teams and overtime.

Player ratings come from ridge regressions on every pass and run. QB, receiver, offense and defense are fit together, so a QB doesn't get credit for his receivers. Ratings are shrunk toward average when a player has few snaps. QBs are rated over three seasons and everything else mostly on the current one. Backup QBs with little data start from their Madden rating.

## Results

Walk forward backtest, weeks 5 to 18, 200 sims per game:

| Season | Picks winner (sim / Vegas) | Brier (sim / Vegas) |
|---|---|---|
| 2021 | 64.3% / 63.8% | .222 / .216 |
| 2022 | 62.6% / 68.9% | .228 / .205 |
| 2023 | 64.4% / 68.8% | .229 / .218 |
| 2024 | 73.6% / 75.5% | .198 / .188 |
| 2025 | 60.1% / 63.0% | .237 / .217 |

It's close to Vegas but doesn't beat it. Vegas knows things this model can't see, like offensive line injuries and weather.

## Running it

```
pip install -r requirements.txt
python tests/run_all.py
python app/server.py
```

Then open http://127.0.0.1:8000.

To pull the latest data and update ratings and predictions:

```
python data_prep/download_data.py 2023 2026
python update.py
```

## Limits

Public play by play doesn't say who blocked or who covered, so offensive linemen and defenders are rated as team units, not individually.

## Data

[nflverse](https://github.com/nflverse/nflverse-data) for play by play, rosters, snaps and injuries. Vegas lines from [nfldata](https://github.com/nflverse/nfldata).