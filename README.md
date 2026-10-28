# Football Sim v2

A play by play NFL game simulator built on real nflverse data. Version 2 adds real players and a live Monte Carlo view on top of the version 1 engine. 

## Folders

- `engine/` the play by play engine: clock rules, penalties, fourth down choices, special teams, overtime, team ratings, schedule, standings, playoffs
- `data_prep/` downloads nflverse play by play and builds the tables the engine samples from
- `data/tables/` the built tables (the engine reads these, `data/raw/` is not in the repo)
- `calibration/` compares the simulator with real games (`mass_sim.py` is the main one)
- `players/` player ratings math (`shrink.py`) and lineups built from rosters and injuries (`lineup.py`)
- `app/` the browser dashboard (`server.py`, `runner.py`, `static/index.html`)
- `tests/` one test file per piece, run them all with `python tests/run_all.py`

## Run it

```
pip install -r requirements.txt
python tests/run_all.py
python play_game.py --seed 42
python app/server.py
```

Then open http://127.0.0.1:8000 in a browser. It has four pages: a live million game run, a matchup picker where you mark players out, a live game with a win probability bar, and a validation page.

## What is not here on purpose

The version 1 franchise mode (cards, cap, draft), the coach mode and the browser franchise screens stay in version 1.

## Known limits

Public play by play has passer, receiver and rusher ids, but no blocking or coverage assignments. Offensive linemen and most defenders can only be rated as units.

## Player ratings

Run `python data_prep/download_data.py` then `python data_prep/build_players.py` to rebuild them from the latest nflverse data (about 3 seconds). `python data_prep/validate_players.py` rechecks them on held out weeks.

Team and receiver ratings come from this season plus a slight pull from last season (half life 8 weeks). Quarterbacks are rated on a longer look, three seasons with a 40 week half life, because their skill lasts for years while a team's line and receivers change fast. In the 2023 to 2025 tests this beat the single short look on Brier and margin error.

Quarterbacks with little data start from their Madden rating instead of the average (`data_prep/madden.py`, `data/external/madden_qb.csv`). Madden ratings have no history here, so this piece cannot be backtested. A bigger quarterback only weight (2.5 times) was tested and was worse in all three seasons, so it is not used (`--qb-scale` in the backtest). Every rating is pulled toward the position average, harder when a player has few plays, and the strength of that pull is picked on games the fit never saw. Quarterback and receiver completion effects are fit together so a quarterback does not get credit for a great receiver.

Limit: public data has no blocking or coverage assignments, so offensive linemen and defenders are rated as team units, not as individuals.

## Accuracy against real games

`python calibration/backtest_all.py` is the honest test. For each season it builds the tables from the four seasons before it, matches the pace and scoring dials to the season right before it (`calibration/era_fit.py`), then simulates every game from week 5 on, 200 times each, using ratings built only from games played before it and the lineup and injury report of that week. It is compared with the closing Vegas line. Team and player effects are stretched by 1.5. Ratings use the season before and the current one with a half life of 8 weeks (an 8 week half life beat 22 weeks on margin error in all three test seasons).

| Season | Picks the winner (sim / Vegas) | Brier score (sim / Vegas) | Correlation with real margin (sim / Vegas) |
|---|---|---|---|
| 2022 | 63.1% / 68.9% | 0.228 / 0.205 | 0.24 / 0.38 |
| 2023 | 64.4% / 68.8% | 0.232 / 0.218 | 0.29 / 0.43 |
| 2024 | 70.2% / 75.5% | 0.202 / 0.188 | 0.50 / 0.59 |
| 2025 | 58.7% / 63.0% | 0.242 / 0.217 | 0.32 / 0.51 |

The simulator is close to the market but does not beat it in any season. A walk forward linear fit of the sim margin (`calibration/platt.py`, tuned on earlier seasons only) did not help, so it is not used. The market also knows about things this model cannot see, such as offensive line and defender injuries, coaching changes and weather.

`python calibration/predict.py` plays out every unplayed game of the current season from today's rosters and injuries and writes `data/predictions/`.

## Live win probability

`engine/winprob.py` has two ways to get the win chance at any moment. `rollout` plays the rest of that exact game many times (exact but slow). `WinProb` is a fast model trained on 18,000 simulated games (`calibration/train_winprob.py`). On held out games it is well calibrated, and `tests/test_winprob.py` checks it against rollouts.
