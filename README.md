# Football Sim

A play by play NFL simulator built from real data. Every snap is drawn from what actually happened in thousands of real NFL plays in the same situation, then shifted by the real players on the field: the quarterback, who he throws to, who carries the ball, and how good each team's line and defense are. It runs a million games of one matchup in the browser, plays single games live with a win probability chart, and is tested against real results and the Vegas closing line.

![Million game run](docs/dashboard_run.jpg)

![Live game with win probability](docs/dashboard_live.jpg)

## What it does

- **Million game run.** Pick two teams. Players on the current injury report start checked out, and you can put them back in or take anyone else out. The page shows the win probability as games finish with a 95% range, the average box score, and a dot for every final score. One button replays the most typical game of the run play by play.
- **Live game.** One game plays out snap by snap with a win probability bar. Big swings get a dot you can hover to see the play that caused it, and an ESPN style box score appears at the end.
- **Matchup picker.** Run the same game with different players out to see what one player is worth.
- **Validation.** The backtest against real games, calibration, and simulated versus real team stats.
- **Season predictions.** `calibration/predict.py` plays out the rest of the current season from today's rosters and injuries.

Every game has its own seed, so any run repeats exactly no matter how many CPU cores it uses.

## Accuracy against real games

The test is walk forward, so the model never sees the future. For each season, the play tables come only from the four seasons before it, and the league scoring level is matched to the season right before. Every game from week 5 on is simulated 200 times, using ratings built only from games already played and that week's real lineups and injury report. The results are compared with the closing Vegas line.

| Season | Games | Picks the winner (sim / Vegas) | Brier score (sim / Vegas) | Average miss on the margin, points (sim / Vegas) |
|---|---|---|---|---|
| 2021 | 208 | 64.3% / 63.8% | 0.222 / 0.216 | 11.3 / 11.0 |
| 2022 | 207 | 62.6% / 68.9% | 0.228 / 0.205 | 9.3 / 8.9 |
| 2023 | 208 | 64.4% / 68.8% | 0.229 / 0.218 | 10.1 / 9.5 |
| 2024 | 208 | 73.6% / 75.5% | 0.198 / 0.188 | 9.7 / 9.3 |
| 2025 | 208 | 60.1% / 63.0% | 0.237 / 0.217 | 10.6 / 10.0 |
| **All** | 1039 | 65.0% / 68.0% | 0.223 / 0.209 | 10.2 / 9.7 |

Brier score is the squared error of the win probability, so lower is better and 0.25 is a coin flip. The simulator gets close to the market. In 2021 it picked slightly more winners than Vegas, but Vegas had the better Brier score and the smaller margin error in every season, so it does not beat the market. Vegas also sees things this model cannot, such as offensive line and defender injuries, coaching changes and weather.

![Validation page](docs/validation.jpg)

The game by game results are in `data/backtest/`, and the Validation page of the dashboard reads them. Rerun them with `python calibration/backtest_all.py --seasons 2021-2025` (needs the raw data from 2017 on, see below).

## How it works

**The engine** (`engine/`) runs a full game: kickoffs, punts, field goals with real make rates by distance, extra points and two point tries, fourth down choices, penalties with rulebook yardage, the play clock, timeouts, the two minute drill, onside kicks and overtime. Each decision and each result is sampled from tables built from real nflverse play by play, grouped by down, distance, field position, score and time left. When a group is too small, the lookup falls back to a wider group.

**Player ratings** (`data_prep/build_players.py`) come from joint ridge regressions on every real pass and run. A completed pass is credited to the quarterback, the receiver, the offense and the defense at the same time, so a quarterback does not get credit for a great receiver. Every rating is pulled toward the position average, harder when a player has few plays, and the strength of that pull is picked on games the fit never saw.

- Team ratings use this season plus a slight pull from last season (games count half as much every 8 weeks).
- Receivers and runners get the longer three season look too, so a back's yards per carry and a receiver's catch rate and yards after the catch keep their own spread instead of all sliding to the average. In the backtest this was a wash on accuracy (about 0.221 Brier either way), so it stays for the realism.
- Linemen and defenders are rated as team units: offensive line, pass rush, run defense and pass coverage. Each side of a matchup has a dropdown for all four, so you can plug in another team's unit. A strong line against a strong run defense cancel out, and a weak pass rush gives the quarterback time.
- Quarterbacks are rated over three seasons with a 40 week half life, because quarterback skill lasts for years while a team's line and receivers change quickly. This beat a single short look in the backtest.
- Quarterbacks with little data start from their Madden rating instead of the league average (`data_prep/madden.py`), so a backup who has barely played is not treated as average.

**Lineups** (`players/lineup.py`) come from real snap counts and the weekly injury report. When a player is out, his snaps go to the others at his position, and targets and carries follow the snaps.

**Win probability** (`engine/winprob.py`) is a logistic model trained on 18,000 simulated games and checked against exact rollouts that replay the rest of the game many times.

## Quick start

Needs Python 3.10 or newer (developed on 3.13).

```
pip install -r requirements.txt
python tests/run_all.py
python app/server.py
```

Then open http://127.0.0.1:8000. On start it checks for new games and rebuilds the ratings and injuries when the saved ones are old, so it needs internet the first time each day. `python app/server.py --no-update` skips that and uses the ratings already in the repo. `python play_game.py --seed 42` plays one game in the terminal one snap at a time (press Enter), and `--auto` plays it straight through.

## Rebuilding from fresh data

```
python data_prep/download_data.py 2023 2026
python data_prep/build_players.py
python calibration/predict.py
```

The first line downloads nflverse play by play, rosters, snap counts and injury reports into `data/raw/` (not in the repo). The next two rebuild the player ratings and the season predictions. `python update.py` does all three for just the current season, and the server runs it on its own at start.

The play tables in `data/tables/` change much less often and are built by `python data_prep/build_all.py`. By default it uses the last four complete seasons, and `SIM_TABLE_SEASONS=2014-2025` picks the years. More seasons fill the rare late game situations better. The backtest builds its own tables per season, so it needs the data from 2017 on (`python data_prep/download_data.py 2017 2026`).

## Project layout

```
engine/       game engine: rules, clock, play results, special teams, penalties, overtime, win probability
players/      lineups from rosters, snaps and injuries, and the rating shrinkage math
data_prep/    downloads nflverse data and builds every table and rating the engine reads
calibration/  backtests against real games and Vegas, scoring level fits, season predictions
app/          the browser dashboard (standard library HTTP server and one HTML page)
tests/        one test file per piece, run them all with python tests/run_all.py
data/         built tables, ratings, predictions and backtest results
docs/         screenshots
```

## Known limits

- Public play by play names the passer, the receiver and the rusher, but not who blocked or who covered. Offensive linemen and defenders are rated as team units, so losing a star left tackle or edge rusher does not change the sim.
- Madden ratings only exist for the current season here, so that piece cannot be backtested.
- A backup quarterback still costs a team less in the sim than the betting market says it should.

## Data

Play by play, rosters, snap counts and injury reports are from [nflverse](https://github.com/nflverse/nflverse-data). Schedules and closing lines are from [nfldata](https://github.com/nflverse/nfldata). Quarterback ratings in `data/external/madden_qb.csv` are EA Sports Madden NFL ratings, used only as a starting point for players with little data.

## License

MIT
