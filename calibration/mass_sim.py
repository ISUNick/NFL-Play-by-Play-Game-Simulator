import argparse
import glob
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.teams import ALL_TEAMS

SCRIM = {"play", "punt", "field_goal", "kneel", "spike"}
SCORE_BINS = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34), (35, 99)]
DRIVE_KINDS = ["Touchdown", "Field goal", "Punt", "Turnover", "Turnover on downs", "Missed field goal",
               "Safety", "Opp touchdown", "End of half"]


KEYS = (["points", "margin", "tie", "ot", "q1pts", "q2pts", "q3pts", "q4pts", "otpts",
         "plays", "runs", "scrambles", "sacks", "pass_att", "completions", "interceptions",
         "pass_yds", "rush_yds", "fumbles_lost", "third_att", "third_conv", "fourth_att", "fourth_conv", "penalties", "penalty_first",
         "punts", "fg_att", "fg_made", "xp_att", "xp_made", "two_att", "two_made", "tds", "safeties",
         "kneels", "spikes", "timeouts", "onside", "drives", "rz_drives", "rz_td"]
        + [f"drive_{k}" for k in DRIVE_KINDS] + [f"score_{a}_{b}" for a, b in SCORE_BINS])


INDEX = {k: i for i, k in enumerate(KEYS)}

METRICS = [
    ("Total points per game", "points", None, 1, "scoring"),
    ("Margin of victory (average)", "margin", None, 1, "scoring"),
    ("Games that go to overtime", "ot", None, 100, "scoring"),
    ("Games that end in a tie", "tie", None, 100, "scoring"),
    ("Points in Q1 (both teams)", "q1pts", None, 1, "scoring"),
    ("Points in Q2", "q2pts", None, 1, "scoring"),
    ("Points in Q3", "q3pts", None, 1, "scoring"),
    ("Points in Q4", "q4pts", None, 1, "scoring"),
    ("Team scores 0 to 6", "score_0_6", None, 50, "scoring"),
    ("Team scores 7 to 13", "score_7_13", None, 50, "scoring"),
    ("Team scores 14 to 20", "score_14_20", None, 50, "scoring"),
    ("Team scores 21 to 27", "score_21_27", None, 50, "scoring"),
    ("Team scores 28 to 34", "score_28_34", None, 50, "scoring"),
    ("Team scores 35 or more", "score_35_99", None, 50, "scoring"),
    ("Scrimmage plays per game", "plays", None, 1, "offense"),
    ("Designed runs per game", "runs", None, 1, "offense"),
    ("QB scrambles per game", "scrambles", None, 1, "offense"),
    ("Sacks per game", "sacks", None, 1, "offense"),
    ("Pass attempts per game", "pass_att", None, 1, "offense"),
    ("Completion percentage", "completions", "pass_att", 100, "offense"),
    ("Passing yards per attempt (completions only)", "pass_yds", "pass_att", 1, "offense"),
    ("Yards per designed run", "rush_yds", "runs", 1, "offense"),
    ("Sack rate (per dropback)", "sacks", "dropbacks", 100, "offense"),
    ("Interception rate (per attempt)", "interceptions", "pass_att", 100, "offense"),
    ("Dropback share of plays", "dropbacks", "plays", 100, "offense"),
    ("Fumbles lost per game", "fumbles_lost", None, 1, "offense"),
    ("Interceptions per game", "interceptions", None, 1, "offense"),
    ("3rd down attempts per game", "third_att", None, 1, "downs"),
    ("3rd down conversion rate", "third_conv", "third_att", 100, "downs"),
    ("4th down go for it per game", "fourth_att", None, 1, "downs"),
    ("4th down conversion rate", "fourth_conv", "fourth_att", 100, "downs"),
    ("Punts per game", "punts", None, 1, "kicking"),
    ("Field goal attempts per game", "fg_att", None, 1, "kicking"),
    ("Field goal percentage", "fg_made", "fg_att", 100, "kicking"),
    ("Extra point attempts per game", "xp_att", None, 1, "kicking"),
    ("Extra point percentage", "xp_made", "xp_att", 100, "kicking"),
    ("Two point tries per game", "two_att", None, 1, "kicking"),
    ("Two point conversion rate", "two_made", "two_att", 100, "kicking"),
    ("Onside kicks per game", "onside", None, 1, "kicking"),
    ("Touchdowns per game", "tds", None, 1, "events"),
    ("Safeties per game (x1000)", "safeties", None, 1000, "events"),
    ("Kneels per game", "kneels", None, 1, "events"),
    ("Spikes per game", "spikes", None, 1, "events"),
    ("Timeouts called per game", "timeouts", None, 1, "events"),
    ("Penalties that wipe out a snap, per game", "penalties", None, 1, "events"),
    ("Share of those that are an automatic first down", "penalty_first", "penalties", 100, "events"),
    ("Drives per game", "drives", None, 1, "drives"),
    ("Drives ending in a touchdown", "drive_Touchdown", "drives", 100, "drives"),
    ("Drives ending in a field goal", "drive_Field goal", "drives", 100, "drives"),
    ("Drives ending in a punt", "drive_Punt", "drives", 100, "drives"),
    ("Drives ending in a turnover", "drive_Turnover", "drives", 100, "drives"),
    ("Drives ending on downs", "drive_Turnover on downs", "drives", 100, "drives"),
    ("Drives ending in a missed FG", "drive_Missed field goal", "drives", 100, "drives"),
    ("Drives ending by the clock", "drive_End of half", "drives", 100, "drives"),
    ("Red zone drives per game", "rz_drives", None, 1, "drives"),
    ("Red zone touchdown rate", "rz_td", "rz_drives", 100, "drives"),
]

_T = None

_TEAMS = True
_MODEL = None


def _init_worker(use_teams: bool = True) -> None:
    global _T, _TEAMS, _MODEL
    from engine.teams import TeamModel
    _TEAMS = use_teams
    _MODEL = TeamModel() if use_teams else None
    from engine.tables import Tables
    _T = Tables()


def game_counts(g, events) -> list:
    c = dict.fromkeys(KEYS, 0)
    c["points"] = sum(g.scores)
    c["margin"] = abs(g.scores[0] - g.scores[1])
    c["tie"] = int(g.scores[0] == g.scores[1])
    c["ot"] = int(g.went_to_ot)
    qkey = {1: "q1pts", 2: "q2pts", 3: "q3pts", 4: "q4pts", 5: "otpts"}
    for e in events:
        if e.scorer is not None and e.state is not None:
            c[qkey[e.state.quarter]] += e.points
        if e.kind == "timeout":
            c["timeouts"] += 1
        if e.kind == "penalty":
            c["penalties"] += 1
            c["penalty_first"] += int(e.detail.first_down and e.state.down > 1)
        elif e.kind == "play" and e.detail is not None:
            o, st = e.detail, e.state
            if o.kind == "complete":
                c["pass_yds"] += o.yards
            elif o.kind == "run":
                c["rush_yds"] += o.yards
            if st.down == 3:
                c["third_att"] += 1
                c["third_conv"] += int(o.first_down or o.touchdown)
            elif st.down == 4:
                c["fourth_att"] += 1
                c["fourth_conv"] += int(o.first_down or o.touchdown)
    for s in g.stats:
        c["plays"] += s["plays"]
        c["runs"] += s["run"]
        c["scrambles"] += s["scramble"]
        c["sacks"] += s["sack"]
        c["completions"] += s["complete"]
        c["interceptions"] += s["interception"]
        c["pass_att"] += s["complete"] + s["incomplete"] + s["interception"]
        c["fumbles_lost"] += s["fumbles_lost"]
        c["punts"] += s["punt"]
        c["fg_att"] += s["fg_att"]
        c["fg_made"] += s["fg_made"]
        c["xp_att"] += s["xp_att"]
        c["xp_made"] += s["xp_made"]
        c["two_att"] += s["two_att"]
        c["two_made"] += s["two_made"]
        c["tds"] += s["td"]
        c["safeties"] += s["safety"]
        c["kneels"] += s["kneel"]
        c["spikes"] += s["spike"]
        c["onside"] += s["onside_att"]
    for team in (0, 1):
        for a, b in SCORE_BINS:
            if a <= g.scores[team] <= b:
                c[f"score_{a}_{b}"] += 1
    _drives(events, c)
    return [c[k] for k in KEYS]


def _drives(events, c) -> None:
    groups, cur, cur_team = [], [], None
    for e in events:
        if e.kind in ("kickoff", "onside", "halftime", "end_regulation"):
            if cur:
                groups.append(cur)
            cur, cur_team = [], None
        elif e.kind in SCRIM:
            if cur_team is not None and e.team != cur_team:
                groups.append(cur)
                cur = []
            cur.append(e)
            cur_team = e.team
    if cur:
        groups.append(cur)
    for grp in groups:
        last = grp[-1]
        c["drives"] += 1
        d = last.detail
        if last.kind == "punt":
            res = "Punt"
        elif last.kind == "field_goal":
            res = "Field goal" if d else "Missed field goal"
        elif last.kind == "play":
            if last.scorer == last.team and last.points == 6:
                res = "Touchdown"
            elif d.defensive_td:
                res = "Opp touchdown"
            elif d.turnover:
                res = "Turnover"
            elif d.safety:
                res = "Safety"
            elif last.state.down == 4 and not d.first_down:
                res = "Turnover on downs"
            else:
                res = "End of half"
        else:
            res = "End of half"
        c[f"drive_{res}"] += 1
        if any(x.state.yardline_100 <= 20 for x in grp if x.state is not None):
            c["rz_drives"] += 1
            c["rz_td"] += int(res == "Touchdown")


def _run_chunk(args):
    start, n = args
    from engine.game import Game
    rows = []
    for seed in range(start, start + n):
        if _TEAMS:
            i, j = random.Random(seed ^ 0x5EED).sample(range(32), 2)
            g = Game(_T, random.Random(seed), edges=_MODEL.edge_pair(ALL_TEAMS[i], ALL_TEAMS[j]))
        else:
            g = Game(_T, random.Random(seed))
        events = []
        while not g.over or g._queue:
            events.append(g.step())
        rows.append(game_counts(g, events))
    return np.array(rows, dtype=np.int32)


def simulate(n_games: int, workers: int, base_seed: int, use_teams: bool = True) -> pd.DataFrame:
    chunk = 500
    jobs = [(base_seed + i, min(chunk, n_games - i)) for i in range(0, n_games, chunk)]
    t0 = time.time()
    parts = []
    with mp.Pool(workers, initializer=_init_worker, initargs=(use_teams,)) as pool:
        for k, part in enumerate(pool.imap(_run_chunk, jobs), 1):
            parts.append(part)
            done = sum(len(p) for p in parts)
            if k % max(1, len(jobs) // 10) == 0 or done == n_games:
                rate = done / (time.time() - t0)
                print(f"  simulated {done:,} of {n_games:,} games ({rate:,.0f} games per second)", flush=True)
    return pd.DataFrame(np.vstack(parts), columns=KEYS)


def real_counts(from_season: int) -> pd.DataFrame:
    cols = ["game_id", "season", "season_type", "qtr", "drive", "play_id", "posteam", "play_type", "down", "ydstogo",
            "yards_gained", "pass_attempt", "complete_pass", "sack", "qb_scramble", "interception", "fumble_lost",
            "touchdown", "safety", "timeout", "third_down_converted", "third_down_failed", "fourth_down_converted",
            "fourth_down_failed", "field_goal_result", "extra_point_result", "two_point_attempt",
            "two_point_conv_result", "kick_distance", "total_home_score", "total_away_score", "fixed_drive_result",
            "drive_inside20", "season", "penalty", "yardline_100"]
    cols = list(dict.fromkeys(cols))
    files = [f for f in sorted(glob.glob(str(ROOT / "data/raw/pbp_*.parquet")))
             if int(Path(f).stem.split("_")[1]) >= from_season]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    d = d[d.season_type == "REG"].sort_values(["game_id", "play_id"])
    f = lambda col: d[col].fillna(0) == 1
    pr = d.play_type.isin(["pass", "run"])
    scr = (d.qb_scramble.fillna(0) == 1) & pr
    run = (d.play_type == "run") & ~scr
    att = f("pass_attempt") & (d.play_type == "pass") & ~f("sack")
    comp = f("complete_pass") & (d.play_type == "pass")

    g = pd.DataFrame(index=d.game_id.unique())
    g.index.name = "game_id"

    def count(mask, name):
        g[name] = mask.groupby(d.game_id).sum().reindex(g.index).fillna(0)

    def total(vals, name):
        g[name] = vals.groupby(d.game_id).sum().reindex(g.index).fillna(0)

    count(pr, "plays"); count(run, "runs"); count(scr, "scrambles")
    count(f("sack") & (d.play_type == "pass"), "sacks"); count(att, "pass_att"); count(comp, "completions")
    count(f("interception") & (d.play_type == "pass"), "interceptions")
    total(d.yards_gained.where(comp, 0), "pass_yds"); total(d.yards_gained.where(run, 0), "rush_yds")
    count(f("fumble_lost") & pr, "fumbles_lost")
    count(f("third_down_converted") | f("third_down_failed"), "third_att"); count(f("third_down_converted"), "third_conv")
    count(f("fourth_down_converted") | f("fourth_down_failed"), "fourth_att"); count(f("fourth_down_converted"), "fourth_conv")
    count(d.play_type == "punt", "punts"); count(d.play_type == "field_goal", "fg_att")
    count((d.play_type == "field_goal") & (d.field_goal_result == "made"), "fg_made")
    count(d.play_type == "extra_point", "xp_att"); count((d.play_type == "extra_point") & (d.extra_point_result == "good"), "xp_made")
    count(f("two_point_attempt"), "two_att"); count(f("two_point_attempt") & (d.two_point_conv_result == "success"), "two_made")
    nxt = d.groupby("game_id")[["down", "posteam"]].shift(-1).rename(columns={"down": "n_down", "posteam": "n_pos"})
    pen = ((d.play_type == "no_play") & f("penalty") & d.down.isin([1, 2, 3, 4]) & ~f("timeout") & (nxt.n_pos == d.posteam)
           & ((nxt.n_down == d.down) | ((nxt.n_down == 1) & (d.down > 1))))
    count(pen, "penalties"); count(pen & (nxt.n_down == 1) & (d.down > 1), "penalty_first")
    count(f("touchdown"), "tds"); count(f("safety"), "safeties")
    count(d.play_type == "qb_kneel", "kneels"); count(d.play_type == "qb_spike", "spikes")
    count(f("timeout") & (d.play_type == "no_play"), "timeouts")
    count((d.play_type == "kickoff") & (d.kick_distance <= 25) & (d.season >= 2018), "onside")

    d["total"] = d.total_home_score + d.total_away_score
    last = d.groupby("game_id").tail(1).set_index("game_id")
    g["points"] = last.total.reindex(g.index)
    g["margin"] = (last.total_home_score - last.total_away_score).abs().reindex(g.index)
    g["tie"] = (last.total_home_score == last.total_away_score).astype(int).reindex(g.index)
    g["ot"] = (d[d.qtr == 5].groupby("game_id").size().reindex(g.index).fillna(0) > 0).astype(int)
    end_of = {q: d[d.qtr == q].groupby("game_id").total.max().reindex(g.index) for q in (1, 2, 3, 4)}
    prev = 0
    for q, key in ((1, "q1pts"), (2, "q2pts"), (3, "q3pts"), (4, "q4pts")):
        g[key] = (end_of[q].fillna(0) - prev)
        prev = end_of[q].fillna(0)
    g["otpts"] = g["points"] - prev
    for team in ("home", "away"):
        s = last[f"total_{team}_score"].reindex(g.index)
        for a, b in SCORE_BINS:
            col = f"score_{a}_{b}"
            g[col] = g.get(col, 0) + s.between(a, b).astype(int)

    dr = d[d.posteam.notna() & d.fixed_drive_result.notna() & d.play_type.isin(
        ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"])]
    one = dr.groupby(["game_id", "drive"]).agg(res=("fixed_drive_result", "first"), inside=("drive_inside20", "max"))
    one = one.reset_index()
    g["drives"] = one.groupby("game_id").size().reindex(g.index).fillna(0)
    for k in DRIVE_KINDS:
        g[f"drive_{k}"] = one[one.res == k].groupby("game_id").size().reindex(g.index).fillna(0)
    rz = one[one.inside == 1]
    g["rz_drives"] = rz.groupby("game_id").size().reindex(g.index).fillna(0)
    g["rz_td"] = rz[rz.res == "Touchdown"].groupby("game_id").size().reindex(g.index).fillna(0)
    return g[KEYS].astype(float).reset_index(drop=True)


def estimate(df: pd.DataFrame, num: str, den, scale: float):
    n = len(df)
    dropbacks = df["pass_att"] + df["sacks"]
    cols = {"dropbacks": dropbacks}
    x = df[num] if num in df else cols[num]
    if den is None:
        return scale * x.mean(), scale * x.std(ddof=1) / math.sqrt(n)
    y = df[den] if den in df else cols[den]
    r = x.sum() / y.sum()
    resid = x - r * y
    return scale * r, scale * resid.std(ddof=1) / math.sqrt(n) / y.mean()


def compare(sim: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, num, den, scale, group in METRICS:
        rv, rse = estimate(real, num, den, scale)
        sv, sse = estimate(sim, num, den, scale)
        diff = sv - rv
        z = diff / math.sqrt(rse ** 2 + sse ** 2) if (rse or sse) else 0.0
        rel = abs(diff) / abs(rv) if rv else 0.0
        verdict = "OK" if abs(z) < 2.5 else ("close" if rel < 0.05 else "OFF")
        rows.append((group, label, rv, sv, diff, z, verdict))
    return pd.DataFrame(rows, columns=["group", "stat", "real", "sim", "diff", "z", "verdict"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--seed", type=int, default=1_000_000, help="first game seed")
    ap.add_argument("--from-season", type=int, default=2018, help="first real season to compare with")
    ap.add_argument("--even", action="store_true", help="no team ratings: every game is two league average teams")
    ap.add_argument("--save", action="store_true", help="write the table and the per game counts to data/")
    args = ap.parse_args()

    print(f"Loading real games (regular season {args.from_season} and later)...")
    real = real_counts(args.from_season)
    print(f"  {len(real):,} real games\n\nSimulating {args.games:,} games on {args.workers} cores...")
    sim = simulate(args.games, args.workers, args.seed, not args.even)

    table = compare(sim, real)
    pd.set_option("display.width", 200)
    print(f"\n{'':>3}{'statistic':<38}{'real':>10}{'sim':>10}{'diff':>9}{'z':>7}   verdict")
    group = None
    for r in table.itertuples():
        if r.group != group:
            group = r.group
            print(f"\n[{group}]")
        print(f"   {r.stat:<38}{r.real:>10.2f}{r.sim:>10.2f}{r.diff:>+9.2f}{r.z:>7.1f}   {r.verdict}")
    counts = table.verdict.value_counts()
    print(f"\n{len(table)} statistics: {counts.get('OK', 0)} OK, {counts.get('close', 0)} close, {counts.get('OFF', 0)} OFF")

    print("\nSPREAD (the sim has no good and bad teams, so a somewhat narrower spread is expected)")
    for label, col in (("Total points per game", "points"), ("Margin of victory", "margin")):
        print(f"   {label:<30} real SD {real[col].std():6.2f}   sim SD {sim[col].std():6.2f}")
    for label, test in (("Games decided by 8 or less", lambda m: (m <= 8)), ("Games decided by 21 or more", lambda m: (m >= 21)),
                        ("One possession (1 to 8)", lambda m: (m >= 1) & (m <= 8)), ("Decided by 3 or less", lambda m: (m <= 3))):
        print(f"   {label:<30} real {test(real.margin).mean():6.1%}   sim {test(sim.margin).mean():6.1%}")

    if args.save:
        out = ROOT / "data" / "mass_sim"
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "comparison.csv", index=False)
        sim.to_csv(out / "sim_games.csv", index=False)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
