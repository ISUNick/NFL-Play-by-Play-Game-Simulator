import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import madden
from players import shrink as sh

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "players"

SEASONS_BACK = int(os.environ.get("SIM_SEASONS_BACK", 1))
HALF_LIFE_WEEKS = float(os.environ.get("SIM_HALF_LIFE", 8))
LONG_SKILL = os.environ.get("SIM_LONG_SKILL", "1") == "1"  # rate runners and receivers on the long look too
QB_SEASONS_BACK = int(os.environ.get("SIM_QB_SEASONS_BACK", 3))
QB_HALF_LIFE_WEEKS = float(os.environ.get("SIM_QB_HALF_LIFE", 40))
SHORT_MAX_AIR_YARDS = 9
MEDIUM_MAX_AIR_YARDS = 19
TEAM_FIX = {"LA": "LAR"}
DEPTHS = ["short", "medium", "deep"]
YARDS_GRID = [0.03, 0.08, 0.15, 0.3, 0.5, 0.8, 1.2]
TAU_GRID = [0.02, 0.05, 0.09, 0.14, 0.2, 0.3, 0.45, 0.7, 1.0]
PBP_COLS = ["season", "week", "game_id", "play_type", "qb_dropback", "qb_scramble", "sack", "interception",
            "complete_pass", "rush_attempt", "penalty", "two_point_attempt", "cp", "air_yards", "yards_after_catch",
            "yards_gained", "ydstogo", "passer_player_id", "receiver_player_id", "rusher_player_id", "posteam", "defteam"]


def latest_season() -> int:
    return max(int(p.stem.split("_")[1]) for p in RAW_DIR.glob("pbp_*.parquet"))


def load_plays(seasons: list, before=None, half_life: float = None) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(RAW_DIR / f"pbp_{s}.parquet", columns=PBP_COLS) for s in seasons], ignore_index=True)
    df = df[df.play_type.isin(["pass", "run"]) & (df.penalty.fillna(0) != 1) & (df.two_point_attempt.fillna(0) != 1)].copy()
    df["posteam"] = df.posteam.replace(TEAM_FIX)
    df["defteam"] = df.defteam.replace(TEAM_FIX)
    df["time"] = df.season * 22 + df.week
    if before is not None:
        df = df[df.time < before[0] * 22 + before[1]].copy()
    df["w"] = 0.5 ** ((df.time.max() - df.time) / (half_life or HALF_LIFE_WEEKS))
    df["dropback"] = df.qb_dropback == 1
    df["is_sack"] = df.sack == 1
    df["is_scramble"] = (df.qb_scramble == 1) & ~df.is_sack
    df["is_run"] = (df.rush_attempt == 1) & ~df.dropback & (df.qb_scramble != 1)
    df["is_throw"] = df.dropback & ~df.is_sack & ~df.is_scramble
    # on a scramble the quarterback is listed as the rusher
    df["qb_id"] = np.where(df.is_scramble, df.rusher_player_id, df.passer_player_id)
    df["depth"] = np.select([df.air_yards <= SHORT_MAX_AIR_YARDS, df.air_yards <= MEDIUM_MAX_AIR_YARDS],
                            ["short", "medium"], default="deep")
    return df


def latest_roster() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(RAW_DIR.glob("roster_week_*.parquet"))]
    r = pd.concat(frames, ignore_index=True)
    r["team"] = r.team.replace(TEAM_FIX)
    r = r.dropna(subset=["gsis_id"]).sort_values(["season", "week"])
    return r


def codes_of(ids: pd.Series):
    codes, uniques = pd.factorize(ids)
    return codes, list(uniques)


def weighted_mean_and_n(g) -> pd.Series:
    # n is the number of equally weighted plays that would give the same precision as the recency weighted ones
    return pd.Series({"n": g.w.sum() ** 2 / (g.w ** 2).sum(), "mean": np.average(g.dev, weights=g.w)})


def calibrate_offset(base, y, w) -> float:
    # one constant so the average of the base probabilities matches the real rate, shifts are then relative to average
    a = 0.0
    for _ in range(30):
        p = sh.expit(base + a)
        a += (np.average(y, weights=w) - np.average(p, weights=w)) / np.average(p * (1 - p), weights=w)
    return float(a)


def fit_players(y, base, id_series: dict, w, game_ids, grid):
    # id_series is {role: ids of that role for every row}. Penalty strengths are chosen on held out games
    groups, ids = {}, {}
    for role, series in id_series.items():
        codes, uniques = codes_of(series)
        groups[role], ids[role] = (codes, len(uniques)), uniques
    y, base, w = np.asarray(y, float), np.asarray(base, float), np.asarray(w, float)
    fold = pd.factorize(game_ids)[0] % 2
    tau = sh.choose_tau(y, base, groups, w, fold, {r: grid for r in groups})
    effects = sh.fit_logistic_effects(y, base, groups, {r: t * t for r, t in tau.items()}, w)
    return {r: pd.Series(effects[r], index=ids[r]) for r in groups}, tau


def fit_linear_players(y, roles: dict, w, game_ids, sigma2):
    groups, ids = {}, {}
    for role, series in roles.items():
        codes, uniques = codes_of(series)
        groups[role], ids[role] = (codes, len(uniques)), uniques
    y, w = np.asarray(y, float), np.asarray(w, float)
    fold = pd.factorize(game_ids)[0] % 2
    tau = sh.choose_tau_linear(y, groups, w, fold, {r: YARDS_GRID for r in groups}, sigma2)
    effects = sh.fit_linear_effects(y, groups, {r: t * t for r, t in tau.items()}, sigma2, w)
    return {r: pd.Series(effects[r], index=ids[r]) for r in groups}, tau


def weighted_var(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float(np.average((x - np.average(x, weights=w)) ** 2, weights=w))


def build_passing(plays: pd.DataFrame, info: dict, qb_fixed: dict = None, qb_prior: dict = None, rec_fixed: dict = None) -> dict:
    # one model per thing that can happen on a dropback. Every model has the passer, the team's offense and
    # the opposing defense in it together, so a quarterback is not credited for his line and a line not for him
    qb = plays[plays.dropback & plays.qb_id.notna()].copy()
    out = {}

    def fixed(model: str, rows: pd.DataFrame) -> np.ndarray:
        # when the quarterbacks come from a longer look, their effect is taken out first and the rest is left to team and receivers
        given = qb_fixed if qb_fixed is not None else qb_prior
        total = np.zeros(len(rows))
        if given is not None:
            total += rows.qb_id.map(given[model]).fillna(0.0).to_numpy(float)
        if rec_fixed is not None and model in rec_fixed:
            total += rows.receiver_player_id.map(rec_fixed[model]).fillna(0.0).to_numpy(float)
        return total

    def roles(model: str, roles_by_name: dict) -> dict:
        return {k: v for k, v in roles_by_name.items()
                if not (qb_fixed is not None and k == "qb") and not (rec_fixed is not None and k == "receiver")}

    throws = qb[qb.is_throw & qb.cp.notna() & qb.receiver_player_id.notna()]
    y, w = (throws.complete_pass == 1).to_numpy(float), throws.w.to_numpy()
    base = sh.logit(throws.cp.to_numpy()) + fixed("completion", throws)
    offset = calibrate_offset(base, y, w)
    fit, tau = fit_players(y, base + offset, roles("completion", {"qb": throws.qb_id, "receiver": throws.receiver_player_id,
                                                                  "off": throws.posteam, "def": throws.defteam}), w, throws.game_id, TAU_GRID)
    info["completion"] = {"cp_offset": offset, **{f"tau_{r}": t for r, t in tau.items()}}
    out["completion"] = fit

    thr = qb[qb.is_throw & qb.air_yards.notna()]
    y, w = (thr.interception == 1).to_numpy(float), thr.w.to_numpy()
    rate = pd.Series(y * w).groupby(thr.depth.to_numpy()).sum() / pd.Series(w).groupby(thr.depth.to_numpy()).sum()
    fit, tau = fit_players(y, sh.logit(thr.depth.map(rate).to_numpy(float)) + fixed("interception", thr),
                           roles("interception", {"qb": thr.qb_id, "off": thr.posteam, "def": thr.defteam}), w, thr.game_id, TAU_GRID)
    info["interception"] = {**{f"tau_{r}": t for r, t in tau.items()}, "rate_by_depth": rate.to_dict()}
    out["interception"] = fit

    y, w = qb.is_sack.to_numpy(float), qb.w.to_numpy()
    p = np.average(y, weights=w)
    fit, tau = fit_players(y, np.full(len(y), float(sh.logit(p))) + fixed("sack", qb),
                           roles("sack", {"qb": qb.qb_id, "off": qb.posteam, "def": qb.defteam}), w, qb.game_id, TAU_GRID)
    info["sack"] = {**{f"tau_{r}": t for r, t in tau.items()}, "league_rate": p}
    out["sack"] = fit

    y, w = qb.is_scramble.to_numpy(float), qb.w.to_numpy()
    p = np.average(y, weights=w)
    if qb_fixed is None:
        fit, tau = fit_players(y, np.full(len(y), float(sh.logit(p))), {"qb": qb.qb_id}, w, qb.game_id, TAU_GRID)
        info["scramble"] = {"tau_qb": tau["qb"], "league_rate": p}
        out["scramble"] = fit

    c = throws[throws.complete_pass == 1].copy()
    c["expected"] = c.groupby("depth").yards_gained.transform(lambda s: np.average(s, weights=c.loc[s.index, "w"]))
    c["dev"] = c.yards_gained - c.expected
    sigma2 = weighted_var(c.dev, c.w)
    fit, tau = fit_linear_players(c.dev - fixed("completion_yards", c), roles("completion_yards", {
        "qb": c.qb_id, "receiver": c.receiver_player_id, "off": c.posteam, "def": c.defteam}), c.w, c.game_id, sigma2)
    info["completion_yards"] = {**{f"tau_{r}": t for r, t in tau.items()}, "within_sd": sigma2 ** 0.5,
                                "yards_variance": weighted_var(c.yards_gained, c.w)}
    out["completion_yards"] = fit
    out["completion_pct"] = throws.assign(c=throws.complete_pass == 1).groupby("qb_id").apply(
        lambda g: np.average(g.c, weights=g.w), include_groups=False)
    out["dropbacks_w"] = qb.groupby("qb_id").w.sum()
    return out


def madden_prior(plain: dict, long: pd.DataFrame, z: pd.Series, info: dict) -> dict:
    # how much a point of Madden rating is worth in each effect, learned from the quarterbacks with the most plays
    # (their effects are mostly their own results). Backups with little data then start at their Madden level, not at average
    n = plain["dropbacks_w"]
    use = n[n >= 300].index.intersection(z.index)
    prior, slopes = {}, {}
    for model in ("completion", "interception", "sack", "completion_yards"):
        eff = plain[model]["qb"].reindex(use).fillna(0.0)
        zz, w = z.reindex(use).to_numpy(float), n.reindex(use).to_numpy(float)
        slope = float(np.polyfit(zz, eff.to_numpy(float), 1, w=np.sqrt(w))[0])
        slopes[model] = slope
        prior[model] = z * slope
    info["madden_qb_slopes"] = slopes
    return prior


def build_passing_two_looks(recent: pd.DataFrame, long: pd.DataFrame, info: dict, z: pd.Series = None) -> dict:
    # quarterback skill lasts for years, a team's line and receivers change fast. So quarterbacks are rated on a long
    # look, then their effect is fixed and the recent look only sets the team and receiver effects around them
    long_fit = build_passing(long, {})
    models = ("completion", "interception", "sack", "completion_yards")
    if z is not None and len(z):
        prior = madden_prior(long_fit, long, z, info)
        refit = build_passing(long, {}, qb_prior=prior)
        for m in models:
            refit[m]["qb"] = refit[m]["qb"].add(prior[m], fill_value=0.0)
        refit["scramble"] = long_fit["scramble"]
        long_fit = refit
    fixed = {m: long_fit[m]["qb"] for m in models}
    rec_fixed = {m: long_fit[m]["receiver"] for m in ("completion", "completion_yards")} if LONG_SKILL else None
    out = build_passing(recent, info, qb_fixed=fixed, rec_fixed=rec_fixed)
    for m in models:
        out[m]["qb"] = long_fit[m]["qb"]
    for m in ("completion", "completion_yards"):
        if LONG_SKILL:
            out[m]["receiver"] = long_fit[m]["receiver"]
    out["scramble"] = long_fit["scramble"]
    out["completion_pct"], out["dropbacks_w"] = long_fit["completion_pct"], long_fit["dropbacks_w"]
    info["qb_look"] = {"seasons_back": QB_SEASONS_BACK, "half_life_weeks": QB_HALF_LIFE_WEEKS}
    return out


def build_run_model(plays: pd.DataFrame, pos_of: pd.Series, info: dict, runner_fixed: pd.Series = None) -> dict:
    r = plays[plays.is_run & plays.rusher_player_id.notna()].copy()
    r["group"] = r.rusher_player_id.map(pos_of)
    r = r[r.group.isin(["QB", "RB", "WR", "TE"])]
    r["bucket"] = np.select([r.ydstogo <= 2, r.ydstogo <= 6], ["short", "medium"], default="long")
    r["expected"] = r.groupby(["group", "bucket"]).yards_gained.transform(lambda s: np.average(s, weights=r.loc[s.index, "w"]))
    r["dev"] = r.yards_gained - r.expected
    sigma2 = weighted_var(r.dev, r.w)
    roles = {"runner": r.rusher_player_id, "off": r.posteam, "def": r.defteam}
    dev = r.dev
    if runner_fixed is not None:
        # runners come from the long look, so the recent look only sets the line and the defense around them
        dev = dev - r.rusher_player_id.map(runner_fixed).fillna(0.0)
        roles = {k: v for k, v in roles.items() if k != "runner"}
    fit, tau = fit_linear_players(dev, roles, r.w, r.game_id, sigma2)
    info["run_yards" if runner_fixed is None else "run_yards_recent"] = {
        **{f"tau_{k}": t for k, t in tau.items()}, "within_sd": sigma2 ** 0.5, "yards_variance": weighted_var(r.yards_gained, r.w)}
    return fit


def build_run_two_looks(recent: pd.DataFrame, long: pd.DataFrame, pos_of: pd.Series, info: dict) -> dict:
    long_fit = build_run_model(long, pos_of, info)
    if not LONG_SKILL:
        return build_run_model(recent, pos_of, info)
    out = build_run_model(recent, pos_of, info, runner_fixed=long_fit["runner"])
    out["runner"] = long_fit["runner"]
    return out


def team_units(passing: dict, run: dict) -> pd.DataFrame:
    teams = sorted(set(passing["completion"]["off"].index) | set(passing["completion"]["def"].index))
    rows = {}
    for side in ("off", "def"):
        for name, source in (("completion", passing["completion"]), ("interception", passing["interception"]),
                             ("sack", passing["sack"]), ("comp_yards", passing["completion_yards"]), ("run_yards", run)):
            rows[f"{side}_{name}"] = source[side].reindex(teams).fillna(0.0)
    return pd.DataFrame(rows, index=teams).rename_axis("team")


def build_qbs(passing: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=sorted(passing["dropbacks_w"].index))
    out["dropbacks_w"] = passing["dropbacks_w"]
    out["comp_shift"] = passing["completion"]["qb"]
    out["int_shift"] = passing["interception"]["qb"]
    out["sack_shift"] = passing["sack"]["qb"]
    out["scramble_shift"] = passing["scramble"]["qb"]
    out["yards_effect"] = passing["completion_yards"]["qb"]
    out["completion_pct"] = passing["completion_pct"]
    return out.fillna({"comp_shift": 0.0, "int_shift": 0.0, "sack_shift": 0.0, "scramble_shift": 0.0, "yards_effect": 0.0})


def pass_run_snaps(plays: pd.DataFrame, roster: pd.DataFrame, seasons: list) -> pd.DataFrame:
    # snaps per player-game, split into snaps on passing and running plays with the team's mix in that game
    snaps = pd.concat([pd.read_parquet(RAW_DIR / f"snaps_{s}.parquet") for s in seasons], ignore_index=True)
    snaps["team"] = snaps.team.replace(TEAM_FIX)
    ids = roster.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last").set_index("pfr_id").gsis_id
    snaps["gsis_id"] = snaps.pfr_player_id.map(ids)
    snaps = snaps.dropna(subset=["gsis_id"])
    mix = plays.groupby(["season", "week", "posteam"]).agg(n=("dropback", "size"), drops=("dropback", "sum"),
                                                          runs=("is_run", "sum")).reset_index()
    mix = mix.rename(columns={"posteam": "team"})
    s = snaps.merge(mix, on=["season", "week", "team"], how="inner")
    s["pass_snaps"] = s.offense_snaps * s.drops / s.n
    s["run_snaps"] = s.offense_snaps * s.runs / s.n
    s["w"] = 0.5 ** ((s.season.max() * 22 + s.week[s.season == s.season.max()].max() - (s.season * 22 + s.week)) / HALF_LIFE_WEEKS)
    return s


def build_receivers(plays: pd.DataFrame, snaps: pd.DataFrame, pos_of: pd.Series, passing: dict, info: dict) -> pd.DataFrame:
    t = plays[plays.is_throw & plays.receiver_player_id.notna() & plays.air_yards.notna()].copy()
    t["group"] = t.receiver_player_id.map(pos_of)
    t = t[t.group.isin(["WR", "TE", "RB"])]
    exposure = (snaps.assign(e=snaps.pass_snaps * snaps.w).groupby("gsis_id").e.sum())
    rows = pd.DataFrame(index=sorted(t.receiver_player_id.unique()))
    rows["position"] = pos_of
    rows["targets_w"] = t.groupby("receiver_player_id").w.sum()
    rows["pass_snaps_w"] = exposure
    rows = rows.dropna(subset=["pass_snaps_w"])
    rows = rows[rows.pass_snaps_w > 0]

    rows["target_rate"] = np.nan
    info["target_rate"] = {}
    for g in ["WR", "TE", "RB"]:
        m = rows.position == g
        shape, rate = sh.poisson_prior(rows.targets_w[m], rows.pass_snaps_w[m])
        rows.loc[m, "target_rate"] = sh.shrink_rate(rows.targets_w[m], rows.pass_snaps_w[m], shape, rate)
        info["target_rate"][g] = {"league_rate": shape / rate, "prior_snaps": rate}

    counts = (t.groupby(["receiver_player_id", "depth"]).w.sum().unstack(fill_value=0).reindex(columns=DEPTHS, fill_value=0))
    for g in ["WR", "TE", "RB"]:
        ids = [i for i in rows.index[rows.position == g] if i in counts.index]
        k = sh.dirichlet_strength(counts.loc[ids].to_numpy())
        shares = sh.shrink_shares(counts.loc[ids].to_numpy(), k)
        for j, d in enumerate(DEPTHS):
            rows.loc[ids, f"share_{d}"] = shares[:, j]
        info.setdefault("depth_mix", {})[g] = {"prior_throws": k,
                                               "league_mix": (counts.loc[ids].sum() / counts.loc[ids].sum().sum()).to_dict()}
    rows["comp_shift"] = passing["completion"]["receiver"].reindex(rows.index).fillna(0.0)
    rows["yards_effect"] = passing["completion_yards"]["receiver"].reindex(rows.index).fillna(0.0)
    return rows


def build_runners(plays: pd.DataFrame, snaps: pd.DataFrame, pos_of: pd.Series, run_fit: dict, info: dict) -> pd.DataFrame:
    r = plays[plays.is_run & plays.rusher_player_id.notna()].copy()
    r["group"] = r.rusher_player_id.map(pos_of)
    r = r[r.group.isin(["QB", "RB", "WR", "TE"])]
    exposure = snaps.assign(e=snaps.run_snaps * snaps.w).groupby("gsis_id").e.sum()
    rows = pd.DataFrame(index=sorted(r.rusher_player_id.unique()))
    rows["position"] = pos_of
    rows["carries_w"] = r.groupby("rusher_player_id").w.sum()
    rows["run_snaps_w"] = exposure
    rows = rows.dropna(subset=["run_snaps_w"])
    rows = rows[rows.run_snaps_w > 0]
    rows["carry_rate"] = np.nan
    info["carry_rate"] = {}
    for g in ["QB", "RB", "WR", "TE"]:
        m = rows.position == g
        if m.sum() < 5:
            continue
        shape, rate = sh.poisson_prior(rows.carries_w[m], rows.run_snaps_w[m])
        rows.loc[m, "carry_rate"] = sh.shrink_rate(rows.carries_w[m], rows.run_snaps_w[m], shape, rate)
        info["carry_rate"][g] = {"league_rate": shape / rate, "prior_snaps": rate}
    rows["yards_over_expected"] = run_fit["runner"].reindex(rows.index).fillna(0.0)
    return rows


def snap_shares(snaps_all: pd.DataFrame, roster: pd.DataFrame, before: tuple, games: int = 4) -> pd.Series:
    # share of the team's offensive plays each player was on the field for, over his last few games
    s = snaps_all[(snaps_all.season * 100 + snaps_all.week) < before[0] * 100 + before[1]].copy()
    ids = roster.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last").set_index("pfr_id").gsis_id
    s["gsis_id"] = s.pfr_player_id.map(ids)
    s = s.dropna(subset=["gsis_id"]).sort_values(["season", "week"])
    team_max = s.groupby(["season", "week", "team"]).offense_snaps.transform("max")
    s["share"] = s.offense_snaps / team_max
    return s.groupby("gsis_id").tail(games).groupby("gsis_id").share.mean()


def build_lineup_inputs(roster_all: pd.DataFrame, snaps_all: pd.DataFrame, injuries_all: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    # who is on each team going into this week, and who is listed out. Uses nothing from the week itself or later
    r = roster_all[roster_all.season == season]
    r = r[r.week <= week]
    last = r.groupby("team").week.max().rename("last_week")
    cur = r.merge(last, on="team")
    cur = cur[cur.week == cur.last_week].sort_values("week").drop_duplicates("gsis_id", keep="last")
    cur = cur[cur.status.isin(["ACT", "INA", "RES"])]
    cur = cur[["gsis_id", "full_name", "team", "position", "status"]].copy()
    inj = injuries_all[(injuries_all.season == season) & (injuries_all.week <= week)]
    inj = inj.sort_values("week").drop_duplicates("gsis_id", keep="last")
    inj = inj[inj.week == min(week, inj.week.max())]
    cur = cur.merge(inj[["gsis_id", "report_status"]].rename(columns={"report_status": "injury_status"}), on="gsis_id", how="left")
    cur["out"] = (cur.status != "ACT") | cur.injury_status.isin(["Out", "Doubtful"])
    cur["snap_share"] = cur.gsis_id.map(snap_shares(snaps_all, roster_all, (season, week))).fillna(0.0)
    return cur


def load_raw(kind: str, seasons: list) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(RAW_DIR / f"{kind}_{s}.parquet") for s in seasons], ignore_index=True)


def build_ratings(seasons: list, before=None) -> dict:
    # everything the engine needs, using only plays before (season, week) when before is given
    plays = load_plays(seasons, before)
    long_seasons = list(range(seasons[-1] - QB_SEASONS_BACK, seasons[-1] + 1))
    long_plays = load_plays(long_seasons, before, QB_HALF_LIFE_WEEKS)
    roster = latest_roster()
    pos_of = roster.drop_duplicates("gsis_id", keep="last").set_index("gsis_id").position.replace({"FB": "RB"})
    snaps = pass_run_snaps(plays, roster, seasons)
    info = {"seasons": seasons, "half_life_weeks": HALF_LIFE_WEEKS}
    z = madden.z_scores("QB", roster) if before is None else None
    passing = build_passing_two_looks(plays, long_plays, info, z)
    run_fit = build_run_two_looks(plays, long_plays, pos_of, info)
    qbs = build_qbs(passing)
    qbs = qbs[qbs.index.map(pos_of) == "QB"]
    names = roster.drop_duplicates("gsis_id", keep="last").set_index("gsis_id").full_name
    tables = {"qb": qbs, "receiver": build_receivers(plays, snaps, pos_of, passing, info),
              "runner": build_runners(plays, snaps, pos_of, run_fit, info), "team": team_units(passing, run_fit)}
    for key in ("qb", "receiver", "runner"):
        tables[key] = tables[key].copy()
        tables[key].insert(0, "name", names.reindex(tables[key].index))
        tables[key].index.name = "gsis_id"
    return {"tables": tables, "info": info}


def main():
    last = latest_season()
    seasons = list(range(last - SEASONS_BACK, last + 1))
    print(f"seasons {seasons}, half life {HALF_LIFE_WEEKS} weeks")
    built = build_ratings(seasons)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in built["tables"].items():
        df.round(5).to_csv(OUT_DIR / f"{name}_ratings.csv" if name != "team" else OUT_DIR / "team_units.csv")
    roster = latest_roster()
    week = int(roster[roster.season == last].week.max())
    injuries = load_raw("injuries", seasons)
    inputs = build_lineup_inputs(roster, load_raw("snaps", seasons), injuries, last, week + 1)
    inputs.to_csv(OUT_DIR / "current_roster.csv", index=False)
    latest = injuries[injuries.season == injuries.season.max()]
    built["info"]["as_of"] = f"{int(latest.season.max())} week {int(latest.week.max())}"
    (OUT_DIR / "priors.json").write_text(json.dumps(built["info"], indent=2, default=float))
    t = built["tables"]
    print(f"qbs {len(t['qb'])}, receivers {len(t['receiver'])}, runners {len(t['runner'])}, "
          f"teams {len(t['team'])}, roster rows {len(inputs)}")


if __name__ == "__main__":
    main()
