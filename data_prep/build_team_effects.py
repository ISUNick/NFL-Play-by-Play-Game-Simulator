from pathlib import Path

import numpy as np
import pandas as pd
from paths import pbp_files, table_dir

RAW_DIR = Path("data/raw")
TABLE_DIR = table_dir()
COLS = ["season", "week", "season_type", "posteam", "defteam", "down", "ydstogo", "play_type", "qb_dropback",
        "qb_scramble", "rush_attempt", "sack", "interception", "complete_pass", "air_yards", "penalty",
        "yards_gained", "qtr"]


CHANNELS = ["completion", "interception", "sack", "completion_yards", "run_yards"]


def load() -> pd.DataFrame:
    files = pbp_files(RAW_DIR)
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df[(df["season_type"] == "REG") & df["play_type"].isin(["pass", "run"]) & (df["penalty"].fillna(0) != 1)]
    df = df.dropna(subset=["posteam", "defteam", "down"]).copy()
    df["dropback"] = df["qb_dropback"] == 1
    designed = (df["rush_attempt"] == 1) & (df["qb_scramble"] != 1) & ~df["dropback"]
    return df[df["dropback"] | designed].copy()


def channel_frames(df: pd.DataFrame) -> dict:
    out = {}
    d = df.copy()
    d["dist"] = d["ydstogo"].clip(upper=10)
    d["is_sack"] = (d["sack"] == 1).astype(float)

    db = d[d["dropback"]].copy()
    db["y"] = db["is_sack"]
    db["r"] = db["y"] - db.groupby("down")["y"].transform("mean")
    out["sack"] = db

    th = db[(db["is_sack"] == 0) & (db["qb_scramble"] != 1)].dropna(subset=["air_yards"]).copy()
    th["depth"] = np.select([th["air_yards"] <= 9, th["air_yards"] <= 19], ["short", "medium"], "deep")
    th["y"] = (th["complete_pass"] == 1).astype(float)
    th["r"] = th["y"] - th.groupby("depth")["y"].transform("mean")
    out["completion"] = th
    ti = th.copy()
    ti["y"] = (ti["interception"] == 1).astype(float)
    ti["r"] = ti["y"] - ti.groupby("depth")["y"].transform("mean")
    out["interception"] = ti
    cy = th[th["complete_pass"] == 1].copy()
    cy["y"] = cy["yards_gained"].astype(float)
    cy["r"] = cy["y"] - cy.groupby("depth")["y"].transform("mean")
    out["completion_yards"] = cy

    runs = d[~d["dropback"]].copy()
    runs["y"] = runs["yards_gained"].astype(float)
    runs["r"] = runs["y"] - runs.groupby(["down", "dist"])["y"].transform("mean")
    out["run_yards"] = runs
    return out


def true_sd(frame: pd.DataFrame, side: str) -> tuple:
    team = "posteam" if side == "offense" else "defteam"
    f = frame.copy()
    f["half"] = (f["week"] % 2).astype(int)
    g = f.groupby([team, "season", "half"])["r"].agg(["mean", "count"]).unstack("half").dropna()
    a, b = g[("mean", 0)].to_numpy(), g[("mean", 1)].to_numpy()
    cov = float(np.cov(a, b)[0, 1])
    plays = float(g["count"].sum(axis=1).mean())
    return (max(cov, 0.0) ** 0.5, float(np.std((a + b) / 2)), plays)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df):,} plays, seasons {df['season'].min()} to {df['season'].max()}")
    frames = channel_frames(df)
    base = {"completion": frames["completion"]["y"].mean(), "interception": frames["interception"]["y"].mean(),
            "sack": frames["sack"]["y"].mean(), "completion_yards": frames["completion_yards"]["y"].mean(),
            "run_yards": frames["run_yards"]["y"].mean()}
    rows = []
    print(f"{'channel':18} {'side':8} {'base':>8} {'true sd':>9} {'observed sd':>12} {'plays/team-season':>18}")
    for ch in CHANNELS:
        for side in ("offense", "defense"):
            sd, obs, plays = true_sd(frames[ch], side)
            rows.append({"channel": ch, "side": side, "base": base[ch], "var": frames[ch]["y"].var(), "true_sd": sd})
            print(f"{ch:18} {side:8} {base[ch]:8.4f} {sd:9.4f} {obs:12.4f} {plays:18.0f}")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "team_effects.csv", index=False)
    print("wrote team_effects.csv")


if __name__ == "__main__":
    main()
