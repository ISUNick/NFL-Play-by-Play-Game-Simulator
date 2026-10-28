import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_tables import TABLE_DIR, build_throw_frame, load_plays

TARGET_N = 800
MAX_SAMPLES = 1000
MIN_CELL = 150
SEED = 42

OPEN_FIELD_WINDOW = 30
NEAR_GOAL_YARD = 20
NEAR_OWN_GOAL_YARD = 90
NEAR_GOAL_WINDOW = 4
NEAR_GOAL_LAST_RESORT = 12
SPARSE_WINDOW = 40
SPARSE_MIN_CELL = 40

MIN_YARD_FOR_DEPTH = {"short": 1, "medium": 10, "deep": 20}


def prepare(yl: np.ndarray) -> np.ndarray:
    return np.cumsum(np.bincount(yl, minlength=101))


def window_indices(yl: np.ndarray, cum: np.ndarray, yard: int, rng):
    near = yard <= NEAR_GOAL_YARD or yard >= NEAR_OWN_GOAL_YARD
    for max_w in ([NEAR_GOAL_WINDOW, NEAR_GOAL_LAST_RESORT] if near else [OPEN_FIELD_WINDOW]):
        for w in range(max_w + 1):
            lo, hi = max(1, yard - w), min(99, yard + w)
            if cum[hi] - cum[lo - 1] >= TARGET_N:
                break
        idx = np.flatnonzero((yl >= lo) & (yl <= hi))
        if len(idx) >= MIN_CELL:
            if len(idx) > MAX_SAMPLES:
                idx = rng.choice(idx, MAX_SAMPLES, replace=False)
            return idx
    # a short list of seasons can leave a yard line nearly empty, so widen the window and accept fewer plays
    lo, hi = max(1, yard - SPARSE_WINDOW), min(99, yard + SPARSE_WINDOW)
    idx = np.flatnonzero((yl >= lo) & (yl <= hi))
    if len(idx) >= SPARSE_MIN_CELL:
        return rng.choice(idx, MAX_SAMPLES, replace=False) if len(idx) > MAX_SAMPLES else idx
    return None


def build_bank(df: pd.DataFrame, key_col, rng) -> pd.DataFrame:
    df = df.copy()
    df["yardline_100"] = df["yardline_100"].clip(1, 99).astype(int)
    all_yl = df["yardline_100"].to_numpy()
    all_cum = prepare(all_yl)
    keys = sorted(df[key_col].unique()) if key_col else ["ALL"]

    rows = []
    for yard in range(1, 100):
        pooled = window_indices(all_yl, all_cum, yard, rng)
        assert pooled is not None, f"not enough real plays near yard line {yard}"
        for key in keys:
            if key_col == "depth" and yard < MIN_YARD_FOR_DEPTH[key]:
                continue
            if key_col:
                sub = df[df[key_col] == key]
                sub_yl = sub["yardline_100"].to_numpy()
                idx = window_indices(sub_yl, prepare(sub_yl), yard, rng) if len(sub) else None
                chosen = sub.iloc[idx] if idx is not None else df.iloc[pooled]
            else:
                chosen = df.iloc[pooled]
            yards = chosen["yards"].to_numpy().astype(int)
            reached = (yards >= yard).astype(int)
            if yard <= NEAR_GOAL_YARD:
                td = chosen["td"].to_numpy().astype(int) | reached
            else:
                td = reached
            out = pd.DataFrame({"key": key, "yardline_100": yard, "yards": yards,
                                "oob": chosen["oob"].to_numpy().astype(int), "td": td})
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"yards_gained": "yards"})
    df["oob"] = (df["out_of_bounds"] == 1).astype(int)
    df["td"] = ((df["pass_touchdown"] == 1) | (df["rush_touchdown"] == 1)).astype(int)
    return df.dropna(subset=["yards", "yardline_100"])


def describe(bank: pd.DataFrame, label: str, key=None):
    print(f"\n{label}" + (f" ({key})" if key else ""))
    print("  yard line   mean yards   touchdown share   biggest gain   samples")
    b = bank if key is None else bank[bank["key"] == key]
    for yard in [1, 2, 4, 8, 15, 30, 50, 80, 95]:
        y = b[b["yardline_100"] == yard]["yards"]
        if len(y) == 0:
            print(f"  {yard:9d}   (this depth cannot happen here)")
            continue
        print(f"  {yard:9d}   {y.mean():10.2f}   {b[b["yardline_100"] == yard]["td"].mean():16.1%}   "
              f"{int(y.max()):12d}   {len(y):7d}")


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    plays = load_plays()
    throws = build_throw_frame(plays)

    completions = tidy(throws[throws["p_complete"] == 1])[["depth", "yardline_100", "yards", "oob", "td"]]
    runs = tidy(plays[~plays["is_dropback"]])[["dist_b", "yardline_100", "yards", "oob", "td"]]
    scrambles = tidy(plays[plays["p_scramble"] == 1])[["yardline_100", "yards", "oob", "td"]]
    sacks = tidy(plays[plays["p_sack"] == 1])[["yardline_100", "yards", "oob", "td"]]

    banks = {
        "bank_completion": build_bank(completions, "depth", rng),
        "bank_run": build_bank(runs, "dist_b", rng),
        "bank_scramble": build_bank(scrambles, None, rng),
        "bank_sack": build_bank(sacks, None, rng),
    }
    for name, bank in banks.items():
        bank.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(bank):,} rows")

    ints = throws[throws["p_int"] == 1].dropna(subset=["return_yards"])
    bank_int = pd.DataFrame({
        "depth": ints["depth"], "air_yards": ints["air_yards"].astype(int),
        "return_yards": ints["return_yards"].astype(int),
        "return_td": (ints["return_touchdown"] == 1).astype(int)})
    bank_int.to_csv(TABLE_DIR / "bank_int.csv", index=False)
    print(f"bank_int.csv: {len(bank_int):,} rows")

    describe(banks["bank_run"], "RUN yards by yard line")
    describe(banks["bank_completion"], "COMPLETED PASS yards", "short")
    describe(banks["bank_completion"], "COMPLETED PASS yards", "deep")
    describe(banks["bank_scramble"], "QB SCRAMBLE yards")
    describe(banks["bank_sack"], "SACK yards")
    c4 = banks["bank_completion"]
    c4 = c4[c4["yardline_100"] == 4]
    print(f"\nCompleted passes from the 4 yard line: biggest gain in the bank is "
          f"{int(c4['yards'].max())} yards, so an 11 yard gain from the 4 cannot happen.")
    print(f"Interceptions: mean return {bank_int['return_yards'].mean():.1f} yards, "
          f"pick six rate {bank_int['return_td'].mean():.1%}")
