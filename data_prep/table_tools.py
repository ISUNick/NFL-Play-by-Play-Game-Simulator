import json
from pathlib import Path

import pandas as pd

MIN_PLAYS = 100


def build_levels_table(df: pd.DataFrame, levels: list, outcome_cols: list,
                       min_n: int = MIN_PLAYS, shrink: float = 0) -> pd.DataFrame:
    key_universe = []
    for _, keys in levels:
        for k in keys:
            if k not in key_universe:
                key_universe.append(k)

    pieces = []
    for level_name, keys in levels:
        if keys:
            grouped = df.groupby(keys)
            g = grouped[outcome_cols].mean()
            g["n"] = grouped.size()
            g = g.reset_index()
        else:
            g = pd.DataFrame([df[outcome_cols].mean()])
            g["n"] = len(df)
        g = g[g["n"] >= min_n].copy()
        for k in key_universe:
            if k not in g.columns:
                g[k] = "ALL"
        g["level"] = level_name
        pieces.append(g[["level"] + key_universe + ["n"] + outcome_cols])
    out = pd.concat(pieces, ignore_index=True)
    if shrink > 0:
        out = _shrink(out, levels, key_universe, outcome_cols, shrink)
    return out


def _shrink(table: pd.DataFrame, levels: list, key_universe: list, outcome_cols: list, k: float) -> pd.DataFrame:
    names = [n for n, _ in levels]
    keys_of = dict(levels)
    by_level = {n: table[table["level"] == n].copy() for n in names}
    lookup = {}
    for i in range(len(names) - 1, -1, -1):
        name, keys = names[i], keys_of[names[i]]
        t = by_level[name]
        parent_keys = keys_of[names[i + 1]] if i + 1 < len(names) else None
        for idx, row in t.iterrows():
            if parent_keys is None:
                values = row[outcome_cols].astype(float).to_dict()
            else:
                parent = lookup.get((names[i + 1], tuple(str(row[c]) for c in parent_keys)))
                if parent is None:
                    values = row[outcome_cols].astype(float).to_dict()
                else:
                    n = row["n"]
                    values = {c: (row[c] * n + k * parent[c]) / (n + k) for c in outcome_cols}
            for c in outcome_cols:
                t.at[idx, c] = values[c]
            lookup[(name, tuple(str(row[c]) for c in keys))] = values
        by_level[name] = t
    return pd.concat([by_level[n] for n in names], ignore_index=True)


def lookup_row(table: pd.DataFrame, levels: list, situation: dict):
    for level_name, keys in levels:
        t = table[table["level"] == level_name]
        for k in keys:
            t = t[t[k].astype(str) == str(situation[k])]
        if len(t):
            return level_name, t.iloc[0]
    raise ValueError("no match, impossible because the last level has no keys")


def with_key(levels: list, key: str) -> list:
    return [(name, [key] + keys) for name, keys in levels]


def update_specs(table_dir: Path, name: str, filename: str, levels: list, outcomes: list):
    path = table_dir / "table_specs.json"
    specs = json.loads(path.read_text()) if path.exists() else {}
    specs[name] = {"file": filename, "levels": [[n, k] for n, k in levels], "outcomes": outcomes}
    path.write_text(json.dumps(specs, indent=2))
