import re
from pathlib import Path

import numpy as np
import pandas as pd

EXTERNAL = Path(__file__).resolve().parent.parent / "data" / "external"
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", SUFFIX.sub("", re.sub(r"[^a-z ]", "", str(name).lower().replace(".", "")))).strip()


def ratings(position: str) -> pd.Series:
    # Madden overall by cleaned name, empty if the file is not there
    path = EXTERNAL / f"madden_{position.lower()}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(path)
    return pd.Series(d.overall.to_numpy(float), index=[clean(n) for n in d.name]).groupby(level=0).first()


def z_scores(position: str, roster: pd.DataFrame, floor_z: float = -1.5) -> pd.Series:
    # standard score of Madden overall for every roster player of that position, by gsis id. Unrated players get the floor
    r = ratings(position)
    if r.empty:
        return pd.Series(dtype=float)
    mean, sd = r.mean(), r.std()
    ids = roster[roster.position == position].drop_duplicates("gsis_id", keep="last")
    z = pd.Series([(r.get(clean(n), np.nan) - mean) / sd for n in ids.full_name], index=ids.gsis_id.to_numpy())
    return z.fillna(floor_z)
