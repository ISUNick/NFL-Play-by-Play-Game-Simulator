import sys
from pathlib import Path

import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
# local name -> (release tag, file name on nflverse), {year} is filled in
DATASETS = {
    "pbp": ("pbp", "play_by_play_{year}.parquet"),
    "roster_week": ("weekly_rosters", "roster_weekly_{year}.parquet"),
    "injuries": ("injuries", "injuries_{year}.parquet"),
    "snaps": ("snap_counts", "snap_counts_{year}.parquet"),
    "part": ("pbp_participation", "pbp_participation_{year}.parquet"),
}
RAW_DIR = Path("data/raw")


def download(name: str, year: int, refresh: bool) -> bool:
    tag, pattern = DATASETS[name]
    dest = RAW_DIR / f"{name}_{year}.parquet"
    if dest.exists() and not refresh:
        print(f"{name} {year}: already downloaded")
        return True
    resp = requests.get(f"{BASE}/{tag}/{pattern.format(year=year)}", timeout=120)
    # the current season's participation file doesn't exist until nflverse publishes it
    if resp.status_code != 200 or len(resp.content) < 1000:
        print(f"{name} {year}: not available (HTTP {resp.status_code})")
        return False
    dest.write_bytes(resp.content)
    print(f"{name} {year}: saved {len(resp.content) / 1e6:.1f} MB")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh" in sys.argv
    first = int(args[0]) if args else 2023
    last = int(args[1]) if len(args) > 1 else 2026
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(first, last + 1):
        for name in DATASETS:
            download(name, year, refresh)


if __name__ == "__main__":
    main()
