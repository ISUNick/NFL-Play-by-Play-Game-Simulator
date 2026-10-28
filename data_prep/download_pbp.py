import sys
from pathlib import Path

import requests

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
RAW_DIR = Path("data/raw")


def download_season(year: int) -> bool:
    dest = RAW_DIR / f"pbp_{year}.parquet"
    if dest.exists():
        print(f"{year}: already downloaded")
        return True
    resp = requests.get(URL.format(year=year), timeout=120)
    if resp.status_code != 200:
        print(f"{year}: failed with HTTP {resp.status_code}")
        return False
    dest.write_bytes(resp.content)
    print(f"{year}: saved {len(resp.content) / 1e6:.1f} MB")
    return True


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = sum(download_season(y) for y in range(first, last + 1))
    print(f"Done: {ok}/{last - first + 1} seasons available")


if __name__ == "__main__":
    main()
