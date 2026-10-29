import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data_prep"))
import download_data

STAMP = ROOT / "data" / "players" / "updated.txt"
MAX_AGE_HOURS = 6
SEASONS_NEEDED = 4  # the long quarterback look reaches three seasons back


def current_season(today: datetime.date = None) -> int:
    # the NFL season that is under way, a new one starts in August
    today = today or datetime.date.today()
    return today.year if today.month >= 8 else today.year - 1


def last_update():
    if not STAMP.exists():
        return None
    return datetime.datetime.fromisoformat(STAMP.read_text().strip())


def is_fresh(now: datetime.datetime = None) -> bool:
    stamp, now = last_update(), now or datetime.datetime.now()
    return stamp is not None and now - stamp < datetime.timedelta(hours=MAX_AGE_HOURS)


def download_seasons(season: int, log=print) -> bool:
    download_data.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(season - SEASONS_NEEDED + 1, season + 1):
        for name in download_data.DATASETS:
            got = download_data.download(name, year, refresh=year == season)
            if not got and name == "pbp" and year == season:
                log(f"could not get {year} play by play")
                return False
    return True


def refresh(season: int = None, force: bool = False, predictions: bool = True, log=print) -> bool:
    # downloads the newest games, injuries and rosters, rebuilds every rating and starts the season predictions.
    # Returns False when it could not (no internet, say) and the ratings already on disk are kept
    season = season or current_season()
    if not force and is_fresh():
        log(f"ratings were updated {last_update():%b %d %H:%M}, skipping")
        return True
    try:
        if not download_seasons(season, log):
            return False
        built = subprocess.run([sys.executable, "data_prep/build_players.py"], cwd=ROOT)
    except Exception as e:
        log(f"update failed: {e}")
        return False
    if built.returncode != 0:
        log("rebuilding ratings failed, keeping the old ones")
        return False
    STAMP.write_text(datetime.datetime.now().isoformat(timespec="seconds"))
    if predictions:
        subprocess.Popen([sys.executable, "calibration/predict.py", "--season", str(season), "--workers", "1"], cwd=ROOT)
    return True


if __name__ == "__main__":
    ok = refresh(force=True, predictions=False)
    if ok:
        subprocess.run([sys.executable, "calibration/predict.py", "--season", str(current_season())], cwd=ROOT)
        print("Updated. Restart the dashboard to load the new ratings.")
    else:
        sys.exit("update failed")
