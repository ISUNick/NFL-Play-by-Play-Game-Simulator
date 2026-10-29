import datetime
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import update


def test_the_season_changes_in_august():
    assert update.current_season(datetime.date(2026, 10, 28)) == 2026
    assert update.current_season(datetime.date(2027, 1, 20)) == 2026
    assert update.current_season(datetime.date(2027, 7, 31)) == 2026
    assert update.current_season(datetime.date(2027, 8, 1)) == 2027


def test_a_recent_update_is_skipped():
    update.STAMP = Path(tempfile.mkdtemp()) / "updated.txt"
    now = datetime.datetime(2026, 10, 28, 12, 0)
    assert not update.is_fresh(now)
    update.STAMP.write_text((now - datetime.timedelta(hours=2)).isoformat())
    assert update.is_fresh(now)
    update.STAMP.write_text((now - datetime.timedelta(hours=7)).isoformat())
    assert not update.is_fresh(now)


def test_no_internet_keeps_the_old_ratings():
    update.STAMP = Path(tempfile.mkdtemp()) / "updated.txt"
    calls = []
    update.download_seasons = lambda season, log=print: False
    update.subprocess = types.SimpleNamespace(run=lambda *a, **k: calls.append(a))
    assert update.refresh(season=2026, predictions=False, log=lambda m: None) is False
    assert not calls, "nothing is rebuilt when the download failed"
    assert not update.STAMP.exists()


def test_a_good_update_rebuilds_and_stamps():
    update.STAMP = Path(tempfile.mkdtemp()) / "updated.txt"
    ran = []

    def fake_run(cmd, **k):
        ran.append(Path(cmd[1]).name)
        return subprocess.CompletedProcess(cmd, 0)

    update.download_seasons = lambda season, log=print: True
    update.subprocess = types.SimpleNamespace(run=fake_run)
    assert update.refresh(season=2026, predictions=False, log=lambda m: None) is True
    assert ran == ["build_players.py"] and update.is_fresh()
    ran.clear()
    assert update.refresh(season=2026, predictions=False, log=lambda m: None) is True
    assert not ran, "a second start within hours does not rebuild"
    assert update.refresh(season=2026, force=True, predictions=False, log=lambda m: None) is True
    assert ran == ["build_players.py"]


def test_a_failed_rebuild_keeps_the_old_ratings():
    update.STAMP = Path(tempfile.mkdtemp()) / "updated.txt"
    update.download_seasons = lambda season, log=print: True
    update.subprocess = types.SimpleNamespace(run=lambda cmd, **k: subprocess.CompletedProcess(cmd, 1))
    assert update.refresh(season=2026, predictions=False, log=lambda m: None) is False
    assert not update.STAMP.exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All update checks passed.")
