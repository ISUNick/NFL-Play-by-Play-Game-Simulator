import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["test_madden_qb.py", "test_state.py", "test_clock_rules.py", "test_tables.py", "test_clock_table.py", "test_special.py",
         "test_engine_tables.py", "test_engine_special.py", "test_play.py", "test_game.py", "test_half.py",
         "test_onside.py", "test_overtime.py", "test_penalty.py", "test_late_timeout.py", "test_teams.py",
         "test_boxscore.py", "test_season.py", "test_home_field.py", "test_ratings.py", "test_lineup.py",
         "test_backtest.py", "test_winprob.py", "test_app.py", "test_era_predict.py"]


def main() -> int:
    failed = 0
    for name in TESTS:
        result = subprocess.run([sys.executable, str(ROOT / "tests" / name)], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"PASSED  {name}")
        else:
            failed += 1
            print(f"FAILED  {name}")
            print((result.stdout + result.stderr).strip()[-2000:])
    print("\nAll tests passed." if not failed else f"\n{failed} test file(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
