import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["build_tables.py", "build_banks.py", "build_clock.py", "build_timeout.py", "build_penalty.py",
         "build_special.py", "build_half.py", "build_late_timeout.py", "build_overtime.py", "build_team_effects.py", "build_final_calls.py"]


if __name__ == "__main__":
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        if subprocess.run([sys.executable, str(HERE / step)], cwd=HERE.parent).returncode != 0:
            sys.exit(f"{step} failed, stopping")
    print("\nAll tables built. Now run: python tests/run_all.py")
