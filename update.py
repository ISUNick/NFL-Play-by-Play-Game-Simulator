import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YEAR = 2026


def run(*cmd):
    print(" ".join(cmd), flush=True)
    if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode != 0:
        sys.exit("step failed")


if __name__ == "__main__":
    run("data_prep/download_data.py", str(YEAR), str(YEAR), "--refresh")
    run("data_prep/build_players.py")
    run("calibration/predict.py")
    print("Updated. Restart the dashboard to load the new ratings.")
