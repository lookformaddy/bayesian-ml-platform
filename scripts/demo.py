"""Run the Bayesian ML Platform end-to-end demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bayesian.demo import run_demo

if __name__ == "__main__":
    run_demo()
