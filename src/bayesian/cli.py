"""CLI for the Bayesian ML Platform.

Usage:
    python -m bayesian.cli demo
    python -m bayesian.cli train [--version v1]
    python -m bayesian.cli predict 0.1 -0.2 0.3 0.0 0.5 -0.1
    python -m bayesian.cli bandit
    python -m bayesian.cli drift
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .demo import run_demo
from .platform import BayesianPlatform


def _platform() -> BayesianPlatform:
    return BayesianPlatform(Settings())


def cmd_demo(_args) -> int:
    run_demo()
    return 0


def cmd_train(args) -> int:
    platform = _platform()
    result = platform.run_pipeline(version=args.version)
    print(json.dumps(result.summary(), indent=2))
    return 0


def cmd_predict(args) -> int:
    platform = _platform()
    result = platform.run_pipeline(version="v1")
    out = result.server.predict_one(args.features).to_dict()
    print(json.dumps(out, indent=2))
    return 0


def cmd_bandit(_args) -> int:
    print(json.dumps(_platform().run_bandit_comparison(), indent=2))
    return 0


def cmd_drift(_args) -> int:
    platform = _platform()
    result = platform.run_pipeline(version="v1")
    print(json.dumps(platform.run_drift_scenario(result), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bayesian-ml-platform")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="run the end-to-end demo")
    p_train = sub.add_parser("train", help="train, calibrate, gate, register")
    p_train.add_argument("--version", default="v1")
    p_pred = sub.add_parser("predict", help="predict one row (6 features)")
    p_pred.add_argument("features", nargs="+", type=float)
    sub.add_parser("bandit", help="Thompson sampling vs epsilon-greedy")
    sub.add_parser("drift", help="run the drift scenario")

    args = parser.parse_args(argv)
    handlers = {
        "demo": cmd_demo,
        "train": cmd_train,
        "predict": cmd_predict,
        "bandit": cmd_bandit,
        "drift": cmd_drift,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
