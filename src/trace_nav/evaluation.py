"""Summarize actual episode logs without inventing confidence intervals."""
import argparse
import json
from pathlib import Path
import numpy as np


def prediction_metrics(predicted, target, bins=10):
    p, y = np.asarray(predicted), np.asarray(target)
    if p.shape != y.shape or not p.size:
        raise ValueError("Predictions and targets must have equal nonempty shapes")
    if np.any((p < 0) | (p > 1) | (y < 0) | (y > 1)):
        raise ValueError("Probabilities and labels must lie in [0, 1]")
    assignments = np.minimum((p*bins).astype(int), bins-1)
    ece = sum(np.mean(assignments == i)*abs(p[assignments == i].mean()-y[assignments == i].mean())
              for i in range(bins) if np.any(assignments == i))
    return {"brier": float(np.mean((p-y)**2)), "ece": float(ece)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.episodes).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No completed episodes")
    successes = [r for r in rows if r["status"] == "success"]
    result = {"episodes": len(rows), "SR": len(successes)/len(rows),
              "CR": sum(r["status"] == "collision" for r in rows)/len(rows),
              "TR": sum(r["status"] == "timeout" for r in rows)/len(rows),
              "other": sum(r["status"] not in {"success", "collision", "timeout"} for r in rows)/len(rows),
              "AT": float(np.mean([r["simulation_seconds"] for r in successes])) if successes else None}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
