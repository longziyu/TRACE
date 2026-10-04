"""Calibrated screening, lexicographic fallback, and immutable decision evidence."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class Calibration:
    x: list
    y: list
    beta: float
    threshold: float
    model_sha256: str

    def __post_init__(self):
        if len(self.x) != len(self.y) or not len(self.x):
            raise ValueError("Calibration interpolation knots must be nonempty and aligned")
        if not np.isfinite(self.x+self.y+[self.beta, self.threshold]).all():
            raise ValueError("Calibration values must be finite")
        if np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.y) < 0):
            raise ValueError("Calibration knots must be sorted and isotonic")
        if self.beta < 0 or not 0 < self.threshold < 1 or min(self.y) < 0 or max(self.y) > 1:
            raise ValueError("Invalid risk calibration range")

    def apply(self, p):
        return np.interp(p, self.x, self.y)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Decision:
    timestamp: float
    candidates: list
    predictions: list
    uncertainty: list
    risk: list
    admissible: list
    costs: list
    selected: int
    comparator: int
    contributions: list
    no_admissible: bool
    behavior: str

    def dictionary(self):
        return asdict(self)


def behavior(sequence):
    v, w = np.mean(sequence, axis=0)
    if v < 0.03 and abs(w) < 0.06:
        return "brake"
    if abs(w) > 0.12:
        return "turn_left" if w > 0 else "turn_right"
    return "continue"


def select(candidates, ensemble, calibration, cfg, timestamp):
    if not np.isfinite(ensemble).all() or not np.isfinite(candidates).all():
        raise ValueError("Nonfinite controller outputs")
    mean = ensemble.mean(axis=0)
    std = ensemble[..., 0].std(axis=0)
    risk = np.clip(calibration.apply(mean[:, 0]) + calibration.beta * std, 0, 1)
    admissible = risk <= calibration.threshold
    cost = mean[:, 1:] @ np.asarray(cfg.weights)
    valid = np.flatnonzero(admissible)
    chosen = int(valid[np.argmin(cost[valid])]) if len(valid) else int(np.lexsort((cost, risk))[0])
    # Prefer the explicit forward candidate as the contrast, unless it was selected.
    comparator = cfg.learned_candidates + 1
    if comparator == chosen:
        comparator = min((i for i in range(len(cost)) if i != chosen), key=lambda i: cost[i])
    delta = (mean[comparator, 1:] - mean[chosen, 1:]) * np.asarray(cfg.weights)
    return Decision(float(timestamp), candidates.tolist(), mean.tolist(), std.tolist(), risk.tolist(),
                    admissible.tolist(), cost.tolist(), chosen, int(comparator), delta.tolist(),
                    not bool(len(valid)), behavior(candidates[chosen]))
