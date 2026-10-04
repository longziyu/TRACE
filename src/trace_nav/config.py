"""Configuration and reproducible initialization."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import random
import numpy as np
import torch
import yaml


@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    dt: float = 0.1
    history: int = 8
    horizon: int = 20
    learned_candidates: int = 8
    ensemble: int = 5
    hidden: int = 256
    max_v: float = 0.5
    max_w: float = 1.0
    max_a: float = 0.5
    max_alpha: float = 1.5
    robot_radius: float = 0.18
    proximity_radius: float = 0.6
    goal_scale: float = 20.0
    distance_cap: float = 40.0
    weights: list = field(default_factory=lambda: [0.6, 0.3, 0.1])
    risk_threshold: float = 0.1
    change_weight: float = 1.0
    generator_temperature: float = 0.1
    goal_tolerance: float = 0.3
    episode_seconds: float = 240.0
    explanation_max_age: float = 2.0
    explanation_action_tolerance: float = 0.08
    language_model: str = "Qwen/Qwen2.5-7B-Instruct"
    language_revision: str | None = None

    def __post_init__(self):
        for key in ("dt", "history", "horizon", "learned_candidates", "ensemble", "hidden",
                    "max_v", "max_w", "max_a", "max_alpha", "goal_scale", "proximity_radius",
                    "distance_cap", "generator_temperature", "episode_seconds"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.hidden % 8 or self.ensemble < 2:
            raise ValueError("hidden must be divisible by 8; ensemble must be at least 2")
        if len(self.weights) != 3 or min(self.weights) < 0 or sum(self.weights) <= 0:
            raise ValueError("weights must contain three nonnegative values with positive sum")
        if not 0 < self.risk_threshold < 1:
            raise ValueError("risk_threshold must be in (0, 1)")

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**yaml.safe_load(handle))

    def dictionary(self):
        return asdict(self)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
