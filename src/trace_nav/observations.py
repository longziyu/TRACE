"""Sensor-only tensor contract shared by simulation and robot adapters."""
from collections import deque
import numpy as np
import torch

# Fixed-size groups avoid variable pedestrian counts in the learning interface.
GROUP_DIMS = {"route": 4, "obstacles": 16, "pedestrians": 40, "robot": 4}


def validate(frame):
    expected = {"rgb": (3, 64, 64), "bev": (2, 64, 64), **{k: (v,) for k, v in GROUP_DIMS.items()},
                "previous_action": (2,)}
    for key, shape in expected.items():
        value = np.asarray(frame[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid observation {key}: expected finite {shape}, got {value.shape}")
    return {k: np.asarray(frame[k], dtype=np.float32).copy() for k in expected}


class History:
    def __init__(self, length):
        self.frames = deque(maxlen=length)
        self.length = length

    def push(self, frame):
        frame = validate(frame)
        if not self.frames:
            self.frames.extend([frame] * self.length)
        else:
            self.frames.append(frame)
        return {key: np.stack([x[key] for x in self.frames]) for key in frame}

    def clear(self):
        self.frames.clear()


def tensors(history, device):
    return {key: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0)
            for key, value in history.items()}
