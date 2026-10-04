"""Shared inference controller; requires a trained model and matching calibration."""
import hashlib
from pathlib import Path
import torch
from .config import Config
from .models import TraceModel
from .observations import History, tensors
from .decision import Calibration, select


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class Controller:
    def __init__(self, checkpoint, calibration, device="cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("stage") != "consequence":
            raise ValueError("Deployment requires a stage-2 consequence checkpoint")
        self.cfg = Config(**payload["config"])
        self.cfg.device = device
        self.model = TraceModel(self.cfg).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.calibration = Calibration.load(calibration)
        if self.calibration.model_sha256 != file_hash(checkpoint):
            raise ValueError("Calibration belongs to a different checkpoint")
        self.history = History(self.cfg.history)
        self.current_scene = None

    @torch.no_grad()
    def step(self, frame, timestamp):
        obs = tensors(self.history.push(frame), self.cfg.device)
        scene = self.model.encoder(obs)
        self.current_scene = scene
        candidates = self.model.generator.candidates(scene, obs["previous_action"][:, -1])
        prediction = self.model.predict(scene, candidates)
        return select(candidates[0].cpu().numpy(), prediction[0].cpu().numpy(),
                      self.calibration, self.cfg, timestamp)

    @torch.no_grad()
    def reevaluate(self, sequences, previous, timestamp):
        """Reevaluate cited sequences, not freshly assigned candidate IDs."""
        import numpy as np
        from .models import project_sequences
        if self.current_scene is None:
            return None
        values = torch.as_tensor(np.asarray(sequences), dtype=torch.float32, device=self.cfg.device)[None]
        prev = torch.as_tensor(previous, dtype=torch.float32, device=self.cfg.device)[None]
        projected = project_sequences(values, prev, self.cfg)
        if not torch.allclose(values, projected, atol=1e-4):
            return None
        pred = self.model.predict(self.current_scene, values)[0].cpu().numpy()
        mean = pred.mean(0)
        risk = np.clip(self.calibration.apply(mean[:, 0]) + self.calibration.beta*pred[..., 0].std(0), 0, 1)
        return risk <= self.calibration.threshold, mean[:, 1:] @ np.asarray(self.cfg.weights)
