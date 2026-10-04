"""Manifest-based NPZ data with source-scene split isolation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..observations import GROUP_DIMS


class Records(Dataset):
    def __init__(self, manifest, split, config=None):
        self.config = config
        self.root = Path(manifest).resolve().parent
        entries = [json.loads(x) for x in Path(manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
        groups = {}
        for entry in entries:
            # group_id must include all map/scene descendants required by the split policy.
            group = entry["group_id"]
            if group in groups and groups[group] != entry["split"]:
                raise ValueError(f"Source-group leakage: {group}")
            groups[group] = entry["split"]
        self.entries = [e for e in entries if e["split"] == split]
        if not self.entries:
            raise ValueError(f"Empty split {split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path = (self.root / self.entries[index]["file"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("NPZ path must stay inside the manifest directory")
        with np.load(path, allow_pickle=False) as data:
            result = {k: torch.from_numpy(np.array(data[k], dtype=np.float32)) for k in data.files}
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise ValueError(f"Nonfinite data in {path}")
        self.validate(result, path)
        return result

    def validate(self, row, path):
        if "obs_rgb" not in row or row["obs_rgb"].ndim != 4:
            raise ValueError(f"{path}: obs_rgb must have shape [L,3,64,64]")
        length = row["obs_rgb"].shape[0]
        shapes = {"rgb": (length, 3, 64, 64), "bev": (length, 2, 64, 64),
                  "previous_action": (length, 2),
                  **{key: (length, dim) for key, dim in GROUP_DIMS.items()}}
        paired = "intervention_targets" in row
        prefixes = ["obs_"] + (["intervention_obs_"] if paired else [])
        for prefix in prefixes:
            for key, shape in shapes.items():
                if prefix+key not in row or tuple(row[prefix+key].shape) != shape:
                    raise ValueError(f"{path}: {prefix+key} must have shape {shape}")
            image = row[prefix+"rgb"]
            if image.min() < 0 or image.max() > 1:
                raise ValueError(f"{path}: RGB must be normalized to [0,1]")
        if "demo" not in row or row["demo"].ndim != 2 or row["demo"].shape[-1] != 2:
            raise ValueError(f"{path}: demo must have shape [H,2]")
        horizon = row["demo"].shape[0]
        if self.config and (length != self.config.history or horizon != self.config.horizon):
            raise ValueError(f"{path}: L/H do not match the configuration")
        if "candidates" in row:
            candidates = row["candidates"]
            if candidates.ndim != 3 or tuple(candidates.shape[-2:]) != (horizon, 2):
                raise ValueError(f"{path}: candidates must have shape [K,H,2]")
            if self.config and len(candidates) != self.config.learned_candidates+4:
                raise ValueError(f"{path}: candidate count does not match the configuration")
            for key in ["targets"] + (["intervention_targets"] if paired else []):
                target = row.get(key)
                if target is None or tuple(target.shape) != (len(candidates), 4):
                    raise ValueError(f"{path}: {key} must have shape [K,4]")
                if (target < 0).any() or (target[:, 0] > 1).any():
                    raise ValueError(f"{path}: outcomes must be nonnegative; collision targets at most one")
        elif paired:
            raise ValueError(f"{path}: intervention targets require candidates")


def observation(batch, prefix="obs_"):
    return {k[len(prefix):]: value for k, value in batch.items() if k.startswith(prefix)}
