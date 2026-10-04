"""Disjoint isotonic fit and source-cluster bootstrap rule selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from ..config import Config
from ..controller import file_hash
from ..models import TraceModel
from .data import Records, observation


@torch.no_grad()
def predict(model, records, device):
    means, stds, labels, groups = [], [], [], []
    for i in range(len(records)):
        sample = {k: v[None].to(device) for k, v in records[i].items()}
        scene = model.encoder(observation(sample))
        p = model.predict(scene, sample["candidates"])[0, ..., 0].cpu().numpy()
        means.extend(p.mean(0)); stds.extend(p.std(0))
        labels.extend(sample["targets"][0, :, 0].cpu().numpy())
        groups.extend([records.entries[i]["group_id"]] * p.shape[1])
    return tuple(np.asarray(x) for x in (means, stds, labels, groups))


def upper_frequency(labels, accepted, groups, rng, repeats=2000):
    unique = np.unique(groups)
    if len(unique) < 2 or not accepted.any():
        return 1.0
    numerator = np.array([labels[(groups == g) & accepted].sum() for g in unique])
    denominator = np.array([np.sum((groups == g) & accepted) for g in unique])
    draws = rng.integers(0, len(unique), (repeats, len(unique)))
    n, d = numerator[draws].sum(1), denominator[draws].sum(1)
    rates = np.divide(n, d, out=np.ones_like(n, dtype=float), where=d > 0)
    return float(np.quantile(rates, 0.95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**ckpt["config"])
    model = TraceModel(cfg).eval()
    model.load_state_dict(ckpt["model"])
    fit = predict(model, Records(args.manifest, "calibration_fit", cfg), "cpu")
    selection = predict(model, Records(args.manifest, "calibration_select", cfg), "cpu")
    iso = IsotonicRegression(out_of_bounds="clip").fit(fit[0], fit[2])
    mean, std, labels, groups = selection
    choices = []
    for beta in [0.0, 0.5, 1.0, 2.0, 3.0]:
        accepted = np.clip(iso.predict(mean)+beta*std, 0, 1) <= cfg.risk_threshold
        upper = upper_frequency(labels, accepted, groups, np.random.default_rng(cfg.seed))
        if accepted.any() and upper <= cfg.risk_threshold:
            choices.append((accepted.mean(), -beta, beta))
    if not choices:
        raise RuntimeError("Calibration failed: no rule meets the empirical selection criterion")
    result = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
              "beta": max(choices)[2], "threshold": cfg.risk_threshold,
              "model_sha256": file_hash(args.checkpoint)}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
