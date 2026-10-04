"""Stage 1 demonstration NLL; stage 2 branching and intervention supervision."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ..config import Config, seed_all
from ..models import TraceModel
from .data import Records, observation


def supervised(prediction, targets):
    return (F.binary_cross_entropy(prediction[..., 0].clamp(1e-6, 1-1e-6), targets[..., 0])
            + F.huber_loss(prediction[..., 1:], targets[..., 1:]))


def branch_loss(model, batch, cfg, change_weight):
    scene = model.encoder(observation(batch))
    predictions = model.predict(scene, batch["candidates"])
    targets = batch["targets"]
    total = predictions.new_zeros(())
    # Independent bootstrap samples provide member diversity, beyond initialization.
    for m in range(predictions.shape[1]):
        index = torch.randint(len(scene), (len(scene),), device=scene.device)
        total = total + supervised(predictions[index, m], targets[index]) / predictions.shape[1]
    if "intervention_targets" in batch:
        intervention_scene = model.encoder(observation(batch, "intervention_obs_"))
        changed = model.predict(intervention_scene, batch["candidates"])
        target_i = batch["intervention_targets"]
        for m in range(changed.shape[1]):
            total = total + supervised(changed[:, m], target_i)/changed.shape[1]
        differences = (target_i-targets)[:, None].expand_as(changed)
        total = total + change_weight * F.l1_loss(changed-predictions, differences)
    # Refine the generator against measured outcomes, not its own predicted scores.
    eligible = targets[..., 0] <= cfg.risk_threshold
    valid = eligible.any(-1)
    costs = targets[..., 1:] @ targets.new_tensor(cfg.weights)
    if valid.any():
        logits = (-costs[valid]/cfg.generator_temperature).masked_fill(~eligible[valid], -torch.inf)
        weights = logits.softmax(-1).detach()
        total = total + (weights * model.generator.nll(scene[valid], batch["candidates"][valid])).sum(-1).mean()
    if "demo" not in batch:
        raise ValueError("Stage 2 requires a demonstration replay sequence in every record")
    total = total + 0.1 * model.generator.nll(scene, batch["demo"][:, None]).mean()
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=["init", "consequence"], required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--change-weight", type=float)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    seed_all(cfg.seed)
    model = TraceModel(cfg).to(cfg.device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
    elif args.stage == "consequence":
        parser.error("Stage 2 requires --resume with a stage-1 checkpoint")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    data = Records(args.manifest, "train", cfg)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    weight = cfg.change_weight if args.change_weight is None else args.change_weight
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            if args.stage == "init":
                scene = model.encoder(observation(batch))
                loss = model.generator.nll(scene, batch["demo"][:, None]).mean()
            else:
                loss = branch_loss(model, batch, cfg, weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; check observations and cost normalization")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
        print(f"epoch={epoch+1} loss={epoch_loss/max(1,len(loader)):.6f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.dictionary(), "stage": args.stage}, output)


if __name__ == "__main__":
    main()
