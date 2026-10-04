"""Generate demonstration/branch records from the reference simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..config import Config, seed_all
from ..models import TraceModel
from ..observations import History, tensors
from .world import World, outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["demos", "branches"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--intervention", choices=["reverse_pedestrian", "block_passage"], default="reverse_pedestrian")
    args = parser.parse_args()
    cfg = Config.load(args.config); seed_all(cfg.seed)
    if args.scenes < 10 or min(args.states, args.repetitions) < 1:
        parser.error("Use at least 10 scenes and positive states/repetitions")
    model = None
    if args.mode == "branches":
        if not args.checkpoint:
            parser.error("Branch collection requires a frozen --checkpoint")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = Config(**ckpt["config"]); cfg.device = "cpu"
        model = TraceModel(cfg).eval(); model.load_state_dict(ckpt["model"])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/"manifest.jsonl"
    if manifest.exists():
        raise FileExistsError("Use a new output directory for each collection round")
    entries = []
    split_names = ["train"]*6 + ["validation", "calibration_fit", "calibration_select", "test"]
    for scene in range(args.scenes):
        world = World(cfg, seed=scene+args.seed_offset)
        history = History(cfg.history)
        for state_index in range(args.states):
            obs = history.push(world.observe())
            demo = world.expert_sequence()
            record = {"obs_"+k: v for k, v in obs.items()}
            record["demo"] = demo
            saved = world.snapshot()
            if model is not None:
                with torch.no_grad():
                    inp = tensors(obs, "cpu")
                    scene_encoding = model.encoder(inp)
                    candidates = model.generator.candidates(scene_encoding, inp["previous_action"][:, -1], sample=True)[0].numpy()
                world.intervene(args.intervention)
                if not world.valid_initial_state():
                    world.restore(saved)
                    terminal = world.step(demo[0])
                    if any(terminal.values()):
                        break
                    continue
                changed_state = world.snapshot()
                # Patch only the current observation; prior history precedes the intervention.
                obs_i = {k: v.copy() for k, v in obs.items()}
                changed_frame = world.observe()
                for k in obs_i:
                    obs_i[k][-1] = changed_frame[k]
                record.update({"intervention_obs_"+k: v for k, v in obs_i.items()})
                targets = [[], []]
                for branch_state, bucket in [(saved, targets[0]), (changed_state, targets[1])]:
                    for candidate in candidates:
                        repeats = []
                        for repeat in range(args.repetitions):
                            world.restore(branch_state)
                            world.reseed(cfg.seed+scene*10000+state_index*100+repeat)
                            repeats.append(outcome(world, candidate))
                        bucket.append(np.mean(repeats, axis=0))
                record.update(candidates=candidates, targets=np.asarray(targets[0]),
                              intervention_targets=np.asarray(targets[1]))
            world.restore(saved)
            name = f"scene_{scene:05d}_state_{state_index:05d}.npz"
            np.savez_compressed(root/name, **record)
            entries.append({"file": name, "group_id": f"reference_scene_{scene+args.seed_offset}",
                            "split": split_names[scene % 10], "simulator": "reference2d"})
            terminal = world.step(demo[0])
            if any(terminal.values()):
                break
    manifest.write_text("\n".join(json.dumps(e) for e in entries)+"\n", encoding="utf-8")
    (root/"collection.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
