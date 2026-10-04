"""Closed-loop reference simulation or a user-provided benchmark adapter."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ..controller import Controller
from ..explanation import AsyncExplainer, HFClaimsBackend, is_current
from .world import World
from .adapters import load_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--adapter", help="module:factory returning an external adapter")
    parser.add_argument("--language-model", help="Optional local model directory")
    parser.add_argument("--language-adapter")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    controller = Controller(args.checkpoint, args.calibration, args.device)
    cfg = controller.cfg
    backend = HFClaimsBackend(args.language_model, adapter=args.language_adapter) if args.language_model else None
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if (root/"episodes.jsonl").exists():
        raise FileExistsError("Use a new run directory to prevent mixing evaluation runs")
    # External adapters should execute their own reset and assign episode-specific seeds.
    for episode in range(args.episodes):
        controller.history.clear()
        world = None if args.adapter else World(cfg, args.seed+episode)
        adapter = load_adapter(args.adapter, {"config": cfg, "seed": args.seed+episode}) if args.adapter else None
        explainer = AsyncExplainer(backend)
        status, cycle = "timeout", 0
        last_submitted = None
        started = time.monotonic()
        simulation_started = adapter.timestamp() if adapter else world.time
        try:
            with (root/f"episode_{episode:05d}.jsonl").open("w", encoding="utf-8") as stream:
                while (adapter.timestamp() if adapter else world.time)-simulation_started < cfg.episode_seconds:
                    before = time.monotonic()
                    frame = adapter.observe() if adapter else world.observe()
                    timestamp = adapter.timestamp() if adapter else world.time
                    record = controller.step(frame, timestamp)
                    command = np.asarray(record.candidates[record.selected])[0]
                    terminal = adapter.execute(command) if adapter else world.step(command)
                    control_ms = (time.monotonic()-before)*1000
                    signature = (record.behavior, record.no_admissible, record.admissible[record.comparator],
                                 int(np.sign(sum(record.contributions))))
                    if signature != last_submitted or cycle % max(1, int(1/cfg.dt)) == 0:
                        explainer.submit(record); last_submitted = signature
                    explanation = explainer.poll()
                    text = None
                    if explanation:
                        valid = is_current(explanation, record, controller, frame["previous_action"], timestamp)
                        text = {"text": explanation.text, "historical": not valid, "fallback": explanation.fallback,
                                "latency_ms": 1000*(explanation.completed_at-explanation.submitted_at)}
                    stream.write(json.dumps({"decision": record.dictionary(), "control_ms": control_ms,
                                             "explanation": text, "terminal": terminal})+"\n")
                    cycle += 1
                    if terminal["collision"]:
                        status = "collision"; break
                    if terminal["goal"]:
                        status = "success"; break
                    if terminal["timeout"]:
                        break
                    if args.realtime:
                        time.sleep(max(0, cfg.dt-(time.monotonic()-before)))
        finally:
            explainer.close()
            if adapter:
                adapter.close()
        elapsed_simulation = (adapter.timestamp() if adapter else world.time)-simulation_started
        summary = {"episode": episode, "status": status, "simulation_seconds": elapsed_simulation,
                   "wall_seconds": time.monotonic()-started, "requests": explainer.submitted,
                   "canceled": explainer.canceled, "simulator": args.adapter or "reference2d"}
        with (root/"episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary)+"\n")


if __name__ == "__main__":
    main()
