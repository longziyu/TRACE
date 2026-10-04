# Implementation status

## Delivered

- Shared PyTorch navigation encoder, projected mixture candidates, consequence
  ensemble, empirical risk calibration, selector, and decision records.
- Demonstration initialization, branch/intervention training, measured-outcome
  generator refinement with replay, and local-model LoRA language adaptation.
- Checked claims, deterministic fallback, bounded asynchronous queues, and
  reevaluation of cited candidate sequences for temporal validity.
- Restorable reference simulator, paired-data collection, closed-loop logging,
  a live CARLA sensor adapter, and an external simulator plugin boundary.
- ROS 2 sensor bridge and controller, explicit arming, separate watchdog,
  sensor-age checks, goal stop, laser stop, and override-aware logging.
- English README, configuration files, data contracts, and console entry points.

## Checks performed

- Parsed all 23 Python files with the standard-library AST parser.
- Verified that all seven configured console entry points resolve to source files
  containing their declared top-level functions, without importing those modules.
- Inspected data dimensions, configuration matching, stage/checkpoint guards,
  calibration hashing, and the consistency of documented commands with entry points.
- Confirmed that comments and documentation contain no Chinese text.

These are static checks only. No dependency installation, training, simulator,
ROS node, model inference, or end-to-end test was executed. Runtime correctness,
hardware deadlines, learned performance, and API compatibility remain untested.

## Integration work requiring external assets

- Provide C2N-Bench scenes and source provenance, plus a benchmark-specific
  deterministic branch reconstruction/exporter.
- Replace the reference raster observations and CARLA walker surrogate with the
  actual sensor and robot dynamics used in the manuscript's experiments.
- Supply real sensor transforms, localization, pedestrian tracking, and a command
  mux/driver timeout. Adapt the simplified position-only pedestrian interface if
  velocity tracks are required.
- Train navigation and language models and fit held-out calibration. No learned
  weights or experimental numerical outputs are bundled.
- Add research-specific offline annotation and evaluation for unrestricted
  natural-language explanations. The checked runtime only renders verified slots.
- Run hardware/environment compatibility checks later, in the user's environment.
