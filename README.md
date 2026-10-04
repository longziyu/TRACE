# TRACE

**Grounding Navigation Actions and Explanations in Predicted Consequences**

This repository contains the TRACE navigation implementation, including training, consequence calibration, asynchronous explanations, simulation interfaces, and ROS 2 deployment nodes.

## Prerequisites

- Python 3.10 or newer.
- A PyTorch installation compatible with the target CPU or GPU.
- CARLA server and matching Python client for CARLA evaluation.
- ROS 2, `rclpy`, `cv_bridge`, OpenCV, and standard sensor/navigation message packages for robot deployment.

Datasets, trained checkpoints, and language-model weights are not included. The provided source has undergone static syntax checks; end-to-end execution and reproduction of the reported experimental results have not been validated in this release.

All commands below use Bash syntax and run from the repository root. In PowerShell, execute multiline commands on one line or use PowerShell line continuation.

## 1. Install the package

```bash
git clone https://github.com/longziyu/TRACE.git
cd TRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Optional: language-model training and generation
python -m pip install -e ".[language]"
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. CARLA and ROS are installed separately. Dependency requirements are listed in `pyproject.toml`; a tested environment lockfile is not provided.

## 2. Prepare data and models

Use the following directory layout:

```text
data/
  c2n/processed/
    manifest.jsonl
    *.npz
  language/qwen/train.jsonl
  language/llama/train.jsonl
models/
  Qwen2.5-7B-Instruct/
  Llama-3.1-8B-Instruct/
checkpoints/
runs/
```

Convert navigation data into the NPZ format described in [docs/data_schema.md](docs/data_schema.md). Each sample contains observation history and a demonstration sequence; consequence-training samples also contain candidate sequences, measured outcomes, and optional intervention pairs.

The JSONL manifest assigns each sample to a source group and a split:

```json
{"file":"scene_001.npz","group_id":"source_scene_001","split":"train","simulator":"carla"}
```

Supported workflow splits are `train`, `validation`, `calibration_fit`, `calibration_select`, and `test`. Keep all related trajectories, branches, and intervention variants within one split. NPZ paths are relative to the manifest directory.

C2N-Bench assets and download links are not included in this release. A benchmark-specific exporter is needed to convert existing scenes and rollout records into this format. The reference simulator in the next section provides an independent data-generation workflow.

For learned explanations, place the instruction-model files and tokenizers in the corresponding `models/` directories. Supported configurations include `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`. Model access is subject to their respective licenses. Loading is local-only and does not download files automatically.

## 3. Generate reference simulation data

The included 2-D simulator supports saved-state branching and scene interventions. It uses synthetic sensor representations, a fixed obstacle layout, and deterministic pedestrian motion. It is a development environment, not C2N-Bench or a substitute for its experimental protocol.

```bash
trace-collect --config configs/default.yaml --mode demos \
  --output data/reference_demos --scenes 30 --states 40
```

This creates demonstration NPZ files and `data/reference_demos/manifest.jsonl`. Collection rounds require separate output directories.

## 4. Train the navigation model

### Stage 1: candidate initialization

```bash
trace-train --config configs/default.yaml \
  --manifest data/reference_demos/manifest.jsonl \
  --stage init --epochs 20 --output checkpoints/initialization.pt
```

### Collect candidate outcomes and intervention pairs

```bash
trace-collect --config configs/default.yaml --mode branches \
  --checkpoint checkpoints/initialization.pt \
  --output data/reference_round01 --scenes 30 --states 40 \
  --repetitions 20 --intervention reverse_pedestrian
```

Available interventions are `reverse_pedestrian` and `block_passage`. Candidate generation remains frozen during collection. Repeated deterministic rollouts do not estimate stochastic pedestrian uncertainty; stochastic experiments require a corresponding simulator and matched random seeds.

### Stage 2: consequence learning and generator refinement

```bash
trace-train --config configs/default.yaml \
  --manifest data/reference_round01/manifest.jsonl \
  --stage consequence --resume checkpoints/initialization.pt \
  --epochs 20 --output checkpoints/navigation.pt
```

For additional rounds, collect new branches using the updated checkpoint and resume stage 2 on the new manifest. Preserve source-group split assignments across rounds. Model selection on validation data is a separate step.

Configuration values are defined in `configs/default.yaml`. Observation-history length, horizon, candidate count, and normalization must match the dataset. To disable the intervention-difference loss on the same data, add `--change-weight 0` to stage 2.

## 5. Calibrate collision-risk screening

```bash
trace-calibrate --checkpoint checkpoints/navigation.pt \
  --manifest data/reference_round01/manifest.jsonl \
  --output checkpoints/calibration.json
```

Calibration uses separate `calibration_fit` and `calibration_select` partitions. It fails explicitly if no candidate screening rule meets the empirical selection criterion. Deployment requires a stage-2 checkpoint and a calibration artifact matching its hash. The screening rule is empirical and does not certify collision-free behavior.

## 6. Evaluate in the reference simulator

```bash
trace-sim --checkpoint checkpoints/navigation.pt \
  --calibration checkpoints/calibration.json \
  --episodes 10 --seed 1000 --output runs/reference_eval01

trace-evaluate --episodes runs/reference_eval01/episodes.jsonl \
  --output runs/reference_eval01/metrics.json
```

Outputs include per-episode decision logs, explanation status, control-cycle duration, episode summaries, and navigation metrics. Reported metrics are success rate, collision rate, timeout rate, and average completion time over successful episodes. Completion time is `null` when no episode succeeds.

Use a new output directory for each evaluation run. The reference simulator advances once per control step; its simulated completion time excludes computation time. `--realtime` paces wall-clock execution but does not make the world advance during inference. Latency studies require a continuously advancing simulator or physical robot.

## 7. Train and enable the interpreter

Prepare verified decision-record examples according to the [language data schema](docs/data_schema.md). Stage 3 updates the language adapter without updating the navigation model.

```bash
trace-language-train --model models/Qwen2.5-7B-Instruct \
  --records data/language/qwen/train.jsonl \
  --output checkpoints/qwen_adapter

# Alternative backbone
trace-language-train --model models/Llama-3.1-8B-Instruct \
  --records data/language/llama/train.jsonl \
  --output checkpoints/llama_adapter
```

Enable a local model and trained adapter during simulation:

```bash
trace-sim --checkpoint checkpoints/navigation.pt \
  --calibration checkpoints/calibration.json \
  --language-model models/Qwen2.5-7B-Instruct \
  --language-adapter checkpoints/qwen_adapter \
  --output runs/language_eval01
```

Without `--language-model`, the runner uses deterministic explanations from the decision record. The learned backend produces structured claims that are checked and rendered into text. Invalid output falls back to a template. The checker does not validate arbitrary free-form prose.

The asynchronous worker uses bounded queues. Pending requests can be replaced, but an active model call is not forcibly interrupted. Queue cancellation therefore does not imply that inference computation was canceled. Results whose premises no longer hold are marked historical.

## 8. Run with CARLA

Start an asynchronously advancing CARLA server and install its matching Python client. Initialize the scene externally, identify an existing `walker.pedestrian.*` ego actor, and stop its AI controller before connecting TRACE.

```bash
export CARLA_HOST=127.0.0.1
export CARLA_PORT=2000
export TRACE_CARLA_EGO_ID=123
export TRACE_CARLA_GOAL=25.0,10.0

trace-sim --checkpoint checkpoints/carla_navigation.pt \
  --calibration checkpoints/carla_calibration.json \
  --adapter trace_nav.simulation.carla_adapter:create \
  --episodes 1 --output runs/carla_episode01
```

Actor ID and goal coordinates must match the initialized scene. Reset the scene before each episode. The adapter attaches RGB, semantic LiDAR, and collision sensors, then releases those sensors on shutdown.

The walker is a surrogate for velocity-command integration; it does not reproduce differential-drive dynamics. Semantic LiDAR also differs from real-world pedestrian detection. Benchmark-specific actuation, perception, and deterministic branch reconstruction require additional integration. The included adapter does not supply a C2N-Bench scenario loader or exact CARLA snapshot restoration.

Other simulator wrappers can implement `observe`, `execute`, `timestamp`, and `close`, then be loaded with `--adapter module:factory`. The interface is defined in `src/trace_nav/simulation/adapters.py`.

## 9. Run the ROS 2 nodes

Source the ROS and robot workspaces, install the package into the compatible Python environment, and configure absolute checkpoint paths and topic names in `configs/ros.yaml`. Robot drivers, localization, tracking, coordinate transforms, and the command multiplexer are external components.

The package uses Python entry points and does not require `colcon build`.

```bash
# Sensor bridge
python -m trace_nav.real_world.sensor_bridge --ros-args --params-file configs/ros.yaml

# Controller, in a separate terminal; motion is disabled by default
trace-ros --ros-args --params-file configs/ros.yaml
```

Default topics:

| Topic | Message type | Purpose |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB input |
| `/scan` | `sensor_msgs/LaserScan` | Range input |
| `/odom` | `nav_msgs/Odometry` | Robot pose and motion |
| `/trace/goal` | `geometry_msgs/PoseStamped` | Navigation goal |
| `/trace/people` | `geometry_msgs/PoseArray` | Pedestrian positions |
| `/trace/observation` | `std_msgs/String` | Serialized observation |
| `/trace/cmd_vel` | `geometry_msgs/Twist` | Output command |
| `/trace/decision` | `std_msgs/String` | Decisions and execution overrides |
| `/trace/explanation` | `std_msgs/String` | Explanations and validity flags |

Goals and pedestrian positions must already be expressed in the configured odometry frame. The bridge assumes aligned planar LiDAR coordinates; sensor extrinsics require an appropriate transform adapter. Fresh pedestrian messages are required by default, including empty messages when no pedestrians are detected. PoseArray does not supply track velocities, so those feature slots remain zero and motion information relies on history. Training data must match the deployed observation convention.

Inspect outputs and connect `/trace/cmd_vel` through the robot's command mux before enabling motion. Maintain an independent driver timeout and physical stop mechanism.

```bash
ros2 topic echo /trace/decision
ros2 topic echo /trace/explanation

# Enable or disable motion
ros2 param set /trace_controller armed true
ros2 param set /trace_controller armed false
```

The controller stops on stale input, deadline misses, goal arrival, laser-stop conditions, or disarming. Policy selection and the actual command are recorded separately. If a downstream driver modifies the command, configure `executed_command_topic` to reference that actual command stream.

Optional `language_model` and `language_adapter` parameters select local model directories; an empty model path uses template explanations. A reference-raster checkpoint is not suitable for direct robot deployment without sensor-compatible training and calibration.

## 10. Checks and reproducibility records

A dependency-free source check is available:

```bash
python tools/static_check.py
```

This parses Python files without importing runtime dependencies or executing training, simulation, model inference, or ROS. It does not replace environment-specific integration tests.

For each experiment, retain the source revision, dataset provenance, split manifest, independent training seeds, checkpoint and calibration artifacts, model revisions, simulator/client versions, hardware, precision, and actual command limits. Confidence intervals require source-group and training-seed information rather than treating correlated branches as independent observations.

Generic IL/RL baselines, human annotation, unrestricted text evaluation, and full experimental sweeps are not included. Further details are available in [docs/data_schema.md](docs/data_schema.md) and [docs/implementation_status.md](docs/implementation_status.md).
