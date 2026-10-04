# TRACE navigation: simulation and ROS 2 reference implementation

This directory implements the method described in `../latex/main.tex`: temporal
scene encoding, candidate generation, action-conditioned consequence prediction,
branch/intervention supervision, explicit risk screening, decision records, and
asynchronous checked explanations. It contains a shared learning/control package,
a reference simulation workflow, a live CARLA adapter, and ROS 2 robot nodes.

**Implementation status:** the code has been written and statically reviewed.
Training, dependency installation, model inference, CARLA, and ROS have not been
run as part of this delivery. No checkpoints or experimental results are bundled.
This is a reference implementation of the design, not a recovered implementation
of the paper's reported experiments. No runtime or reproduction claim is made.

## 1. Project layout

```text
code/
  pyproject.toml                 Python package and optional language dependencies
  configs/default.yaml          Core reference hyperparameters
  configs/ros.yaml              ROS node parameters and explicit arming switch
  docs/data_schema.md           Tensor, manifest, language, and ROS data formats
  tools/static_check.py         Dependency-free Python syntax inspection
  src/trace_nav/
    config.py                   Configuration validation and random seeds
    observations.py             Fixed-shape sensor contract and temporal history
    models.py                   Cross-attention encoder, mixture generator, ensemble
    decision.py                 Screening, selection, fallback, decision record
    controller.py               Checkpoint/calibration loading and cited-sequence reevaluation
    explanation.py              Claim verification, rendering, asynchronous worker
    evaluation.py               SR/CR/TR/AT and probability metric utilities
    training/
      data.py                   NPZ manifest loader and source-group leakage check
      train.py                  Stage 1 and stage 2 optimization
      calibrate.py              Isotonic fitting and cluster-bootstrap rule selection
      language.py               Stage 3 LoRA adaptation for local instruction models
    simulation/
      world.py                  Restorable 2-D reference crowd world and outcome targets
      collect.py                Demonstration, branching, and intervention collection
      run.py                    Closed-loop execution and JSONL logging
      adapters.py               External simulator protocol and dynamic adapter loading
      carla_adapter.py          Live sensor-based CARLA walker surrogate
    real_world/
      sensor_bridge.py          RGB/LiDAR/odometry/goal/pedestrian input bridge
      node.py                   ROS controller, watchdog, stop logic, record publication
```

Create these directories when supplying your own assets:

```text
data/
  c2n/raw/                      Original benchmark/scenario assets
  c2n/processed/                Converted NPZ records and manifest.jsonl
  peac_qa/raw/                  Existing state-action-reason examples, if available
  language/qwen/train.jsonl     Verified record-conditioned training examples
  language/llama/train.jsonl    Matched examples for the second backbone
  robot/                       Sensor recordings and domain-adapted training samples
models/
  Qwen2.5-7B-Instruct/          Downloaded model files, tokenizer, configuration
  Llama-3.1-8B-Instruct/         Downloaded licensed model files
checkpoints/
runs/
```

No public download location for C2N-Bench or PEAC-QA was supplied with the paper.
The project does not invent one or automatically fetch those datasets. Put your
existing data in the paths above and convert it to the documented tensor schema.
Old PEAC-QA examples alone are not enough to train a consequence predictor or to
reconstruct candidate comparisons that were never logged.

## 2. Method-to-code mapping

| Paper component | Implementation | Important boundary |
|---|---|---|
| Temporal multimodal `b_t` | `SceneEncoder`, `History` | RGB, BEV, four feature groups, previous commands; two temporal attention layers |
| Candidate generator | `MixtureGenerator` | Eight mixture components by default, one sequence from each component, plus brake/forward/left/right |
| Feasible controls | `project_sequences` | Velocity and acceleration limits; only the first command is executed |
| Four consequences | `TraceModel.predict` | Ensemble outputs collision probability and three nonnegative normalized costs |
| Branch/intervention supervision | `collect.py`, `branch_loss` | Matched initial snapshots and fixed candidate sequences |
| Measured-outcome generator update | `branch_loss` | Empirical admissibility, soft cost weighting, demonstration replay |
| Risk calibration | `calibrate.py` | Disjoint fit/selection partitions; no successful rule means explicit failure |
| Explicit action choice | `select` | Admissible minimum-cost choice; lexicographic risk/cost fallback |
| Shared decision evidence | `Decision` | Candidate sequences, outputs, uncertainty, masks, selected/comparison IDs, contributions |
| Checked language | `expected_claims`, `verify`, `render` | Fixed claim schema; invalid output falls back to a deterministic template |
| Asynchronous generation | `AsyncExplainer` | One pending request, one completed-result slot; replacements counted |
| Temporal validity | `is_current` | Reevaluate cited sequences; invalidate infeasible/mismatched/reversed comparisons |
| Real robot execution | `real_world/node.py` | Publication, arming, input-age checks, watchdog, and independent laser stop |

The margin condition `m_t > 2B` is not used to skip control updates. Ensemble
disagreement is not a certified probability bound. Risk calibration applies to
the checkpoint hash stored in the calibration artifact; mismatched artifacts are
rejected. Initialization-only checkpoints are rejected for deployment.

The generator is a Gaussian sequence-mixture proposal followed by projection to
the feasible control set. Its NLL is evaluated in proposal coordinates; this is
not an exact density of the projected distribution. This explicit approximation
is an implementation choice where the manuscript leaves details open.

## 3. Environment setup (commands for you to run later)

Use Python 3.10 or newer. ROS 2 installations commonly require the system Python
and a matching package environment; do not mix a Windows Python environment with
a Linux ROS installation. A Linux ROS 2 Humble-style environment is the intended
robot integration target, but it has not been tested here.

From the `code` directory, a simulation environment can be prepared with:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell alternative:
# .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Install a PyTorch build matching your compute environment. The package specifies
minimum dependencies rather than claiming an experimentally tested lockfile.
After your first successful setup, save a lockfile or `pip freeze` alongside the
run manifest. CARLA and ROS are optional external runtimes and are deliberately
not installed by the core package.

For language training or the local Hugging Face backend:

```bash
python -m pip install -e ".[language]"
```

Download the model files yourself from the model repositories named in the paper:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

Observe the respective access requirements and licenses. Record the exact model
revision, tokenizer, adapter, and precision. All included model-loading calls use
`local_files_only=True`; they do not silently download a model. The language
backend defaults to CPU unless instantiated with another device. A 7B/8B model
can require substantial memory; no quantization configuration is assumed.

## 4. Data preparation

Read [the complete data schema](docs/data_schema.md) before converting data.
Each NPZ file stores one source-state history and either a demonstration or a
candidate set with outcome targets. A JSONL manifest supplies paths, source-group
identities, and split names. Use the same observation units and channel ordering
in training, simulation, and on the robot.

The loader checks that a declared source group does not cross partitions. It
cannot discover hidden duplicates or map aliases. Keep all branches, intervention
variants, and related trajectories together. A map-unseen claim requires maps to
be disjoint, not merely different trajectory seeds on the same map.

Train/validation/test selection is a research protocol, not an automatic outcome
of the directory name. The supplied training script trains on `train`; it does
not automatically pick a best checkpoint on validation or report confidence
intervals. Run a separate validation campaign before choosing checkpoints.

## 5. Reference simulation: collection and training

The 2-D world is provided to make the data path and branch semantics concrete.
It supplies a synthetic local raster in the RGB slot, idealized range sectors,
visible pedestrian tracks, and a geodesic distance target. It is **not CARLA,
not C2N-Bench, and not photorealistic camera training**. It uses deterministic
pedestrian trajectories and a fixed obstacle layout. Its scene-seed splits are
not map-held-out splits.

Example workflow, from `code/`:

```bash
# Stage 1: collect demonstrations from the geometric local teacher.
trace-collect --config configs/default.yaml --mode demos \
  --output data/reference_demos --scenes 30 --states 40

trace-train --config configs/default.yaml --manifest data/reference_demos/manifest.jsonl \
  --stage init --epochs 20 --output checkpoints/initialization.pt

# Freeze that generator while collecting matched branch outcomes.
trace-collect --config configs/default.yaml --mode branches \
  --checkpoint checkpoints/initialization.pt --output data/reference_round01 \
  --scenes 30 --states 40 --repetitions 20 --intervention reverse_pedestrian

# Stage 2: learn consequences and refine proposals using measured outcomes.
trace-train --config configs/default.yaml --manifest data/reference_round01/manifest.jsonl \
  --stage consequence --resume checkpoints/initialization.pt \
  --epochs 20 --output checkpoints/navigation.pt

# Fit the isotonic map, then select beta on a disjoint calibration partition.
trace-calibrate --checkpoint checkpoints/navigation.pt \
  --manifest data/reference_round01/manifest.jsonl --output checkpoints/calibration.json

# Run the policy with asynchronous deterministic explanations.
trace-sim --checkpoint checkpoints/navigation.pt --calibration checkpoints/calibration.json \
  --episodes 10 --seed 1000 --output runs/reference_eval01

trace-evaluate --episodes runs/reference_eval01/episodes.jsonl \
  --output runs/reference_eval01/metrics.json
```

PowerShell does not use `\` as a line continuation: put each command on one line
or use PowerShell's backtick continuation. The examples above use Bash syntax.

There is no promise that the untested reference defaults produce a passing
calibration rule or useful navigation. If calibration fails, inspect data quality,
prediction error, and coverage; do not bypass it by labelling an uncalibrated
model safe. Collection rounds must use new directories. To alternate training
and data refresh, run branch collection with the newly trained checkpoint and
then resume stage 2 on the new manifest. Keep source split assignments fixed.

`block_passage` is another available intervention. An intervention is applied at
the current observation only; earlier history remains the factual pre-intervention
history. Check that your intervention is physically valid and does not start the
robot in collision. The geometric reference layout is illustrative; custom scenes
need their own intervention validators. A distant change may legitimately have
zero effect within the finite prediction horizon.

The reference environment advances once per controller step. Its simulated
completion time excludes computation time; `--realtime` only paces wall-clock
execution and does not turn it into a simulator that advances during inference.
Do not use reference-world logs as evidence for the asynchronous latency claim.
Use a continuously advancing external simulator or physical robot for that test.

## 6. CARLA live evaluation and C2N-Bench integration

Install and launch the CARLA server separately, and install its matching Python
client in your environment. Server/client versions must match your deployment.
Prepare your pedestrian scenario externally. The included adapter deliberately
does not assume the unavailable C2N-Bench map list, split definition, scenario
loader, or proprietary robot actor.

The supplied `carla_adapter.py` attaches RGB, semantic LiDAR, and collision
sensors to an existing `walker.pedestrian.*` ego actor. Stop that actor's AI
controller before TRACE takes over. Configure the server to advance
asynchronously, including while Python performs inference. On Linux, for example:

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

Replace actor ID and goal with actual values. The adapter owns and destroys only
its attached sensors; it stops but does not destroy the externally owned ego.
Reset the scene/ego/goal before each episode. Do not request repeated episodes on
the same unreset actor and treat them as independent trials.

CARLA's walker is a **surrogate** for velocity-command integration. In particular,
in-place turning and acceleration do not reproduce TurtleBot dynamics. Replace
`execute()` with your benchmark's robot actuator when measuring the paper's
navigation performance. The adapter converts CARLA's left-handed coordinates to
the core's x-forward/y-left convention. Semantic LiDAR supplies observed class/ID
information; it is not an omniscient actor-state query, but it is privileged
semantic sensing relative to a real detector. Report that input difference or
replace it with the same perception system used on the robot. Verify semantic
labels against your installed CARLA version before collecting data.

The adapter requires fresh camera and LiDAR packets and reports collision before
success. It does not implement an exact world snapshot/restore. CARLA actor
transforms alone are insufficient to restore controller states and randomness.
For branch training in C2N-Bench, rebuild each source scenario using your scenario
runner and matched random seeds, execute the fixed candidate sequences, and export
the NPZ schema. The reference world's snapshot mechanism is not used to make
false deterministic-reset claims about CARLA.

To use an existing simulator wrapper, implement:

```python
class MyAdapter:
    def observe(self):
        # Return the single-frame observation dictionary from data_schema.md.
        ...
    def execute(self, command):
        # Apply [v, w], then report booleans collision, goal, and timeout.
        ...
    def timestamp(self):
        # Return simulation time in seconds, monotonically within an episode.
        ...
    def close(self):
        # Stop commands and release only resources owned by this adapter.
        ...

def create(options):
    # options contains a Config object and an episode seed.
    return MyAdapter()
```

Then pass `--adapter your_module:create`. Ensure that the simulator progresses
during inference and that timestamps, episode timeout, and the actual applied
commands reflect that progression.

## 7. Language training, checks, and asynchronous output

The template backend is the default and does not require a language model.
For learned explanations, prepare verified decision-record JSONL and train:

```bash
trace-language-train --model models/Qwen2.5-7B-Instruct \
  --records data/language/qwen/train.jsonl --output checkpoints/qwen_adapter

trace-language-train --model models/Llama-3.1-8B-Instruct \
  --records data/language/llama/train.jsonl --output checkpoints/llama_adapter

trace-sim --checkpoint checkpoints/navigation.pt --calibration checkpoints/calibration.json \
  --language-model models/Qwen2.5-7B-Instruct \
  --language-adapter checkpoints/qwen_adapter --output runs/language_eval01
```

Stage 3 loads only the language model: it cannot update the navigation weights.
Prompts are masked out of the causal loss. An example whose prompt consumes the
entire sequence budget is rejected rather than silently trained with no targets.
The trainer saves a LoRA adapter, not a merged full model. Match examples,
training budgets, and validation criteria within each backbone for ablations.

The checked backend requests an exact JSON claim object and verifies candidate
references, reason category, numeric values, contribution signs via equality,
and the required uncertainty field. It then renders trusted claims into text.
Invalid JSON, extra fields, unsupported causes, or inference failures trigger a
record-derived fallback. **Arbitrary generated natural language is not certified
by this checker.** Natural free-form explanations and their human readability
study need an extended extraction/annotation pipeline; no fake human scores are
generated by this package.

The worker has one waiting request and one waiting result. Replacement counters
measure queue drops; invalidity is represented by `historical=true` at publication.
An already running Transformers call is not forcefully interrupted, and the
daemon thread may continue until it finishes. Thus cancellation counters do not
measure GPU work saved. For hard cancellation or resource isolation, host the
language worker in a separate service/process and retain the same record schema.
The current thread implementation separates waiting but can still contend for
CPU/GPU resources; measure complete loop timing on your hardware.

Temporal validity rechecks the old selected/comparison sequences at the current
scene representation. A changed numeric candidate ID by itself does not invalidate
the explanation. Infeasible sequences, changed admissibility, reversed cost order,
different semantic behavior, excessive age, or mismatched first commands make an
output historical. This conservative implementation may discard useful text;
report coverage and cancellation alongside stale-output rate.

## 8. Real-world ROS 2 deployment

Run this on the robot computer or a ROS-networked host with compatible ROS 2,
PyTorch, `rclpy`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `std_msgs`, `cv_bridge`,
and OpenCV installed. Source your ROS and robot workspaces before using the Python
entry points. For example, use your actual ROS setup path rather than assuming
the example distribution is installed:

```bash
source /opt/ros/humble/setup.bash
source /path/to/robot_ws/install/setup.bash
python -m pip install -e .
```

The package is invoked as Python/console entry points; it is not an ament package
and does not require `colcon build`. A ROS-native launch package can wrap these
entry points later. The TurtleBot driver, SLAM/localization, object detector,
pedestrian tracker, and command multiplexer are external dependencies.

### Input and output topics

| Topic | ROS type | Meaning |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Camera image; converted to RGB |
| `/scan` | `sensor_msgs/LaserScan` | Range sensing and independent near-obstacle stop |
| `/odom` | `nav_msgs/Odometry` | Pose and measured motion |
| `/trace/goal` | `geometry_msgs/PoseStamped` | Goal already transformed into the odometry frame |
| `/trace/people` | `geometry_msgs/PoseArray` | Detected/tracked positions in the same frame |
| `/trace/observation` | `std_msgs/String` | Explicit JSON sensor tensor packet |
| `/trace/cmd_vel` | `geometry_msgs/Twist` | Controller output, before robot-specific mux/driver |
| `/trace/decision` | `std_msgs/String` | Decision evidence, actual command, override reason, latency |
| `/trace/explanation` | `std_msgs/String` | Text, fallback flag, historical flag, override reason |

The bridge requires a fresh people message by default. Publish an **empty but
fresh** PoseArray when the detector sees no pedestrians; do not disable freshness
checks to hide a failed detector. PoseArray has no stable identity or velocity
fields, so the bridge leaves velocity features zero. For the full temporal
feature contract, replace this input with a tracked-object message and estimate
velocities in a common frame. Train with the representation actually deployed.

The bridge does not transform sensor extrinsics with TF. It assumes the laser is
aligned with the robot's x-forward/y-left planar frame and that the goal/people
poses are already in the configured odometry frame. Add your actual TF/extrinsic
transforms before using an offset or rotated sensor. The `/trace/goal` timestamp
does not need continual refresh, but sensor packets do. Record actual driver and
sensor frame names; do not rename headers to conceal frame mismatches.

### Starting the nodes

Edit absolute model paths and topic names in `configs/ros.yaml` first. Use a
checkpoint trained with real-sensor-compatible observations and a corresponding
calibration artifact; the reference raster checkpoint is not suitable for direct
robot transfer.

```bash
# Terminal 1: after camera, LiDAR, localization, and pedestrian perception start
python -m trace_nav.real_world.sensor_bridge --ros-args --params-file configs/ros.yaml

# Terminal 2: computation runs, but motion is disabled by default
trace-ros --ros-args --params-file configs/ros.yaml

# Inspect outputs before connecting the output topic to the robot command mux
ros2 topic echo /trace/decision
ros2 topic echo /trace/explanation

# Enable motion only after verifying your driver/mux, physical E-stop, and frame setup
ros2 param set /trace_controller armed true

# Disable motion
ros2 param set /trace_controller armed false
```

`/trace/cmd_vel` intentionally does not automatically take over `/cmd_vel`. Route
it through your robot's command mux/velocity limiter. If the final actuator command
is modified downstream, set `executed_command_topic` to that actual command stream
so history reflects what was sent to the robot. The sensor bridge keeps measured
odometry velocities separately in `robot` features.

The controller publishes zero commands when disarmed, input is stale, inference
misses the watchdog deadline, the goal is reached, or the laser stop is triggered.
The watchdog uses a separate executor callback group; a blocked inference callback
cannot deliberately hold the last command indefinitely through that callback.
Python scheduling and a failed host still require an independent driver timeout
and physical stop mechanism. Configure those in the robot driver/mux.

Policy selection and the executed command are separate fields. A stop override
must not be explained as if the neural selector chose it. During an override,
pending explanations are marked historical and new policy explanations are not
requested. Record intervention counts when evaluating physical trials.

Optional ROS language parameters `language_model` and `language_adapter` accept
local directories; empty `language_model` uses the template backend. Leave
navigation and language on separate devices/processes if your measured deadline
cannot tolerate contention. Match the simulator's speed, acceleration, sensor
normalization, and safety-shield rules to deployment before interpreting transfer.

## 9. Evaluation, logging, and ablations

`trace-sim` logs decisions and complete observation-to-command wall-clock duration
to per-episode JSONL files. Episode summaries separately retain simulation time
and wall time. `trace-evaluate` reports SR, CR, timeout rate, other outcomes, and
successful-episode AT. AT is `null` when no episode succeeds. Collision has priority
over success. No synthetic result table is generated or copied from the manuscript.

The probability utility computes Brier score and equal-width-bin ECE. Use the
calibrated **mean probability**, not the conservative screening score, for these
metrics. Training data in this compact format contains mean repeated outcomes;
for exact binary-sample Brier score, score the original repeated binary labels.
MAEs, candidate regret, accepted-candidate frequency, and intervention differences
can be calculated from exported predictions and the target arrays. Keep their
denominators and normalization explicit.

The following comparisons are supported by concrete modules or controlled changes:

- Same data without difference supervision: stage 2 with `--change-weight 0`.
- Equal-budget ordinary data: collect an unpaired dataset with the same simulated
  candidate-step budget and training updates; the default collector creates pairs,
  so use a converter/custom collector and omit `intervention_*` keys consistently.
- No history: retrain with `history: 1` and recollect/convert matching observation
  arrays. Do not load an incompatible position-embedding checkpoint.
- No conservative uncertainty term: recalibrate/evaluate a separately recorded
  `beta=0` rule. Do not silently edit a published calibration artifact.
- No risk screening or geometric evaluation: replace the selector in a named
  experimental controller and log that variant. These baselines are not silently
  substituted into the main TRACE controller.
- Explanation access/check/monitor ablations: keep navigation records frozen;
  build restricted prompts or bypass checks only in isolated offline evaluation.
  Preserve generation outputs and annotations for WRR/UCR/coverage scoring.
- Synchronous language delay: implement a separate control scheduling experiment
  in a continuously advancing simulator; the default runner is asynchronous.
  Inject delay into language generation, not by pausing the simulator clock.

Generic RL/IL baseline training, human scoring, automatic free-text claim extraction,
full experiment sweeps, and a benchmark-specific C2N-Bench collector are not
bundled. Their absence is explicit rather than replaced with fabricated scores
or no-op implementations. They require the corresponding baseline repositories,
annotations, and scenario definitions.

Store independent training seeds, scene provenance, calibration partition counts,
command limits, simulator/client versions, model hashes, model revisions, hardware,
precision, and exact source revision with every run. Bootstrap source groups and
training seeds rather than treating correlated branches as independent trials.
The empirical calibration selector has limited reliability with few groups or
zero observed collisions; its bootstrap bound is not a safety certificate.

## 10. Static checking and current limitations

The only check intended without installing runtime dependencies is:

```bash
python tools/static_check.py
```

This parses Python source with `ast.parse`; it does not import PyTorch, ROS,
Transformers, or CARLA, launch anything, download files, or execute a model.
Syntax checks cannot prove tensor-shape correctness for arbitrary external data,
ROS topic compatibility, simulator API compatibility, runtime memory fit, or robot
behavior. Those must be checked later in your actual environment.

Before a first runtime attempt, provide:

1. Actual dataset/scenario assets and a provenance-preserving manifest.
2. A stage-2 checkpoint and matching successful calibration artifact.
3. Local language model files if using learned explanations.
4. A running simulator with the expected sensor/actuator setup, or ROS topics,
   localization, tracking, frame transforms, and the robot command mux.
5. Hardware-specific dependency versions and configuration values.

The repository deliberately leaves unresolved scientific choices visible in this
README. Completing these integration steps is different from merely fixing Python
syntax, and matching the paper's numerical tables requires actual experiments.
