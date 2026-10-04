# TRACE data contracts

All numeric learning arrays are float32. Coordinates use meters, seconds, and
radians. Core commands are `[linear_speed, angular_speed]`, with positive angular
speed turning left, no reverse motion, and a fixed configured control interval.
Array axes below omit the training batch dimension.

## Observation history

| NPZ key | Shape | Definition |
|---|---|---|
| `obs_rgb` | `[L,3,64,64]` | RGB in `[0,1]`; channel-first |
| `obs_bev` | `[L,2,64,64]` | Local obstacle and observed pedestrian layers |
| `obs_route` | `[L,4]` | Local goal x/scale, y/scale, Euclidean distance/scale, bearing/pi |
| `obs_obstacles` | `[L,16]` | Minimum range in each sector, clipped to sensor range and divided by it |
| `obs_pedestrians` | `[L,40]` | Eight nearest visible tracks, each `[x/range,y/range,vx,vy,present]` |
| `obs_robot` | `[L,4]` | Measured linear speed, measured angular speed, sin(yaw), cos(yaw) |
| `obs_previous_action` | `[L,2]` | Executed command preceding each observation |
| `demo` | `[H,2]` | Feasible demonstration sequence for initialization/replay |

The local frame is x-forward, y-left. The local raster maps x to columns and y
to rows, with the robot at `(32,32)` and a four-meter half-width. This indexing is
a tensor convention, not a conventional image display orientation. The sixteen
sectors start at -pi and increase counterclockwise. Missing pedestrian slots are
zeros and have `present=0`. Do not pad missing whole sensor messages with zeros;
stop or reject observations instead. At episode start, the first frame is
repeated L times. If timestamps jump backward, reset the history.

The reference simulator supplies idealized visible-track velocities. The included
PoseArray ROS bridge and semantic-LiDAR CARLA adapter supply zero velocity slots
and depend on position history; train with this same convention or implement a
track-velocity adapter before transfer. Zero velocity here means unavailable,
not verified stationary motion; the present bit does not encode velocity validity.

Route features above are observable goal features, not the privileged geodesic
outcome target. A production route encoder can additionally use a sensor-derived
map, but changing feature dimensions requires a new model configuration and training.

## Branch and intervention fields

| Key | Shape | Definition |
|---|---|---|
| `candidates` | `[K,H,2]` | Executable command sequences, same ordering for both scene variants |
| `targets` | `[K,4]` | Mean of repeated `[collision, goal_cost, space_cost, motion_cost]` outcomes |
| `intervention_obs_*` | same as `obs_*` | Sensor history after intervention |
| `intervention_targets` | `[K,4]` | Repeated-branch outcome means after the scene intervention |

Goal cost is remaining traversable path distance divided by `goal_scale`.
Unreachable distance is capped at `distance_cap` before division. The reference
simulator uses a four-neighbor Dijkstra field, not straight-line distance.
Space cost is horizon-average maximum clipped pedestrian intrusion. Motion cost
is the horizon-average of the two squared normalized command changes divided by
two. A collision creates an absorbing terminal suffix with space cost one and
zero further motion. The collision step itself includes its commanded motion.
Do not normalize intervention differences with different scales.

Collision frequencies are repeated outcomes, not model probabilities. Save raw
branch logs separately for confidence intervals; the compact NPZ format stores
means only. The geometric reference simulator is deterministic under fixed scene
and commands, so repeating its rollouts alone does not estimate uncertain human
behavior. Replace it with stochastic, matched-seed simulator branches for that study.

## Manifest

One JSON object per line in `manifest.jsonl`:

```json
{"file":"scene_00000_state_00000.npz","group_id":"Town01_source_family_001","split":"train","simulator":"carla"}
```

Allowed workflow split names: `train`, `validation`, `calibration_fit`,
`calibration_select`, and `test`. Assign groups before collecting branches.
`group_id` must group all related scene variants and trajectories together. For
map-held-out tests, group at map level. The loader rejects an explicitly repeated
group assigned to different splits, but cannot detect two aliases for one map.
Audit provenance yourself. NPZ paths must resolve below the manifest directory.

All samples in a mini-batch need identical keys, L, K, H, and image dimensions.
Do not mix paired/unpaired NPZ files in one default DataLoader; train them in
separate passes or supply a custom collator with an explicit pair mask.

## Language training records

JSONL row format:

```json
{"record_id":"scene_001_step_002","group_id":"Town01_source_family_001","verified":true,"prompt":"Explain this serialized decision record: ...","response":"{\"selected\":0,\"comparator\":9,\"reason\":\"weighted_cost\",\"selected_risk\":0.04,\"comparator_risk\":0.06,\"contributions\":[0.1,-0.02,0.01],\"uncertainty\":\"predictions_are_uncertain\"}"}
```

This row documents syntax, not a real experiment. Replace it with checked records.
`verified=true` is an authoring assertion, not an automatic verification performed
by the language loader. Build targets using `expected_claims` and compare against
record templates or human annotations. Split language data by the same source
provenance before passing a training-only file to the trainer.

The checked decoder accepts only the fixed claims schema and renders verified
slots. It does not certify arbitrary prose. State-action and unchecked-language
baselines need separate datasets/adapters and offline annotation. Never add their
unverified prose to the checked online path by treating JSON syntax as truth.

## ROS observation packet

`/trace/observation` is `std_msgs/String` carrying JSON with `frame`, `stamp`,
`sensor_age`, and `min_range`. `frame` is a single-frame version of the observation
contract without `obs_` prefixes. `stamp` is the oldest sensor timestamp in ROS
seconds, `sensor_age` is age at publication, and `min_range` is the closest valid
LiDAR range in meters. The controller adds local monotonic transport age to the
reported age. Do not compare ROS epoch timestamps directly with monotonic clocks.

The JSON bridge favors explicit debugging over throughput. Replace it with a
typed custom message or an in-process tensor bridge for performance measurements,
and include serialization/transport in measured sensor-to-command latency.
