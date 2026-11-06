# ViNT ROS 2 Humble Integration

This directory is a standalone ROS 2 workspace for collecting ViNT data,
building metric-registered visual topological maps, training or fine-tuning
ViNT models, and deploying ViNT through Nav2. It targets Ubuntu 22.04 with ROS 2
Humble on both an x86-64 training workstation and an NVIDIA Jetson running the
corresponding JetPack 6 release.

This README is also the durable implementation tracker. Update the checklist
and the **Resume here** section whenever implementation pauses.

## Status

Last updated: 2026-09-06

- Current phase: image-goal route collection and supervised deployment testing;
  PointGoal training is paused
- Build status: `vint_ros2` and `vint_nav2_plugins` build successfully; the
  latest isolated test run is recorded below
- Hardware status: no robot motion test has been performed
- Model status: the released image-goal `checkpoints/vint.pth` loads and runs
  through the ROS 2 backend; no trained PointGoal checkpoint exists

### Resume here

Record and convert the first directional route, configure the released
image-goal checkpoint, then run an operator-submitted dashboard task with
`VINT_MODE=image_goal_local`. PointGoal model development is intentionally
paused; its interfaces and deterministic mock remain for later work, but it is
not a user-study profile. Image-goal controller lifecycle, cancellation,
one-second prediction expiry, and zero-command behavior are covered by automated
tests. No trained ViNT profile has yet moved the simulated or real robot.

## Why this is not a literal ROS 1 port

The original deployment launches a LoCoBot camera and base, performs ViNT
inference in a ROS 1 Python node, publishes a float array on `/waypoint`, and
uses a separate PD node to publish velocity commands. That design assumes ROS
1 Noetic, a particular robot, relative working-directory paths, and manual tmux
process ownership.

The ROS 2 implementation keeps the model-specific code replaceable and follows
Nav2 lifecycle and cancellation semantics. ViNT nodes never publish directly to
the robot velocity topic. The task executor remains the owner of task lifecycle,
and Nav2 remains the owner of path execution and velocity output.

The implementation follows two facts from the ViNT paper:

1. Released ViNT is image-goal conditioned. It consumes the current image plus
   temporal context and a goal image, then predicts a short robot-relative
   trajectory and temporal distance.
2. PointGoal ViNT is an adaptation, not the released image-goal checkpoint. A
   learned goal encoder replaces the image-goal encoder with a token derived
   from the goal coordinates expressed in the robot frame.

References:

- [ViNT paper](https://arxiv.org/abs/2306.14846)
- [ViNT project page](https://general-navigation-models.github.io/vint/index.html)
- [Original ROS 1 deployment guide](../README.md#deployment)
- [RVN-Bench PointGoal formulation](https://arxiv.org/abs/2603.03953)

## Workspace and packages

```text
vint_ros2/
├── README.md
├── requirements/
└── src/
    ├── vint_ros2/          Python inference, data, topomap, and training tools
    └── vint_nav2_plugins/  C++ Nav2 controller and global-planner plugins
```

`vint_ros2` is ViNT-specific and belongs in this fork rather than the portable
RWD workspace. `vint_nav2_plugins` uses standard ROS 2 and Nav2 interfaces. The
RWD repository only supplies optional configuration and launch composition, so
its normal workspace can still build without PyTorch or this repository.

## Two supported Nav2 roles

### Mode A: ViNT as a local navigation policy

Mode A keeps the existing occupancy-grid global planner and evaluates ViNT in
the local-controller position:

```text
semantic destination
  -> RWD task planner and executor
  -> Nav2 Theta* global path (route activation and destination check)
  -> ViNT distance-based local topomap search
  -> image-goal ViNT
  -> third robot-relative predicted waypoint
  -> original ViNT PD conversion in a Nav2 controller plugin
  -> Nav2 velocity output
```

The registered visual route contains ordered camera observations and map-frame
poses. Registration identifies which directional route belongs to the Nav2
task, but it does not change ViNT's local navigation algorithm. As in
`deployment/src/navigate.py`, the inference node searches a temporal
neighborhood around its current topomap index, runs the distance and action
heads for all candidate goal images in one batch, and advances to the image
after the closest candidate when its predicted distance is at most the close
threshold.

This mode can use the released image-goal checkpoint. A route must be recorded
for each travel direction because a forward-facing image sequence recorded from
A to B is not an equivalent visual goal sequence for B to A.

#### Fidelity to the original ROS 1 deployment

The study profile intentionally preserves nominal ViNT behavior and limits the
adapter to the interfaces required by Nav2. The comparison below uses
`deployment/src/navigate.py`, `deployment/src/pd_controller.py`, and
`deployment/config/robot.yaml` as the executable reference.

| Behavior | Original ROS 1 deployment | ROS 2/Nav2 study profile |
|---|---|---|
| Model | Released image-goal ViNT checkpoint | Same architecture and checkpoint; exact state-dictionary validation is added at load time |
| Observation | Current image plus five context images | Same six-image tensor |
| Image transform | Resize to `[85, 64]`, ImageNet normalization, no center crop in `navigate.py` | Same transform |
| Route search | Batched images from `closest_node - 4` through `closest_node + 5`, clipped to the route | Same batched temporal neighborhood |
| Topological localization | Minimum distance-head prediction | Same minimum-distance selection |
| Node advance | Use the following image when minimum distance is at most `3` | Same threshold and advance rule |
| Inference rate | `4 Hz` | `4 Hz` |
| Control rate | PD node runs at `9 Hz` and holds the latest waypoint | Nav2 controller runs at the common `10 Hz` study rate and holds the latest prediction |
| Metric conversion | Example robot uses `0.2 / 4 = 0.05 m` per normalized unit | ScoutMini study uses `0.8 / 4 = 0.20 m` per normalized unit |
| Control target | Predicted waypoint index `2` (the third of five points) | Same waypoint index |
| PD conversion | `v=x/DT`; `w=atan(y/x)/DT`; predicted heading only when X/Y are nearly zero | Same equations with `DT=0.25 s` |
| Command limits | Example robot uses `[0, 0.2] m/s`; yaw rate `[-0.4, 0.4] rad/s` | ScoutMini study uses `[0, 0.8] m/s`; yaw rate remains `[-0.4, 0.4] rad/s` before common platform-level limiting |
| Lost waypoint | PD input expires after one second and no new motion is published | A missing or one-second-old prediction returns zero velocity without aborting the Nav2 goal |
| Camera stops after context is full | Inference continues from the retained context | Same behavior; the inference-time prediction stamp does not add a camera-age gate |
| Collision checking | None in `navigate.py` or `pd_controller.py` | None in `ImageGoalController`; the local costmap does not veto ViNT output |
| Final orientation | No separate goal-yaw rotation | No adapter-added final rotation; the study goal checker accepts any yaw |

The speed and metric-scale values differ by robot rather than by algorithm.
The ViNT paper defines embodiment-normalized relative waypoints and delegates
unnormalization to a robot-specific controller using the target robot's top
speed. The released repository implements this contract as `max_v / frame_rate`.
The ScoutMini study therefore pairs a 0.8 m/s operating limit with a 0.20 m
scale at 4 Hz while retaining the original third-point target.

Six integration differences remain. First, route images are stored in a
metric manifest so the task can select the correct A-to-B route; the map poses
do not select the local goal image. Second, the Nav2 global plan identifies the
active destination but does not steer, prune, replace, or collision-check the
ViNT rollout. Third, the Python node publishes the five predicted points as a
stamped `nav_msgs/Path`, while the C++ plugin performs the original PD
conversion and returns `TwistStamped` to the controller server. Fourth, the
controller callback runs at the common 10 Hz study rate instead of the ROS 1
PD node's 9 Hz rate, while both hold each 4 Hz model output between updates.
Fifth, Nav2 owns action lifecycle, cancellation, a 15-second shared progress
check, speed-limit requests, and metric goal completion instead of terminating
when the topomap index reaches its last image. Sixth, the common Nav2 velocity
smoother can enforce platform acceleration limits after the ViNT controller's
original velocity clamps. These differences are necessary for the common
task-execution boundary and are recorded as part of the experimental system
rather than attributed to ViNT.

#### Evaluation safety boundary

The image-goal adapter performs message-contract checks for a valid frame,
point count, origin, finite values, and timestamp. These checks reject malformed
transport data but do not alter valid model trajectories. In particular, the
adapter does not query the Nav2 costmap or add obstacle-based trajectory
selection. Operator supervision, emergency stopping, and any platform-level
safeguards therefore remain outside the evaluated ViNT policy.

### Mode B: ViNT-PointGoal as a Nav2 global planner (paused)

Mode B evaluates the adapted PointGoal model in Nav2's global-planner position
while retaining a conventional controller such as DWB:

```text
semantic destination
  -> RWD task planner and executor
  -> Nav2 ComputePathToPose
  -> vint_nav2_plugins PointGoal planner
       current RGB history + goal in base frame
       -> learned short-horizon path prefix
       -> occupancy-grid validation
       -> map-aware path tail to the final goal
  -> DWB or MPPI
```

ViNT predicts only a short trajectory, so treating its last predicted point as
the final Nav2 path endpoint would cause premature task success. The planner
adapter instead returns a complete start-to-goal path: it transforms the learned
prefix into the map frame, truncates it at the first colliding or out-of-bounds
sample, and joins the remaining continuous prefix to a conventional
occupancy-grid path tail. The planner fails closed when fewer than two prefix
poses remain; it never substitutes a purely classical plan while reporting the
PointGoal profile. Nav2 periodically
replans, allowing each new camera observation to update the learned prefix.

This mode requires a PointGoal-adapted checkpoint. The implemented goal-token
encoder concatenates robot-relative goal coordinates with a learned latent and
uses a two-layer MLP to produce the goal token. The loader does not silently
accept an image-goal checkpoint as a PointGoal model.

The current encoder and data path are a prototype, not a reproduction of the
paper's GPS adaptation. The paper concatenates the 2-D goal with a fixed learned
tensor of size 3000, while this prototype currently defaults to a latent size
of 32 and hidden size of 256. The generic dataset also samples goals anywhere
in its configured future range rather than restricting them to the paper's
1.25--1.75 s interval. Because the new encoder begins with random weights, the
released image-goal checkpoint cannot make this mode operational by itself.
PointGoal training and deployment are therefore paused until the design and
sampling contract below are completed and evaluated.

Although this adapter implements the `nav2_core::GlobalPlanner` contract, its
learned component is a receding-horizon visual planner. Experiment reports must
distinguish that learned prefix from the conventional map-aware tail.

### User-study comparison matrix

The initial configuration will support these controlled combinations:

| Profile | Global path | Local control | Initial feasibility |
|---|---|---|---|
| Classical baseline | Theta* | DWB | Existing |
| ViNT local | Theta* | image-goal ViNT | One or two recorded A-to-B routes |
| ViNT PointGoal planner | PointGoal ViNT prefix + map tail | DWB | Paused; requires an adapted checkpoint |
| ViNT full | PointGoal ViNT prefix + map tail | image-goal ViNT | Paused; confounds both learned components |

The first three profiles isolate the changed component. The fourth profile is
useful for feasibility tests but confounds global and local learned behavior.

## Data contracts

### ROS 2 collection topics

All topic names are parameters. The initial ScoutMini profile will use:

- a forward-facing `sensor_msgs/msg/Image` or
  `sensor_msgs/msg/CompressedImage` camera topic;
- `/odom` as `nav_msgs/msg/Odometry` for measured linear and angular
  velocity and motion diagnostics;
- `/tf` and `/tf_static` for map registration;
- a sampled `geometry_msgs/msg/PoseStamped` map pose produced at camera time;
- `/plan` and task status for debugging and experiment audit.

The converter aligns every retained image with a pose at the same timestamp and
rejects samples outside a configurable synchronization tolerance. It produces
the existing ViNT training layout:

```text
<dataset>/<trajectory>/
├── 0.jpg
├── 1.jpg
├── ...
└── traj_data.pkl  # position: float[T,2], yaw: float[T]
```

### Registered visual route manifest

Each route has an explicit direction, map frame, image sequence, and map pose:

```yaml
schema_version: "1.0"
route_id: fuse_3312_to_lab_front
map_id: fuse_3rd
frame_id: map
source_waypoint_id: fuse.floor_3.3312
destination_waypoint_id: fuse.floor_3.lab_front
nodes:
  - index: 0
    image: images/000000.jpg
    stamp_ns: 0
    pose: {x: 0.0, y: 0.0, yaw: 0.0}
```

Full waypoint IDs are metadata and are never inferred from filenames. For
deployment, place each selected converted trajectory below one route directory;
the runtime recursively loads its `route.yaml` files. It permits one route per
map, source waypoint ID, and destination waypoint ID tuple, then selects a route
from `/map_name` and the Nav2 global plan's start and destination positions.
The default source and destination tolerances are both 2 m. Ambiguous endpoint
matches and plans without a registered directional route fail closed.
The latched `/vint/image_goal/selected_route` topic reports the active
`route_id` and publishes an empty string when no route is selected.

## Environment strategy

ROS 2 Python packages are installed by Ubuntu under the system Python. Create a
Python 3.10 virtual environment with `--system-site-packages` so `rclpy`,
`rosbag2_py`, and generated ROS interfaces remain visible. Do not add automatic
activation to shell startup files.

The workstation profile targets the current Ubuntu 22.04 machine with an RTX
3090 Ti. Training and checkpoint conversion run there. The Jetson profile
targets JetPack 6 and installs the NVIDIA-provided PyTorch wheel matching the
installed JetPack release; the guide will not hard-code one wheel URL across
JetPack versions.

Model weights, bags, datasets, and experiment logs are ignored artifacts and
must not be committed.

### Workstation setup: Ubuntu 22.04 and ROS 2 Humble

Run these commands from the repository root. The virtual environment inherits
ROS packages from the system Python but keeps ViNT dependencies local:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv-vint
source .venv-vint/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch and torchvision versions selected for the
workstation through the [official PyTorch installation
selector](https://pytorch.org/get-started/locally/), then install the remaining
dependencies and the original training package:

```bash
python -m pip install -r vint_ros2/requirements/workstation.txt
python -m pip install -e train
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import rclpy, rosbag2_py; print('ROS 2 Python available')"
```

Build the standalone workspace without sourcing `rwd_ws` or `ros2_ws`:

```bash
cd vint_ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select vint_ros2 vint_nav2_plugins
source install/setup.bash
```

For integrated ScoutMini tests, source the existing workspaces first and add
this standalone workspace as the final local overlay. `local_setup.bash` adds
ViNT without replaying the underlays recorded when it was built:

```bash
cd ~/repos/ScoutMini
source /opt/ros/humble/setup.bash
source rwd_ws/install/setup.bash
source ros2_ws/install/local_setup.bash
source visualnav-transformer/vint_ros2/install/local_setup.bash
```

### Robot setup: JetPack 6 and ROS 2 Humble

Use the same `--system-site-packages` virtual-environment layout. Install the
[NVIDIA Jetson PyTorch wheel](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
that exactly matches the robot's JetPack release, then install a compatible
torchvision build and `requirements/jetson.txt`. Do not copy the workstation's
x86-64 CUDA wheel or virtual environment to the Jetson. Build with colcon on
the robot after both `python -c "import torch"` and `python -c "import rclpy"`
succeed in the same shell.

After copying the selected checkpoint and its matching YAML to the robot, run
these checks in the same virtual environment and sourced ROS shell used for
deployment:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
python -c "import rclpy, rosbag2_py; print('ROS 2 Python available')"
ros2 pkg prefix vint_ros2
ros2 pkg prefix vint_nav2_plugins
```

The `mock` backend is the default in both inference YAML files. It supports
interface and Nav2 lifecycle testing without torch, a GPU, or a checkpoint.
Switch `backend` to `torch` only after setting an absolute trusted checkpoint
path, a model config path, and a device. Legacy image-goal checkpoints may
contain pickled Python modules and must only be loaded from the official ViNT
release or a trusted local training run.

## ROS 2 workflow

### 1. Record one directional route

Start the normal robot or simulation stack without submitting a task. Drive
the exact A-to-B route manually and record the forward camera, a camera-time
map pose, TF, odometry, plans, and task diagnostics:

```bash
# Robot: standard topics supplied by scoutmini_rwd_adapter
mkdir -p ~/ssd/vint_bags
ros2 launch vint_ros2 collect_route.launch.py \
  profile:=$(ros2 pkg prefix --share vint_ros2)/config/scoutmini_route_collection.yaml \
  output:=~/ssd/vint_bags/fuse_3312_to_lab_front_01

# Simulation: raw Gazebo camera and simulated clock
ros2 launch vint_ros2 collect_route.launch.py \
  profile:=$(ros2 pkg prefix --share vint_ros2)/config/simulation_route_collection.yaml \
  use_sim_time:=true \
  output:=~/ssd/vint_bags/fuse_3312_to_lab_front_sim_01
```

Stop the launch with `Ctrl-C` after the traversal. The recorder uses MCAP with
a 100 MiB cache. A route is directional: collect B-to-A separately rather than
reversing image order. Collect multiple runs per direction because dataset
splits assign complete trajectories, and at least two trajectories are
required for separate training and test sets.

Run the normal handheld robot controller separately while collecting. Keyboard
teleoperation and the Nav2 velocity smoother are not collection requirements.
The selected remote bypasses ROS velocity commands, so the hardware profile
does not wait for or record `/cmd_vel`; `/odom` retains measured linear
and angular velocity instead. The converter follows the original ViNT data
contract and derives normalized actions from synchronized map-pose differences,
not commanded velocity. Stop the collector terminal with `Ctrl-C` only after
the robot is stationary.

### 2. Convert and validate the route

The converter samples camera frames at 4 Hz, synchronizes each retained frame
to `/vint/map_pose`, and atomically writes both the original ViNT training
layout and `route.yaml` with map-frame registration:

```bash
ros2 run vint_ros2 convert_bag ~/ssd/vint_bags/fuse_3312_to_lab_front_01 \
  --output-root datasets/scoutmini \
  --trajectory-name fuse_3312_to_lab_front_01 \
  --image-topic /camera/image/compressed \
  --sample-rate 4.0 \
  --sync-tolerance 0.10 \
  --route-id fuse_3312_to_lab_front \
  --map-id fuse_3rd \
  --source-waypoint-id fuse.floor_3.3312 \
  --destination-waypoint-id fuse.floor_3.lab_front

ros2 run vint_ros2 validate_route \
  datasets/scoutmini/fuse_3312_to_lab_front_01/route.yaml
```

Use `/camera/image` for a bag collected with the simulation profile. Output
directories are deliberately non-overwriting; choose a new trajectory name
instead of replacing evidence from an earlier run.

### 3. Prepare deterministic splits

```bash
ros2 run vint_ros2 prepare_dataset datasets/scoutmini \
  --split-output data_splits/scoutmini \
  --test-fraction 0.2 \
  --seed 0
```

The command prints sample count, path length, and mean per-trajectory sample
spacing. Set `scoutmini.metric_waypoint_spacing` in
`train/vint_train/data/data_config.yaml` and
`metric_waypoint_spacing_m` in both fine-tuning YAML files to the selected
measured spacing before training. The former normalizes training targets; the
latter converts predictions back to meters during deployment.

### 4. Fine-tune the image-goal model

Run training from `train/` because its defaults and output paths are relative
to that directory:

```bash
cd train
python train.py -c ../vint_ros2/src/vint_ros2/config/scoutmini_vint_finetune.yaml
```

The ScoutMini profile initializes from the released `checkpoints/vint.pth`
weights with a fresh optimizer. Each run writes `latest.pth` after every epoch,
an `epoch_NNN.pth` checkpoint every five completed epochs, and `best.pth`
whenever mean validation loss improves. It also updates `validation_loss.csv`
and `validation_loss.png` below the configured
`<log_root>/rwd-vint/<timestamped-run>/`. Keep each checkpoint paired with the
copied `training_config.yaml` used to construct it. Fine-tuning is optional for the first
image-goal experiment: start with the released checkpoint and a registered
route, then collect site data if its held-out behavior is inadequate.

#### PointGoal data and training plan (paused)

PointGoal does not require recreating ViNT's broad pretraining corpus. The ViNT
paper reports 80 h of data used across eight platforms for general
pretraining, but its goal-modality adaptations use at most 5 h and often work
with 1--2 h. At a 4 Hz collection rate, 1 h, 2 h, and 5 h correspond to
approximately 14,400, 28,800, and 72,000 camera frames before synchronization
and quality filtering. These figures are planning estimates, not a guarantee
that a particular building or robot will reach acceptable performance.

A reasonable resume plan begins with at least 1 h of clean, varied trajectories
as a pilot and targets 2 h before deciding whether to expand toward 5 h. The
dataset should contain both directions of each study route, different starting
offsets, turns, corridors, doorways, lighting, and pedestrian conditions.
Training and test sets must be separated by complete trajectories or route
segments to prevent adjacent frames from leaking into both sets; reserve at
least 20% of trajectories for held-out evaluation. `collect_route.launch.py`
records the camera, camera-time map pose, odometry, and TF. `convert_bag` then
creates image frames and `traj_data.pkl`; robot-relative coordinate goals and
future path labels are derived from those poses, so no hand annotation is
required.

Before resuming training, complete these steps:

1. Match the published 3000-element learned goal latent, or define and evaluate
   an explicit smaller-latent ablation.
2. Add a PointGoal-specific sampler for goals 1.25--1.75 s into the future. At
   4 Hz, this interval is approximately five to seven frames.
3. Initialize all compatible observation encoder, transformer, and prediction
   weights from the released image-goal model while initializing the new goal
   encoder separately; compare frozen and full-fine-tuning variants.
4. Reproduce the paper's four-epoch warmup to a learning rate of `1e-4`
   followed by cosine decay, then evaluate data-volume checkpoints at 0.5, 1,
   2, and 5 h as available.
5. Report held-out path error, goal progress, collision-proxy violations, and
   closed-loop success on routes excluded from training.
6. Export a checkpoint with explicit `model_type: pointgoal_vint`, the complete
   model YAML, dataset split, and configuration hashes.
7. Restore periodic-replanning, final-goal, simulation, and supervised robot
   tests only after the offline checks pass.

### 5. Configure inference

Copy the applicable inference YAML to an experiment directory. For a trained
model, change `backend` from `mock` to `torch`, set `checkpoint_path` and
`model_config_path` to absolute trusted files, and select `cuda` or `cpu`.
Use `image_goal_inference.yaml` and `pointgoal_inference.yaml` on the robot;
their camera topic is supplied by the ScoutMini adapter. The `_sim.yaml`
profiles use Gazebo's raw camera and simulated time.

For the released image-goal weights, use
`config/vint_release_model.yaml`. The official checkpoint contains pickled
training optimizer metadata; the backend supplies a narrow compatibility shim
for its missing warmup-scheduler class and loads the trusted checkpoint onto
CPU before transferring only model weights to the configured device. Do not
use this legacy unpickling path for an untrusted checkpoint.

After loading the weights, the Torch backend runs one discarded synthetic
inference in the selected mode. This startup warmup initializes CUDA kernels
before the node accepts a live global path, preventing startup latency from
affecting the first measured prediction.

The inference nodes can be exercised independently:

```bash
ros2 launch vint_ros2 image_goal_inference.launch.py \
  params_file:=/absolute/path/to/image_goal_experiment.yaml \
  route_directory:=/absolute/path/to/deployment_routes

ros2 launch vint_ros2 pointgoal_inference.launch.py \
  params_file:=/absolute/path/to/pointgoal_experiment.yaml
```

Neither node publishes `/cmd_vel`. The image-goal node publishes a short
`base_link` path for its Nav2 controller, while the PointGoal node serves a
correlated `nav_msgs/srv/GetPlan` request from its Nav2 planner plugin.

#### Released-checkpoint validation and 4 Hz deployment

The official 430 MB `checkpoints/vint.pth` was loaded through
`TorchViNTBackend` on the development workstation's RTX 3090 Ti. With synthetic
85-by-64 RGB inputs, a 50-run check measured 1.496 s to load, 0.189 s for the
first inference, and steady-state p50/p95/p99 inference times of
6.38/6.65/7.34 ms. The backend returned the expected five trajectory points.
This smoke test verifies checkpoint format, architecture configuration, CUDA
execution, and output shape; synthetic images cannot establish whether the
policy will follow a recorded route.

After restoring the original local-topomap algorithm, a second checkpoint
smoke test passed six candidate goal images through one batch. The backend
returned six finite distance predictions and one five-point trajectory per
candidate, matching the selector's expected contract.

The target robot loop is the original 4 Hz, or one new prediction every 0.25 s.
The Nav2 controller runs at the common 10 Hz study rate and reuses the most
recent prediction between inference updates. Its one-second expiry reproduces
the ROS 1 PD node's waypoint timeout; expiry returns zero velocity rather than
aborting the Nav2 action. Before motion, measure model warmup and sustained p50,
p95, and p99 end-to-end latency on the Jetson, including image decoding,
host/device transfer, batched local-topomap inference, and ROS publication.
Also monitor missed cycles, thermal throttling, camera timestamp age, and GPU
memory during a route-length run.

The paper and released deployment make the metric conversion robot-specific:
ViNT normalizes relative waypoints across embodiments, and deployment multiplies
them by `max_v / frame_rate`. The ScoutMini study sets the operating maximum to
0.8 m/s, uses a paired 0.20 m scale at 4 Hz, and retains waypoint index 2, the
third predicted point. The larger scale already increases both the physical
feedback distance and attainable command by four relative to the example 0.2
m/s robot; selecting a farther point at the same time would add a second change
and can reduce corner responsiveness. A later 1.0 m/s trial must change the
scale and clamp together to 0.25 m and 1.0 m/s. If batched inference cannot
sustain 4 Hz on the robot, report the measured rate as a deviation rather than
changing the scale to compensate for runtime jitter.

### 6. Compose controlled Nav2 profiles

The composer starts from the complete rendered RWD configuration and replaces
only the implementation stored under the stable task-selected `FollowPath`
and/or `GridBased` plugin ID:

```bash
ros2 run rwd_config render_config nav2_sim --output /tmp/nav2_sim_base.yaml
ros2 run vint_ros2 compose_nav2_profile /tmp/nav2_sim_base.yaml \
  --mode image_goal_local --output /tmp/nav2_sim_vint_local.yaml

ros2 run rwd_config render_config nav2 --output /tmp/nav2_base.yaml
ros2 run vint_ros2 compose_nav2_profile /tmp/nav2_base.yaml \
  --mode pointgoal_global --output /tmp/nav2_vint_pointgoal.yaml
```

Valid modes are `classical`, `image_goal_local`, `pointgoal_global`, and
`vint_full`. Door and elevator controller IDs are preserved, so existing typed
task-executor steps keep their configured behavior. Plugin tuning lives in
`vint_nav2_plugins/config/image_goal_controller.yaml` and
`pointgoal_planner.yaml`. Copy either file and pass
`--image-goal-controller-config` or `--pointgoal-planner-config` to the
composer for an experiment-specific profile.

`pointgoal_global` and `vint_full` remain available only for deterministic mock
interface testing while PointGoal training is paused. Do not select either mode
for a navigation-system comparison or robot deployment.

### 7. Run an end-to-end dashboard task in simulation

Source the three workspaces in the order shown above, then run one of these
commands from the ScoutMini repository. `sim_byobu` starts no destination; use
the dashboard to submit the task after the stack is ready:

```bash
VINT_MODE=image_goal_local \
VINT_ROUTE_DIRECTORY=/absolute/path/to/deployment_routes \
./scripts/sim_byobu

VINT_MODE=pointgoal_global ./scripts/sim_byobu
```

The default inference profiles use the deterministic mock backend for interface
testing. Supply `VINT_IMAGE_GOAL_PARAMS_FILE` or
`VINT_POINTGOAL_PARAMS_FILE` to select a copied experiment YAML with trained
weights. `VINT_MODE=none` is the unchanged RWD simulation. The dashboard,
semantic planner, C++ behavior-tree executor, cancellation, status, and bag
recorder follow the same code path in every mode.

For plugin-parameter experiments, set `VINT_IMAGE_GOAL_CONTROLLER_CONFIG` or
`VINT_POINTGOAL_PLANNER_CONFIG` to the corresponding copied C++ plugin YAML.
These variables are separate from the inference parameter files because one
set configures Nav2 safety/control behavior and the other configures model
loading and camera inference.

For each explicit `VINT_MODE`, including `classical`, `sim_byobu` creates an
`experiment_context.json` beside the composed Nav2 YAML. The context records
the selected global planner and local controller plus absolute paths, sizes,
and SHA-256 digests for the effective model checkpoint, model YAML, inference
YAML, every route manifest plus its referenced images, and plugin YAML. It is loaded
once by both the task executor and bag recorder and copied into their
task-associated metadata. A
Torch profile with an empty or missing checkpoint fails before simulation
starts. To generate the artifact outside `sim_byobu`, use:

```bash
ros2 run vint_ros2 write_experiment_context \
  --mode pointgoal_global \
  --pointgoal-params /absolute/path/to/pointgoal_experiment.yaml \
  --output /tmp/pointgoal_context.json
```

### Hardware user-study launch

The ScoutMini hardware wrapper exposes image-goal ViNT as one of three
mutually exclusive study conditions. Source the RWD, ScoutMini, and ViNT
workspaces in that order, then provide a copied inference YAML configured with
`backend: torch` and a directory of directional routes registered to the task
paths:

```bash
NAVIGATION_MODE=vint \
VINT_IMAGE_GOAL_PARAMS_FILE=/absolute/path/to/frozen_torch_profile.yaml \
VINT_ROUTE_DIRECTORY=/absolute/path/to/deployment_routes \
./scripts/nav2_byobu
```

The wrapper selects the complete frozen `nav2_vint.yaml` hardware profile and
starts `image_goal_inference.launch.py` in its own Byobu tab. It refuses mock
inference, missing model resources, and invalid route data before creating the
hardware session. The wrapper also adds `visualnav-transformer/train` to the
ViNT pane's Python module path so the inference backend can construct the
original model classes used by the checkpoint. The shared experiment context
hashes the Nav2 snapshot, inference configuration, model checkpoint, controller
configuration, route manifests, and route images for the task and bag metadata. Use
`CHECK_ONLY=true` with the same arguments to validate all files without
launching drivers or navigation.

## Implementation checklist

### Phase 0: audit and architecture

- [x] Read the original ROS 1 collection, topomap, inference, and PD-control code.
- [x] Read the ViNT model, training configuration, and paper implementation details.
- [x] Review the existing RWD task executor, semantic planner, and Nav2 ownership.
- [x] Define separate image-goal local-policy and PointGoal global-planner modes.
- [x] Choose the standalone workspace and package boundaries.

### Phase 1: workspace and shared contracts

- [x] Scaffold the `vint_ros2` ament package.
- [x] Scaffold the `vint_nav2_plugins` ament package.
- [x] Implement and validate the registered-route manifest.
- [x] Implement registered path projection and route-corridor utilities for
  PointGoal experiments without using them to steer image-goal ViNT.
- [x] Add example one-way A-to-B route manifests without site data or images.
- [x] Add package tests and a dependency-light CI path.

### Phase 2: ROS 2 data collection and training

- [x] Add a timestamp-aligned map-pose publisher for rosbag2 recording.
- [x] Add configurable rosbag2 collection launch and topic profiles.
- [x] Convert raw and compressed image messages plus registered poses into ViNT datasets.
- [x] Build metric-registered visual topomaps from rosbag2.
- [x] Add deterministic train/test splitting and dataset validation.
- [x] Add a ScoutMini fine-tuning configuration without machine-specific paths.
- [x] Add workstation setup and training commands.
- [x] Add JetPack 6 deployment setup and checkpoint validation commands.

Phase 1 verification on 2026-09-06: both packages built with
`colcon build --symlink-install --packages-select vint_ros2 vint_nav2_plugins`;
`colcon test` reported 20 tests with zero failures or errors. The Python tests
include manifest failure cases, route-corridor utilities, registered topomap
selection, and PointGoal map/robot-frame transform round trips.

### Phase 3: image-goal ViNT local policy

- [x] Refactor model loading and image transforms out of the ROS 1 script.
- [x] Add a lazy PyTorch backend and a deterministic mock backend.
- [x] Implement ROS 2 image history and the original batched local-topomap selector.
- [x] Publish predicted trajectories as stamped `nav_msgs/msg/Path` in `base_link`.
- [x] Implement the original PD waypoint conversion as a Nav2 controller plugin.
- [x] Add a live mock-inference topic test and controller-server lifecycle smoke test.
- [x] Add Nav2 cancellation, waypoint-expiry, and zero-command regression tests.
- [x] Load the released `vint.pth` through the production backend and benchmark
  warm-up plus repeated synthetic inference on the development GPU.
- [ ] Record a directional route and evaluate the released checkpoint in
  closed-loop simulation.

### Phase 4: ViNT-PointGoal planner (paused before model training)

- [x] Implement the PointGoal goal-token encoder.
- [x] Add PointGoal dataset labels and fine-tuning configuration.
- [x] Add a PointGoal inference service with request correlation and timeout.
- [x] Implement the Nav2 global-planner plugin with a dedicated executor thread.
- [x] Validate learned prefixes against the local/global costmap.
- [x] Join accepted prefixes to a complete conventional map-aware path tail.
- [ ] Add periodic-replanning and final-goal regression tests.
- [x] Reject missing or incompatible PointGoal checkpoints.
- [ ] Align the goal-token architecture and future-goal sampler with the
  published PointGoal/GPS adaptation, or evaluate documented ablations.
- [ ] Collect, split, train, and evaluate a PointGoal dataset and export a
  checkpoint with explicit type metadata.

The completed items in this phase establish interfaces and deterministic mock
behavior only. Model training, closed-loop learned planning, and study use are
paused according to the data-and-training plan above.

Plugin verification on 2026-09-10: both packages build on ROS 2 Humble. The
latest isolated-domain run reports 122 tests with zero errors or failures and
11 environment-dependent lint skips.
The Python suite includes a live mock PointGoal service request, topic-profile
contracts, and profile composition. The C++ functional tests pass for plugin
loading, robot-to-map prefix transforms, continuous collision-free-prefix
truncation, and complete-path joining for the paused PointGoal planner. The
image-goal tests cover the original topomap advance rule, PD command equations,
speed bounds, activation, a live prediction, cancellation, waypoint expiry,
and terminal zero commands.

### Phase 5: RWD and user-study integration

- [x] Add optional RWD controller and planner configuration profiles.
- [x] Keep normal RWD builds independent of this workspace and PyTorch.
- [x] Add launch composition for the three controlled study profiles.
- [x] Preserve dashboard, cancellation, task status, and bag-recording behavior
  by retaining the normal RWD task path and stable Nav2 plugin IDs.
- [x] Record selected model/checkpoint/profile in task-associated experiment logs.
- [x] Add one-command dry-run and mock-inference tests.
- [x] Run one A-to-B route in simulation with deterministic mock inference and
  without automatic task submission.
- [ ] Resume PointGoal work and repeat the simulated A-to-B route only after a
  compatible checkpoint passes the offline evaluation gates.
- [ ] Run one A-to-B route on the robot under operator supervision.
- [ ] Repeat for the second route or reverse direction using a separately recorded route.

Headless integration smoke test on 2026-09-06: `sim_byobu` composed the
`pointgoal_global` profile on isolated ROS domain 220, the mock inference
service returned a six-pose `base_link` prefix, Nav2 loaded and activated the
PointGoal planner plus its Theta* tail, and terminating the byobu session
stopped the supervised processes. No navigation goal or velocity command was
sent. This test also found and fixed unsupported YAML aliases in rendered RWD
parameter files. Image-goal closed-loop task execution remains untested; the
released-checkpoint workstation latency is now recorded above.

A second headless smoke test on isolated domain 228 loaded and activated the
image-goal controller behind `FollowPath`, started mock image-goal inference,
passed one shared context file to the executor and recorder, and verified a
three-image route-content digest. Session termination cleaned up the supervised
simulation. This test verifies plugin lifecycle and experiment wiring, but it
does not replace an A-to-B trained-model navigation. Cancellation and expired
inference are covered by the isolated controller-server launch test.

A moving headless test on isolated domain 219 submitted `FUSE 3200` from the
simulated `lab_front` start through `ExecuteTask`. The mission remained one
normal semantic navigation step, the PointGoal planner supplied a short mock
prefix plus a Theta* tail, DWB reached the destination, the result retained the
original alias, and the task log recorded the shared `pointgoal_global`
experiment context. Terminating the byobu session left no matching simulation
or inference process. The simulation mock uses a 5 cm prefix because its
straight-line prediction has no visual or map understanding; a separate test
to `FUSE 3338` correctly failed closed at a corner once fewer than two
collision-free prefix poses remained. Consequently, the successful run checks
ROS 2 and Nav2 integration only and is not evidence of trained PointGoal model
performance.

## Image-goal completion criteria

The active milestone is complete when an operator can select a predefined
single-floor destination through the existing dashboard, the normal task
executor runs the task, and the image-goal ViNT controller can be selected
entirely through configuration. Cancellation must stop motion, stale inference
must produce zero velocity, the dashboard must show the same task lifecycle as the
classical profile, and the bag metadata must identify the model checkpoint and
navigation profile. PointGoal has separate resume gates and is not required for
this milestone.
