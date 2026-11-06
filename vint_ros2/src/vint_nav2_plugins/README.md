# vint_nav2_plugins

This ament C++ package provides two independent Nav2 plugins:

- an image-goal ViNT local controller used with a conventional global planner;
- a PointGoal ViNT global planner that joins a collision-checked learned path
  prefix to a complete map-aware path tail.

Both plugins use stamped ROS interfaces to separately managed Python inference
nodes. The paused PointGoal planner retains its experimental map-aware prefix
validation. The image-goal controller instead reproduces the original released
deployment behavior and deliberately performs no costmap collision check.

## PointGoal global planner

> **Paused:** the plugin and deterministic mock are retained for interface
> testing, but no trained PointGoal checkpoint exists. Do not use this profile
> for the user study or robot deployment. The workspace README records the
> required architecture, data-collection, training, and evaluation work.

`vint_nav2_plugins/PointGoalGlobalPlanner` implements
`nav2_core::GlobalPlanner`. It calls `/vint/pointgoal_inference` through a
dedicated executor thread, transforms the returned robot-relative prefix at the
planning start pose, checks the complete padded robot footprint between every
predicted point, and asks a nested Theta* plugin for the remaining path to the
actual Nav2 goal. If a later part of the prediction is unsafe, the planner
retains only the longest continuously collision-free prefix and joins Theta* at
its last safe pose. Fewer than `minimum_prefix_points` safe poses still fail
closed; the plugin does not silently replace an immediately unsafe prediction
with a fully classical path. This complete-path splice prevents the predicted
prefix endpoint from being mistaken for task completion.

Copy the parameters in `config/pointgoal_planner.yaml` into the active Nav2
planner-server configuration. The nested tail planner is configured under
`<plugin_name>.tail`. Normal Nav2 replanning reruns visual prefix inference.
The `vint_ros2 compose_nav2_profile` command reads this installed file by
default and accepts `--pointgoal-planner-config` for a copied, tuned version.

## Image-goal local controller

`vint_nav2_plugins/ImageGoalController` implements `nav2_core::Controller`.
Each Nav2 global plan is published to the registered-route inference node. The
plugin checks the stamped `base_link` message contract and converts the selected
predicted point with the equations from the original ROS 1 `pd_controller.py`.
It does not inspect the local costmap, reject a valid model rollout based on
obstacles, or add a final goal-yaw rotation.

Copy `config/image_goal_controller.yaml` into the active controller-server
configuration. `waypoint_index` counts predicted model points and does not count
the origin pose inserted at the start of the ROS path.
The profile composer reads this installed file by default and accepts
`--image-goal-controller-config` for an experiment-specific copy.

The controller applies
`v = linear_gain * waypoint_x` and
`w = angular_gain * atan(waypoint_y / waypoint_x)`, then clamps both commands.
With the original `4 Hz` model convention, both gains are `4`. Linear velocity
is clamped to `[0, 0.8] m/s` for the ScoutMini study, and yaw rate is clamped to
`[-0.4, 0.4] rad/s`. When both waypoint coordinates are nearly zero, the model's
predicted heading supplies the angular command. A negative X value is not an
adapter error; the original forward clamp turns it into zero linear velocity.
The paired model profile converts embodiment-normalized points with
`0.8 m/s / 4 Hz = 0.20 m`. This pairing follows ViNT's robot-specific
unnormalization contract; a 1.0 m/s experiment requires both a 0.25 m action
scale and a 1.0 m/s clamp.

The YAML file documents every parameter in place. The most important groups
are:

- `prediction_timeout_sec`: the original one-second waypoint expiry, expressed
  as an explicit zero command for Nav2;
- `maximum_path_start_error_m`: the robot-frame origin contract;
- `waypoint_index`: original model target selection; and
- control gains and maximum speeds: the original PD conversion and clamps.

Unlike the original ROS 1 ViNT deployment, this plugin participates in Nav2
lifecycle, cancellation, goal checking, speed-limit, and terminal-zero
semantics. The Python inference node publishes the predicted path, the C++
plugin returns a velocity command, and Nav2's controller server publishes that
command on `/cmd_vel`. The registered metric route selects the task-compatible
direction, while the inference node retains the original distance-based local
topomap neighborhood search.

## Current validation boundary

The plugins compile and their geometry/command tests pass on ROS 2 Humble. A
mock PointGoal inference service is tested through ROS middleware. The
image-goal controller-server launch test covers activation, live commands,
cancellation, one-second prediction expiry, and zero commands after both
stopping paths. The released image-goal checkpoint loads and runs through the Python
backend on the development GPU. Closed-loop trained-model navigation and robot
motion remain unchecked and are tracked in the workspace README.
