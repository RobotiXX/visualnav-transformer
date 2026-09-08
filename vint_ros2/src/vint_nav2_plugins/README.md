# vint_nav2_plugins

This ament C++ package provides two independent Nav2 plugins:

- an image-goal ViNT local controller used with a conventional global planner;
- a PointGoal ViNT global planner that joins a collision-checked learned path
  prefix to a complete map-aware path tail.

Both plugins use stamped ROS interfaces to separately managed Python inference
nodes. Missing, stale, mismatched, backward, or colliding inference results fail
closed.

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
plugin accepts only recent `base_link` predictions that start at the robot,
checks the padded footprint along the prediction against the local costmap, and
converts a configurable predicted point into bounded forward velocity. It does
not allow reverse motion. Within the active goal checker's XY tolerance, the
plugin rotates toward the final Nav2 goal orientation.

Copy `config/image_goal_controller.yaml` into the active controller-server
configuration. `waypoint_index` counts predicted model points and does not count
the origin pose inserted at the start of the ROS path.
The profile composer reads this installed file by default and accepts
`--image-goal-controller-config` for an experiment-specific copy.

The controller applies
`v = linear_gain * waypoint_x` and
`w = angular_gain * atan2(waypoint_y, waypoint_x)`, then clamps both commands.
It is forward-only. Nav2's active goal checker determines completion; once the
robot is inside that checker's XY tolerance, the controller rotates toward the
final plan yaw.

The collision gate first transforms the stamped robot-relative prediction into
the local-costmap frame at the current robot pose. It samples every path segment
at `collision_check_step_m`; every sample must have an acceptable center-cell
cost and an acceptable complete padded-footprint cost. This gate catches
obstacles between model outputs but only reflects what the current local
costmap can represent. It is not a substitute for hardware collision or
emergency-stop systems.

The YAML file documents every parameter in place. The most important groups
are:

- `prediction_timeout_sec` and `initial_prediction_grace_sec`: freshness and
  startup failure behavior;
- `maximum_path_start_error_m`: the robot-frame origin contract;
- `collision_check_step_m` and `collision_cost_threshold`: continuous local
  costmap validation;
- `waypoint_index` and `minimum_forward_x_m`: model target selection and
  forward-only enforcement;
- control gains and maximum speeds: bounded conversion from a predicted point
  to velocity; and
- final-rotation parameters: goal-yaw control after Nav2 accepts XY.

Unlike the original ROS 1 ViNT deployment, this plugin participates in Nav2
lifecycle, cancellation, goal checking, speed-limit, costmap, and terminal-zero
semantics. The Python inference node publishes the predicted path, the C++
plugin returns a velocity command, and Nav2's controller server publishes that
command on `/cmd_vel`. The RWD global planner and registered metric visual route
replace the original unregistered topomap neighborhood search.

## Current validation boundary

The plugins compile and their geometry/command tests pass on ROS 2 Humble. A
mock PointGoal inference service is tested through ROS middleware. The
image-goal controller-server launch test covers activation, live commands,
cancellation, stale-prediction abort, and zero commands after both terminal
paths. The released image-goal checkpoint loads and runs through the Python
backend on the development GPU. Closed-loop trained-model navigation and robot
motion remain unchecked and are tracked in the workspace README.
