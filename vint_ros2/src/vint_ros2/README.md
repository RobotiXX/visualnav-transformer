# vint_ros2

This ament Python package owns ROS 2 data conversion, registered visual-route
contracts, model inference, and training/deployment utilities for ViNT. It does
not publish robot velocity commands; Nav2 plugins in `vint_nav2_plugins` own
the Nav2-facing planner and controller contracts.

Validate a recorded route before deployment:

```bash
ros2 run vint_ros2 validate_route /path/to/route.yaml
```

The installed `config/example_route.yaml` illustrates the schema but has no
images. Validate metadata-only examples with `--allow-missing-images`. A real
route must provide every referenced image.

## Collect a ROS 2 route bag

The collector launches a camera-triggered TF sampler and an MCAP recorder. The
default ScoutMini profile records the adapter-owned `/camera/image/compressed`
stream, exact
`map -> base_link` samples on `/vint/map_pose`, odometry, TF, scan, velocity,
global plan, map name, and task status. Its cache is 100 MiB.

```bash
ros2 launch vint_ros2 collect_route.launch.py \
  output:=bags/fuse_3312_to_lab_front
```

Copy and edit `config/scoutmini_route_collection.yaml` to change robot topics
or frames, then pass the copy as `profile:=/absolute/path/profile.yaml`.
`config/simulation_route_collection.yaml` selects the raw Gazebo camera and
must be launched with `use_sim_time:=true`. Stop the launch with Ctrl-C; launch
shutdown also stops the bag recorder.

## Convert and split data

The converter uses message header timestamps, rate-limits images to 4 Hz by
default, rejects image/pose pairs outside the synchronization tolerance, and
refuses to overwrite an existing trajectory.

```bash
ros2 run vint_ros2 convert_bag bags/fuse_3312_to_lab_front \
  --output-root datasets/scoutmini \
  --trajectory-name fuse_3312_to_lab_front_run_01 \
  --image-topic /camera/image/compressed \
  --route-id fuse_3312_to_lab_front \
  --map-id fuse_3rd \
  --source-waypoint-id fuse.floor_3.3312 \
  --destination-waypoint-id fuse.floor_3.lab_front
```

The output directory contains numbered JPEGs and `traj_data.pkl` in the
original ViNT training format, plus a validated `route.yaml` used by the ROS 2
image-goal selector. Record each travel direction separately.

Create reproducible, trajectory-level splits after collecting at least two
runs:

```bash
ros2 run vint_ros2 prepare_dataset datasets/scoutmini \
  --split-output data_splits/scoutmini --test-fraction 0.2 --seed 0
```

The command validates all image and array lengths and prints the observed
metric spacing. Update the `scoutmini` value in
`train/vint_train/data/data_config.yaml` before training if that measured value
differs materially from the initial 0.25 m estimate.

See the workspace-level [`vint_ros2/README.md`](../../README.md) for the
complete workstation, Jetson, training, inference, Nav2 profile, and
`sim_byobu` workflow.
