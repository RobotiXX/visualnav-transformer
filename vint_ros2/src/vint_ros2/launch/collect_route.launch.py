"""Collect a timestamp-registered visual route as a ROS 2 bag."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import EmitEvent
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _launch_collection(context):
    profile_path = Path(
        LaunchConfiguration('profile').perform(context)
    ).expanduser().resolve()
    output = LaunchConfiguration('output').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    with profile_path.open('r', encoding='utf-8') as stream:
        profile = yaml.safe_load(stream)

    image_topic = profile['image_topic']
    pose_topic = profile['pose_topic']
    topics = list(dict.fromkeys([
        image_topic,
        pose_topic,
        *profile.get('topics', []),
    ]))
    sampler = Node(
        package='vint_ros2',
        executable='map_pose_sampler',
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'image_type': profile['image_type'],
            'output_topic': pose_topic,
            'fixed_frame': profile['fixed_frame'],
            'robot_frame': profile['robot_frame'],
            'transform_timeout_sec': profile['transform_timeout_sec'],
            'warning_throttle_sec': profile['warning_throttle_sec'],
            'use_sim_time': use_sim_time.lower() == 'true',
        }],
    )
    recorder = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--output', output,
            '--storage', str(profile.get('storage_id', 'mcap')),
            '--max-cache-size', str(profile.get(
                'max_cache_size', 104857600
            )),
            *topics,
        ],
        output='screen',
    )
    stop_when_recorder_exits = OnProcessExit(
        target_action=recorder,
        on_exit=[EmitEvent(event=Shutdown(
            reason='Route bag recorder exited.',
        ))],
    )
    from launch.actions import RegisterEventHandler
    return [sampler, recorder, RegisterEventHandler(stop_when_recorder_exits)]


def generate_launch_description():
    """Build the visual-route collection launch description."""
    package_share = get_package_share_directory('vint_ros2')
    default_profile = str(
        Path(package_share) / 'config' / 'scoutmini_route_collection.yaml'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'profile',
            default_value=default_profile,
            description='Collection topic/profile YAML.',
        ),
        DeclareLaunchArgument(
            'output',
            default_value='vint_route_bag',
            description='New rosbag2 output directory.',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_launch_collection),
    ])
