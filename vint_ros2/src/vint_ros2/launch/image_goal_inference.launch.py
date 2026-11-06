"""Launch registered-route image-goal ViNT inference."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Build the image-goal inference launch description."""
    package_share = Path(get_package_share_directory('vint_ros2'))
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=str(
                package_share / 'config' / 'image_goal_inference.yaml'
            ),
        ),
        DeclareLaunchArgument(
            'route_directory',
            description='Directory of converted directional routes.',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='vint_ros2',
            executable='image_goal_inference',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'route_directory': LaunchConfiguration('route_directory'),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
