"""Lifecycle tests for image-goal controller stopping behavior."""

from pathlib import Path
import time
import unittest

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path as NavPath
import rclpy
from rclpy.action import ActionClient


def generate_test_description():
    """Launch a minimal controller server with the ViNT plugin."""
    parameters = str(
        Path(__file__).with_name('image_goal_controller_test.yaml')
    )
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[parameters],
    )
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='test_vint_lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': ['controller_server'],
        }],
    )
    transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='test_vint_static_transform',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'odom', '--child-frame-id', 'base_link',
        ],
        output='screen',
    )
    return LaunchDescription([
        transform,
        controller_server,
        lifecycle_manager,
        launch_testing.actions.ReadyToTest(),
    ])


class TestImageGoalControllerLifecycle(unittest.TestCase):
    """Exercise FollowPath cancellation and stale-prediction outcomes."""

    @classmethod
    def setUpClass(cls):
        """Create one ROS client node for the launched server."""
        rclpy.init()
        cls.node = rclpy.create_node('test_image_goal_controller_client')
        cls.action_client = ActionClient(
            cls.node, FollowPath, '/follow_path'
        )
        cls.lifecycle_client = cls.node.create_client(
            GetState, '/controller_server/get_state'
        )
        cls.prediction_publisher = cls.node.create_publisher(
            NavPath, '/test_vint/predicted_path', 10
        )
        cls.commands = []
        cls.command_subscription = cls.node.create_subscription(
            Twist, '/cmd_vel', cls.commands.append, 10
        )

    @classmethod
    def tearDownClass(cls):
        """Release ROS clients."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def _spin_until(self, predicate, timeout_sec=10.0):
        deadline = time.monotonic() + timeout_sec
        while not predicate() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(predicate())

    def _path(self, goal_x):
        path = NavPath()
        path.header.frame_id = 'odom'
        path.header.stamp = self.node.get_clock().now().to_msg()
        for x in (0.0, goal_x):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _prediction(self):
        path = NavPath()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.node.get_clock().now().to_msg()
        for x in (0.0, 0.1, 0.2):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _send_goal(self, goal_x):
        self.assertTrue(
            self.lifecycle_client.wait_for_service(timeout_sec=15.0)
        )
        activation_deadline = time.monotonic() + 15.0
        is_active = False
        while time.monotonic() < activation_deadline:
            state_future = self.lifecycle_client.call_async(
                GetState.Request()
            )
            self._spin_until(state_future.done)
            if (state_future.result().current_state.id ==
                    State.PRIMARY_STATE_ACTIVE):
                is_active = True
                break
            time.sleep(0.05)
        self.assertTrue(is_active)
        self.assertTrue(self.action_client.wait_for_server(timeout_sec=15.0))
        goal = FollowPath.Goal()
        goal.path = self._path(goal_x)
        goal.controller_id = 'FollowPath'
        goal.goal_checker_id = 'goal_checker'
        future = self.action_client.send_goal_async(goal)
        self._spin_until(future.done)
        handle = future.result()
        self.assertTrue(handle.accepted)
        return handle

    def _publish_until_moving(self, start_index):
        def moving():
            return any(
                abs(command.linear.x) > 1.0e-6 or
                abs(command.angular.z) > 1.0e-6
                for command in self.commands[start_index:]
            )

        deadline = time.monotonic() + 4.0
        while not moving() and time.monotonic() < deadline:
            self.prediction_publisher.publish(self._prediction())
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(moving())

    def _assert_zero_after(self, start_index):
        self._spin_until(
            lambda: any(
                abs(command.linear.x) <= 1.0e-9 and
                abs(command.angular.z) <= 1.0e-9
                for command in self.commands[start_index:]
            ),
            timeout_sec=3.0,
        )

    def test_cancel_and_stale_prediction_both_stop_motion(self):
        """Verify grace, cancellation, and stale failure produce zero output."""
        first_start = len(self.commands)
        first_goal = self._send_goal(2.0)

        # During initial inference grace, the plugin commands no motion.
        self._assert_zero_after(first_start)
        self._publish_until_moving(first_start)

        cancel_start = len(self.commands)
        cancel_future = first_goal.cancel_goal_async()
        self._spin_until(cancel_future.done)
        result_future = first_goal.get_result_async()
        self._spin_until(result_future.done)
        self.assertEqual(
            result_future.result().status, GoalStatus.STATUS_CANCELED
        )
        self._assert_zero_after(cancel_start)

        second_start = len(self.commands)
        second_goal = self._send_goal(3.0)
        self._publish_until_moving(second_start)

        stale_start = len(self.commands)
        stale_result = second_goal.get_result_async()
        self._spin_until(stale_result.done, timeout_sec=4.0)
        self.assertEqual(
            stale_result.result().status, GoalStatus.STATUS_ABORTED
        )
        self._assert_zero_after(stale_start)


@launch_testing.post_shutdown_test()
class TestProcessesExit(unittest.TestCase):
    """Check that the launched Nav2 processes terminate normally."""

    def test_exit_codes(self, proc_info):
        """Require clean process termination after the launch test."""
        launch_testing.asserts.assertExitCodes(proc_info)
