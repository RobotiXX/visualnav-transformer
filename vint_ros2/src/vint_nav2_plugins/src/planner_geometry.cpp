// Copyright 2026 nle
// Licensed under the MIT License.

#include "vint_nav2_plugins/planner_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "tf2/utils.h"

namespace vint_nav2_plugins
{

double poseDistance(
  const geometry_msgs::msg::PoseStamped & first,
  const geometry_msgs::msg::PoseStamped & second)
{
  return std::hypot(
    first.pose.position.x - second.pose.position.x,
    first.pose.position.y - second.pose.position.y);
}

double pathLength(const nav_msgs::msg::Path & path)
{
  double distance = 0.0;
  for (std::size_t index = 1; index < path.poses.size(); ++index) {
    distance += poseDistance(path.poses[index - 1], path.poses[index]);
  }
  return distance;
}

nav_msgs::msg::Path collisionFreePrefix(
  const nav_msgs::msg::Path & path,
  const double collision_check_step_m,
  const PoseValidityFunction & pose_is_valid)
{
  if (collision_check_step_m <= 0.0 || !pose_is_valid) {
    throw std::invalid_argument(
            "Collision checking requires a positive step and validity function.");
  }
  nav_msgs::msg::Path output;
  output.header = path.header;
  for (std::size_t index = 0; index < path.poses.size(); ++index) {
    const auto & current = path.poses[index];
    const double current_yaw = tf2::getYaw(current.pose.orientation);
    if (!pose_is_valid(
        current.pose.position.x, current.pose.position.y, current_yaw))
    {
      break;
    }
    if (index > 0) {
      const auto & previous = path.poses[index - 1];
      const double segment = poseDistance(previous, current);
      const int samples = std::max(
        1, static_cast<int>(std::ceil(segment / collision_check_step_m)));
      const double previous_yaw = tf2::getYaw(previous.pose.orientation);
      const double yaw_delta = std::remainder(
        current_yaw - previous_yaw, 2.0 * M_PI);
      bool segment_is_valid = true;
      for (int sample = 1; sample < samples; ++sample) {
        const double ratio = static_cast<double>(sample) / samples;
        if (!pose_is_valid(
            previous.pose.position.x + ratio *
            (current.pose.position.x - previous.pose.position.x),
            previous.pose.position.y + ratio *
            (current.pose.position.y - previous.pose.position.y),
            previous_yaw + ratio * yaw_delta))
        {
          segment_is_valid = false;
          break;
        }
      }
      if (!segment_is_valid) {
        break;
      }
    }
    output.poses.push_back(current);
  }
  return output;
}

nav_msgs::msg::Path transformRobotPathToMap(
  const nav_msgs::msg::Path & robot_path,
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const std::string & map_frame)
{
  if (robot_path.poses.empty()) {
    throw std::invalid_argument("Robot-relative path is empty.");
  }
  if (map_frame.empty()) {
    throw std::invalid_argument("Map frame is empty.");
  }
  nav_msgs::msg::Path output;
  output.header = robot_path.header;
  output.header.frame_id = map_frame;
  const double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
  const double cosine = std::cos(robot_yaw);
  const double sine = std::sin(robot_yaw);
  output.poses.reserve(robot_path.poses.size());
  for (const auto & local_pose : robot_path.poses) {
    geometry_msgs::msg::PoseStamped map_pose = local_pose;
    map_pose.header = output.header;
    map_pose.pose.position.x = robot_pose.pose.position.x +
      cosine * local_pose.pose.position.x - sine * local_pose.pose.position.y;
    map_pose.pose.position.y = robot_pose.pose.position.y +
      sine * local_pose.pose.position.x + cosine * local_pose.pose.position.y;
    const double yaw = robot_yaw + tf2::getYaw(local_pose.pose.orientation);
    map_pose.pose.orientation.x = 0.0;
    map_pose.pose.orientation.y = 0.0;
    map_pose.pose.orientation.z = std::sin(yaw / 2.0);
    map_pose.pose.orientation.w = std::cos(yaw / 2.0);
    output.poses.push_back(std::move(map_pose));
  }
  return output;
}

nav_msgs::msg::Path joinPaths(
  const nav_msgs::msg::Path & prefix,
  const nav_msgs::msg::Path & tail,
  const double duplicate_tolerance)
{
  if (prefix.poses.empty() || tail.poses.empty()) {
    throw std::invalid_argument("Both prefix and tail paths are required.");
  }
  if (prefix.header.frame_id != tail.header.frame_id) {
    throw std::invalid_argument("Prefix and tail path frames differ.");
  }
  if (duplicate_tolerance < 0.0) {
    throw std::invalid_argument("Duplicate tolerance must be non-negative.");
  }
  nav_msgs::msg::Path output = prefix;
  output.header = tail.header;
  std::size_t tail_start = 0;
  if (poseDistance(prefix.poses.back(), tail.poses.front()) <= duplicate_tolerance) {
    tail_start = 1;
  }
  output.poses.insert(
    output.poses.end(), tail.poses.begin() + tail_start, tail.poses.end());
  return output;
}

}  // namespace vint_nav2_plugins
