// Copyright 2026 nle
// Licensed under the MIT License.

#include "vint_nav2_plugins/controller_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vint_nav2_plugins
{

geometry_msgs::msg::Twist waypointCommand(
  const double x, const double y,
  const double linear_gain, const double angular_gain,
  const double maximum_linear_speed, const double maximum_angular_speed,
  const double minimum_forward_x)
{
  if (linear_gain <= 0.0 || angular_gain <= 0.0 ||
    maximum_linear_speed <= 0.0 || maximum_angular_speed <= 0.0 ||
    minimum_forward_x < 0.0)
  {
    throw std::invalid_argument("Invalid waypoint controller parameter.");
  }
  if (!std::isfinite(x) || !std::isfinite(y) || x < minimum_forward_x) {
    throw std::invalid_argument("ViNT waypoint is not safely forward of the robot.");
  }
  geometry_msgs::msg::Twist command;
  command.linear.x = std::clamp(linear_gain * x, 0.0, maximum_linear_speed);
  command.angular.z = std::clamp(
    angular_gain * std::atan2(y, x),
    -maximum_angular_speed,
    maximum_angular_speed);
  return command;
}

double boundedRotationCommand(
  const double yaw_error, const double gain,
  const double minimum_speed, const double maximum_speed)
{
  if (gain <= 0.0 || minimum_speed < 0.0 ||
    maximum_speed <= 0.0 || minimum_speed > maximum_speed)
  {
    throw std::invalid_argument("Invalid rotation controller parameter.");
  }
  if (std::abs(yaw_error) < 1.0e-9) {
    return 0.0;
  }
  const double magnitude = std::clamp(
    std::abs(gain * yaw_error), minimum_speed, maximum_speed);
  return std::copysign(magnitude, yaw_error);
}

}  // namespace vint_nav2_plugins
