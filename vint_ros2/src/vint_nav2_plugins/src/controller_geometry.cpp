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
  const double heading_x, const double heading_y,
  const double linear_gain, const double angular_gain,
  const double maximum_linear_speed, const double maximum_angular_speed)
{
  if (linear_gain <= 0.0 || angular_gain <= 0.0 ||
    maximum_linear_speed <= 0.0 || maximum_angular_speed <= 0.0)
  {
    throw std::invalid_argument("Invalid waypoint controller parameter.");
  }
  if (!std::isfinite(x) || !std::isfinite(y) ||
    !std::isfinite(heading_x) || !std::isfinite(heading_y))
  {
    throw std::invalid_argument("ViNT waypoint contains a non-finite value.");
  }

  constexpr double epsilon = 1.0e-8;
  double angle = 0.0;
  if (std::abs(x) < epsilon && std::abs(y) < epsilon) {
    angle = std::atan2(heading_y, heading_x);
  } else if (std::abs(x) < epsilon) {
    constexpr double half_pi = 1.5707963267948966;
    angle = std::copysign(half_pi, y);
  } else {
    angle = std::atan(y / x);
  }
  geometry_msgs::msg::Twist command;
  command.linear.x = std::clamp(linear_gain * x, 0.0, maximum_linear_speed);
  command.angular.z = std::clamp(
    angular_gain * angle,
    -maximum_angular_speed,
    maximum_angular_speed);
  return command;
}

}  // namespace vint_nav2_plugins
