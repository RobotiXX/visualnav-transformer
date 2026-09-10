// Copyright 2026 nle
// Licensed under the MIT License.

#ifndef VINT_NAV2_PLUGINS__CONTROLLER_GEOMETRY_HPP_
#define VINT_NAV2_PLUGINS__CONTROLLER_GEOMETRY_HPP_

#include "geometry_msgs/msg/twist.hpp"

namespace vint_nav2_plugins
{

geometry_msgs::msg::Twist waypointCommand(
  double x, double y, double heading_x, double heading_y,
  double linear_gain, double angular_gain,
  double maximum_linear_speed, double maximum_angular_speed);

}  // namespace vint_nav2_plugins

#endif  // VINT_NAV2_PLUGINS__CONTROLLER_GEOMETRY_HPP_
