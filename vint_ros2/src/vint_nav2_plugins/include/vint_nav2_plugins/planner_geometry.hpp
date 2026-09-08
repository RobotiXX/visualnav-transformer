// Copyright 2026 nle
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

#ifndef VINT_NAV2_PLUGINS__PLANNER_GEOMETRY_HPP_
#define VINT_NAV2_PLUGINS__PLANNER_GEOMETRY_HPP_

#include <functional>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

namespace vint_nav2_plugins
{

using PoseValidityFunction = std::function<bool (double, double, double)>;

double poseDistance(
  const geometry_msgs::msg::PoseStamped & first,
  const geometry_msgs::msg::PoseStamped & second);

double pathLength(const nav_msgs::msg::Path & path);

nav_msgs::msg::Path collisionFreePrefix(
  const nav_msgs::msg::Path & path,
  double collision_check_step_m,
  const PoseValidityFunction & pose_is_valid);

nav_msgs::msg::Path transformRobotPathToMap(
  const nav_msgs::msg::Path & robot_path,
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const std::string & map_frame);

nav_msgs::msg::Path joinPaths(
  const nav_msgs::msg::Path & prefix,
  const nav_msgs::msg::Path & tail,
  double duplicate_tolerance);

}  // namespace vint_nav2_plugins

#endif  // VINT_NAV2_PLUGINS__PLANNER_GEOMETRY_HPP_
