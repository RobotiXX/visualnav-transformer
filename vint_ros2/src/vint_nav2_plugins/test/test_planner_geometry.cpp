// Copyright 2026 nle
// Licensed under the MIT License.

#include <cmath>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"
#include "tf2/utils.h"
#include "vint_nav2_plugins/planner_geometry.hpp"

namespace
{

geometry_msgs::msg::PoseStamped pose(
  const double x, const double y, const double yaw,
  const std::string & frame = "map")
{
  geometry_msgs::msg::PoseStamped result;
  result.header.frame_id = frame;
  result.pose.position.x = x;
  result.pose.position.y = y;
  result.pose.orientation.z = std::sin(yaw / 2.0);
  result.pose.orientation.w = std::cos(yaw / 2.0);
  return result;
}

TEST(PlannerGeometry, TransformsRobotRelativePathWithRobotHeading)
{
  nav_msgs::msg::Path local;
  local.header.frame_id = "base_link";
  local.poses = {pose(0.0, 0.0, 0.0, "base_link"),
    pose(1.0, 0.0, 0.0, "base_link")};

  const auto output = vint_nav2_plugins::transformRobotPathToMap(
    local, pose(10.0, -2.0, M_PI_2), "map");

  ASSERT_EQ(output.poses.size(), 2u);
  EXPECT_NEAR(output.poses[1].pose.position.x, 10.0, 1.0e-9);
  EXPECT_NEAR(output.poses[1].pose.position.y, -1.0, 1.0e-9);
  EXPECT_NEAR(tf2::getYaw(output.poses[1].pose.orientation), M_PI_2, 1.0e-9);
}

TEST(PlannerGeometry, JoinsCompletePathWithoutDuplicateEndpoint)
{
  nav_msgs::msg::Path prefix;
  prefix.header.frame_id = "map";
  prefix.poses = {pose(0.0, 0.0, 0.0), pose(1.0, 0.0, 0.0)};
  nav_msgs::msg::Path tail;
  tail.header.frame_id = "map";
  tail.poses = {pose(1.01, 0.0, 0.0), pose(2.0, 0.0, 0.0)};

  const auto output = vint_nav2_plugins::joinPaths(prefix, tail, 0.05);

  ASSERT_EQ(output.poses.size(), 3u);
  EXPECT_DOUBLE_EQ(output.poses.back().pose.position.x, 2.0);
  EXPECT_NEAR(vint_nav2_plugins::pathLength(output), 2.0, 1.0e-9);
}

TEST(PlannerGeometry, RejectsMismatchedPathFrames)
{
  nav_msgs::msg::Path prefix;
  prefix.header.frame_id = "map";
  prefix.poses = {pose(0.0, 0.0, 0.0)};
  nav_msgs::msg::Path tail = prefix;
  tail.header.frame_id = "odom";

  EXPECT_THROW(
    vint_nav2_plugins::joinPaths(prefix, tail, 0.05),
    std::invalid_argument);
}

TEST(PlannerGeometry, KeepsOnlyContinuousCollisionFreePrefix)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.poses = {
    pose(0.0, 0.0, 0.0),
    pose(1.0, 0.0, 0.0),
    pose(2.0, 0.0, 0.0),
    pose(3.0, 0.0, 0.0)};

  const auto output = vint_nav2_plugins::collisionFreePrefix(
    path, 0.1,
    [](const double x, const double, const double) {return x < 1.55;});

  ASSERT_EQ(output.poses.size(), 2u);
  EXPECT_DOUBLE_EQ(output.poses.back().pose.position.x, 1.0);
  EXPECT_EQ(output.header.frame_id, "map");
}

TEST(PlannerGeometry, RejectsImmediatelyInvalidPrefix)
{
  nav_msgs::msg::Path path;
  path.poses = {pose(0.0, 0.0, 0.0), pose(1.0, 0.0, 0.0)};

  const auto output = vint_nav2_plugins::collisionFreePrefix(
    path, 0.1,
    [](const double, const double, const double) {return false;});

  EXPECT_TRUE(output.poses.empty());
}

}  // namespace
