// Copyright 2026 nle
// Licensed under the MIT License.

#include <stdexcept>

#include "gtest/gtest.h"
#include "vint_nav2_plugins/controller_geometry.hpp"

TEST(ControllerGeometry, ComputesBoundedForwardCommand)
{
  const auto command = vint_nav2_plugins::waypointCommand(
    1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.01);

  EXPECT_DOUBLE_EQ(command.linear.x, 0.5);
  EXPECT_DOUBLE_EQ(command.angular.z, 1.0);
}

TEST(ControllerGeometry, RejectsBackwardWaypoint)
{
  EXPECT_THROW(
    vint_nav2_plugins::waypointCommand(
      -0.1, 0.0, 1.0, 2.0, 0.5, 1.0, 0.01),
    std::invalid_argument);
}

TEST(ControllerGeometry, AppliesMinimumFinalRotationSpeed)
{
  EXPECT_DOUBLE_EQ(
    vint_nav2_plugins::boundedRotationCommand(0.01, 1.0, 0.1, 1.0),
    0.1);
  EXPECT_DOUBLE_EQ(
    vint_nav2_plugins::boundedRotationCommand(-2.0, 1.0, 0.1, 1.0),
    -1.0);
}
