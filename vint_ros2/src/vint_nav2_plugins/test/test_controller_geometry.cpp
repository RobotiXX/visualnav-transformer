// Copyright 2026 nle
// Licensed under the MIT License.

#include <stdexcept>

#include "gtest/gtest.h"
#include "vint_nav2_plugins/controller_geometry.hpp"

TEST(ControllerGeometry, ComputesBoundedForwardCommand)
{
  const auto command = vint_nav2_plugins::waypointCommand(
    1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.5, 1.0);

  EXPECT_DOUBLE_EQ(command.linear.x, 0.5);
  EXPECT_DOUBLE_EQ(command.angular.z, 1.0);
}

TEST(ControllerGeometry, ClipsBackwardWaypointToZeroLinearVelocity)
{
  const auto command = vint_nav2_plugins::waypointCommand(
    -0.1, 0.0, 1.0, 0.0, 1.0, 2.0, 0.5, 1.0);

  EXPECT_DOUBLE_EQ(command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(command.angular.z, 0.0);
}

TEST(ControllerGeometry, UsesPredictedHeadingOnlyAtZeroDisplacement)
{
  const auto command = vint_nav2_plugins::waypointCommand(
    0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 0.2, 0.4);

  EXPECT_DOUBLE_EQ(command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(command.angular.z, 0.4);
}
