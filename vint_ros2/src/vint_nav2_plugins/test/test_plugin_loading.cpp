// Copyright 2026 nle
// Licensed under the MIT License.

#include <memory>

#include "gtest/gtest.h"
#include "nav2_core/controller.hpp"
#include "nav2_core/global_planner.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"

class PluginLoadingTest : public testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }
};

TEST_F(PluginLoadingTest, LoadsPointGoalPlannerThroughNav2Interface)
{
  pluginlib::ClassLoader<nav2_core::GlobalPlanner> loader(
    "nav2_core", "nav2_core::GlobalPlanner");

  auto plugin = loader.createSharedInstance(
    "vint_nav2_plugins/PointGoalGlobalPlanner");

  ASSERT_NE(plugin, nullptr);
}

TEST_F(PluginLoadingTest, LoadsImageGoalControllerThroughNav2Interface)
{
  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");

  auto plugin = loader.createSharedInstance(
    "vint_nav2_plugins/ImageGoalController");

  ASSERT_NE(plugin, nullptr);
}
