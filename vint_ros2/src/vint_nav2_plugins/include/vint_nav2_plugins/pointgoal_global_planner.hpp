// Copyright 2026 nle
// Licensed under the MIT License.

#ifndef VINT_NAV2_PLUGINS__POINTGOAL_GLOBAL_PLANNER_HPP_
#define VINT_NAV2_PLUGINS__POINTGOAL_GLOBAL_PLANNER_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav_msgs/srv/get_plan.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace vint_nav2_plugins
{

class PointGoalGlobalPlanner : public nav2_core::GlobalPlanner
{
public:
  PointGoalGlobalPlanner() = default;
  ~PointGoalGlobalPlanner() override;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  nav_msgs::msg::Path requestPrefix(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal);
  nav_msgs::msg::Path collisionFreePrefix(
    const nav_msgs::msg::Path & path) const;
  bool poseIsCollisionFree(
    double x, double y, double yaw,
    nav2_costmap_2d::FootprintCollisionChecker<
      nav2_costmap_2d::Costmap2D *> & checker) const;
  void stopClientExecutor();

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  std::string global_frame_;
  std::string robot_frame_;
  std::string inference_service_;
  double inference_timeout_sec_{0.5};
  double maximum_prefix_start_error_m_{0.25};
  double minimum_goal_progress_m_{-0.25};
  double collision_check_step_m_{0.025};
  double duplicate_tolerance_m_{0.05};
  int minimum_prefix_points_{2};
  int collision_cost_threshold_{253};
  std::atomic_bool active_{false};

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  std::shared_ptr<tf2_ros::Buffer> tf_;

  std::unique_ptr<pluginlib::ClassLoader<nav2_core::GlobalPlanner>> tail_loader_;
  nav2_core::GlobalPlanner::Ptr tail_planner_;

  rclcpp::Node::SharedPtr client_node_;
  rclcpp::Client<nav_msgs::srv::GetPlan>::SharedPtr inference_client_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> client_executor_;
  std::thread client_thread_;
};

}  // namespace vint_nav2_plugins

#endif  // VINT_NAV2_PLUGINS__POINTGOAL_GLOBAL_PLANNER_HPP_
