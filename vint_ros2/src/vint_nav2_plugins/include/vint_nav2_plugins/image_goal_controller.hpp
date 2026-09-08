// Copyright 2026 nle
// Licensed under the MIT License.

#ifndef VINT_NAV2_PLUGINS__IMAGE_GOAL_CONTROLLER_HPP_
#define VINT_NAV2_PLUGINS__IMAGE_GOAL_CONTROLLER_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace vint_nav2_plugins
{

class ImageGoalController : public nav2_core::Controller
{
public:
  ImageGoalController() = default;
  ~ImageGoalController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(
    const double & speed_limit,
    const bool & percentage) override;

private:
  void predictionReceived(nav_msgs::msg::Path::SharedPtr path);
  bool pathIsCollisionFree(
    const nav_msgs::msg::Path & path,
    const geometry_msgs::msg::PoseStamped & robot_pose) const;

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  std::string robot_frame_;
  std::string costmap_frame_;
  std::atomic_bool active_{false};
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};

  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr
    global_plan_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr prediction_subscription_;

  mutable std::mutex data_mutex_;
  nav_msgs::msg::Path global_plan_;
  nav_msgs::msg::Path prediction_;
  rclcpp::Time plan_set_time_{0, 0, RCL_ROS_TIME};

  int waypoint_index_{2};
  int minimum_prediction_points_{2};
  int collision_cost_threshold_{253};
  double prediction_timeout_sec_{0.5};
  double initial_prediction_grace_sec_{0.5};
  double maximum_path_start_error_m_{0.25};
  double collision_check_step_m_{0.025};
  double minimum_forward_x_m_{0.02};
  double linear_gain_{1.0};
  double angular_gain_{2.0};
  double configured_maximum_linear_speed_{0.5};
  double maximum_linear_speed_{0.5};
  double maximum_angular_speed_{1.0};
  double final_rotation_gain_{2.0};
  double final_minimum_angular_speed_{0.1};
};

}  // namespace vint_nav2_plugins

#endif  // VINT_NAV2_PLUGINS__IMAGE_GOAL_CONTROLLER_HPP_
