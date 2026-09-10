// Copyright 2026 nle
// Licensed under the MIT License.

#include "vint_nav2_plugins/image_goal_controller.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "vint_nav2_plugins/controller_geometry.hpp"
#include "vint_nav2_plugins/planner_geometry.hpp"

namespace vint_nav2_plugins
{

void ImageGoalController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  const std::string name,
  std::shared_ptr<tf2_ros::Buffer>/*tf*/,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  if (!node_) {
    throw std::runtime_error("Image-goal controller parent node expired.");
  }
  name_ = name;

  const auto declare = [this](const std::string & parameter, const auto & value) {
      nav2_util::declare_parameter_if_not_declared(
        node_, name_ + "." + parameter, rclcpp::ParameterValue(value));
    };
  declare("robot_frame", costmap_ros->getBaseFrameID());
  declare("global_plan_topic", std::string("/vint/image_goal/global_plan"));
  declare("prediction_topic", std::string("/vint/image_goal/predicted_path"));
  declare("waypoint_index", 2);
  declare("minimum_prediction_points", 2);
  declare("prediction_timeout_sec", 1.0);
  declare("maximum_path_start_error_m", 0.25);
  declare("linear_gain", 4.0);
  declare("angular_gain", 4.0);
  declare("maximum_linear_speed", 0.2);
  declare("maximum_angular_speed", 0.4);

  std::string global_plan_topic;
  std::string prediction_topic;
  node_->get_parameter(name_ + ".robot_frame", robot_frame_);
  node_->get_parameter(name_ + ".global_plan_topic", global_plan_topic);
  node_->get_parameter(name_ + ".prediction_topic", prediction_topic);
  node_->get_parameter(name_ + ".waypoint_index", waypoint_index_);
  node_->get_parameter(
    name_ + ".minimum_prediction_points", minimum_prediction_points_);
  node_->get_parameter(name_ + ".prediction_timeout_sec", prediction_timeout_sec_);
  node_->get_parameter(
    name_ + ".maximum_path_start_error_m", maximum_path_start_error_m_);
  node_->get_parameter(name_ + ".linear_gain", linear_gain_);
  node_->get_parameter(name_ + ".angular_gain", angular_gain_);
  node_->get_parameter(
    name_ + ".maximum_linear_speed", configured_maximum_linear_speed_);
  node_->get_parameter(
    name_ + ".maximum_angular_speed", maximum_angular_speed_);
  maximum_linear_speed_ = configured_maximum_linear_speed_;

  if (waypoint_index_ < 0 || minimum_prediction_points_ < 2 ||
    prediction_timeout_sec_ <= 0.0 ||
    maximum_path_start_error_m_ < 0.0 ||
    linear_gain_ <= 0.0 || angular_gain_ <= 0.0 ||
    maximum_linear_speed_ <= 0.0 || maximum_angular_speed_ <= 0.0)
  {
    throw std::runtime_error("Invalid image-goal controller parameters.");
  }

  global_plan_publisher_ = node_->create_publisher<nav_msgs::msg::Path>(
    global_plan_topic, rclcpp::QoS(1).reliable());
  prediction_subscription_ = node_->create_subscription<nav_msgs::msg::Path>(
    prediction_topic,
    rclcpp::QoS(1).reliable(),
    std::bind(
      &ImageGoalController::predictionReceived, this,
      std::placeholders::_1));
  RCLCPP_INFO(
    node_->get_logger(), "Configured %s for predictions on %s.",
    name_.c_str(), prediction_topic.c_str());
}

void ImageGoalController::cleanup()
{
  active_ = false;
  prediction_subscription_.reset();
  global_plan_publisher_.reset();
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    global_plan_ = nav_msgs::msg::Path();
    prediction_ = nav_msgs::msg::Path();
  }
  node_.reset();
}

void ImageGoalController::activate()
{
  global_plan_publisher_->on_activate();
  active_ = true;
}

void ImageGoalController::deactivate()
{
  active_ = false;
  global_plan_publisher_->on_deactivate();
}

void ImageGoalController::setPlan(const nav_msgs::msg::Path & path)
{
  if (path.poses.empty()) {
    throw nav2_core::PlannerException(
            "Image-goal controller received an empty plan.");
  }
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    const bool new_destination = global_plan_.poses.empty() ||
      poseDistance(global_plan_.poses.back(), path.poses.back()) > 0.25;
    global_plan_ = path;
    if (new_destination) {
      prediction_ = nav_msgs::msg::Path();
    }
  }
  if (active_) {
    global_plan_publisher_->publish(path);
  }
}

void ImageGoalController::predictionReceived(nav_msgs::msg::Path::SharedPtr path)
{
  if (!active_ || path->header.frame_id != robot_frame_ ||
    path->poses.size() < static_cast<std::size_t>(minimum_prediction_points_))
  {
    return;
  }
  geometry_msgs::msg::PoseStamped origin;
  origin.pose.orientation.w = 1.0;
  if (poseDistance(path->poses.front(), origin) > maximum_path_start_error_m_) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000,
      "Rejecting a ViNT path that does not begin near the robot origin.");
    return;
  }
  std::lock_guard<std::mutex> lock(data_mutex_);
  prediction_ = *path;
}

geometry_msgs::msg::TwistStamped ImageGoalController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  if (!active_) {
    throw nav2_core::PlannerException("Image-goal controller is inactive.");
  }
  nav_msgs::msg::Path plan;
  nav_msgs::msg::Path prediction;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    plan = global_plan_;
    prediction = prediction_;
  }
  if (plan.poses.empty()) {
    throw nav2_core::PlannerException(
            "Image-goal controller has no global plan.");
  }
  geometry_msgs::msg::TwistStamped output;
  output.header = pose.header;
  const auto & goal = plan.poses.back().pose;
  if (goal_checker != nullptr && goal_checker->isGoalReached(
      pose.pose, goal, velocity))
  {
    return output;
  }

  const auto now = node_->now();
  const rclcpp::Time prediction_stamp(prediction.header.stamp);
  const double age = (now - prediction_stamp).seconds();
  if (prediction.poses.size() <
    static_cast<std::size_t>(minimum_prediction_points_))
  {
    return output;
  }
  if (age < 0.0 || age > prediction_timeout_sec_) {
    return output;
  }
  const std::size_t index = std::min(
    static_cast<std::size_t>(waypoint_index_ + 1),
    prediction.poses.size() - 1);
  const auto & target_pose = prediction.poses[index].pose;
  const auto & target = target_pose.position;
  const double target_yaw = tf2::getYaw(target_pose.orientation);
  output.twist = waypointCommand(
    target.x, target.y, std::cos(target_yaw), std::sin(target_yaw),
    linear_gain_, angular_gain_, maximum_linear_speed_, maximum_angular_speed_);
  return output;
}

void ImageGoalController::setSpeedLimit(
  const double & speed_limit,
  const bool & percentage)
{
  if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    maximum_linear_speed_ = configured_maximum_linear_speed_;
  } else if (percentage) {
    maximum_linear_speed_ = configured_maximum_linear_speed_ *
      std::clamp(speed_limit, 0.0, 100.0) / 100.0;
  } else {
    maximum_linear_speed_ = std::clamp(
      speed_limit, 0.0, configured_maximum_linear_speed_);
  }
}

}  // namespace vint_nav2_plugins

PLUGINLIB_EXPORT_CLASS(
  vint_nav2_plugins::ImageGoalController,
  nav2_core::Controller)
