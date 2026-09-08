// Copyright 2026 nle
// Licensed under the MIT License.

#include "vint_nav2_plugins/pointgoal_global_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <stdexcept>
#include <utility>

#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "vint_nav2_plugins/planner_geometry.hpp"

namespace vint_nav2_plugins
{

using namespace std::chrono_literals;

PointGoalGlobalPlanner::~PointGoalGlobalPlanner()
{
  stopClientExecutor();
}

void PointGoalGlobalPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  const std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  if (!node_) {
    throw nav2_core::PlannerException("PointGoal planner parent node expired.");
  }
  name_ = name;
  costmap_ros_ = std::move(costmap_ros);
  tf_ = std::move(tf);
  costmap_ = costmap_ros_->getCostmap();
  global_frame_ = costmap_ros_->getGlobalFrameID();

  const auto declare = [this](const std::string & parameter, const auto & value) {
      nav2_util::declare_parameter_if_not_declared(
        node_, name_ + "." + parameter, rclcpp::ParameterValue(value));
    };
  declare("robot_frame", costmap_ros_->getBaseFrameID());
  declare("inference_service", std::string("/vint/pointgoal_inference"));
  declare("inference_timeout_sec", 0.5);
  declare(
    "tail_planner_plugin", std::string(
      "nav2_theta_star_planner/ThetaStarPlanner"));
  declare("minimum_prefix_points", 2);
  declare("maximum_prefix_start_error_m", 0.25);
  declare("minimum_goal_progress_m", -0.25);
  declare("collision_check_step_m", 0.025);
  declare("collision_cost_threshold", 253);
  declare("duplicate_tolerance_m", 0.05);

  std::string tail_plugin;
  node_->get_parameter(name_ + ".robot_frame", robot_frame_);
  node_->get_parameter(name_ + ".inference_service", inference_service_);
  node_->get_parameter(name_ + ".inference_timeout_sec", inference_timeout_sec_);
  node_->get_parameter(name_ + ".tail_planner_plugin", tail_plugin);
  node_->get_parameter(name_ + ".minimum_prefix_points", minimum_prefix_points_);
  node_->get_parameter(
    name_ + ".maximum_prefix_start_error_m", maximum_prefix_start_error_m_);
  node_->get_parameter(
    name_ + ".minimum_goal_progress_m", minimum_goal_progress_m_);
  node_->get_parameter(
    name_ + ".collision_check_step_m", collision_check_step_m_);
  node_->get_parameter(
    name_ + ".collision_cost_threshold", collision_cost_threshold_);
  node_->get_parameter(
    name_ + ".duplicate_tolerance_m", duplicate_tolerance_m_);

  if (inference_timeout_sec_ <= 0.0 || minimum_prefix_points_ < 2 ||
    maximum_prefix_start_error_m_ < 0.0 || collision_check_step_m_ <= 0.0 ||
    collision_cost_threshold_ < 1 || collision_cost_threshold_ > 255 ||
    duplicate_tolerance_m_ < 0.0)
  {
    throw nav2_core::PlannerException("Invalid PointGoal planner parameters.");
  }

  tail_loader_ = std::make_unique<
    pluginlib::ClassLoader<nav2_core::GlobalPlanner>>(
    "nav2_core", "nav2_core::GlobalPlanner");
  try {
    tail_planner_ = tail_loader_->createSharedInstance(tail_plugin);
    tail_planner_->configure(parent, name_ + ".tail", tf_, costmap_ros_);
  } catch (const pluginlib::PluginlibException & error) {
    throw nav2_core::PlannerException(
            "Unable to load PointGoal tail planner: " + std::string(error.what()));
  }

  auto options = rclcpp::NodeOptions();
  options.context(node_->get_node_base_interface()->get_context());
  // Do not apply planner_server's process-wide __node remap to this private
  // client node; duplicate node names break rosout publisher ownership.
  options.use_global_arguments(false);
  client_node_ = std::make_shared<rclcpp::Node>(
    name_ + "_vint_client", node_->get_namespace(), options);
  inference_client_ = client_node_->create_client<nav_msgs::srv::GetPlan>(
    inference_service_);
  client_executor_ =
    std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  client_executor_->add_node(client_node_);
  client_thread_ = std::thread([this]() {client_executor_->spin();});

  RCLCPP_INFO(
    node_->get_logger(),
    "Configured %s with PointGoal service %s and tail %s.",
    name_.c_str(), inference_service_.c_str(), tail_plugin.c_str());
}

void PointGoalGlobalPlanner::cleanup()
{
  active_ = false;
  if (tail_planner_) {
    tail_planner_->cleanup();
  }
  tail_planner_.reset();
  tail_loader_.reset();
  stopClientExecutor();
  inference_client_.reset();
  client_node_.reset();
  costmap_ = nullptr;
  costmap_ros_.reset();
  tf_.reset();
  node_.reset();
}

void PointGoalGlobalPlanner::activate()
{
  if (tail_planner_) {
    tail_planner_->activate();
  }
  active_ = true;
}

void PointGoalGlobalPlanner::deactivate()
{
  active_ = false;
  if (tail_planner_) {
    tail_planner_->deactivate();
  }
}

void PointGoalGlobalPlanner::stopClientExecutor()
{
  if (client_executor_) {
    client_executor_->cancel();
  }
  if (client_thread_.joinable()) {
    client_thread_.join();
  }
  client_executor_.reset();
}

nav_msgs::msg::Path PointGoalGlobalPlanner::requestPrefix(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  if (!inference_client_->wait_for_service(0s)) {
    throw nav2_core::PlannerException(
            "PointGoal inference service is unavailable.");
  }
  const double start_yaw = tf2::getYaw(start.pose.orientation);
  const double cosine = std::cos(start_yaw);
  const double sine = std::sin(start_yaw);
  const double dx = goal.pose.position.x - start.pose.position.x;
  const double dy = goal.pose.position.y - start.pose.position.y;

  auto request = std::make_shared<nav_msgs::srv::GetPlan::Request>();
  request->start.header.frame_id = robot_frame_;
  request->start.header.stamp = start.header.stamp;
  request->start.pose.orientation.w = 1.0;
  request->goal.header.frame_id = robot_frame_;
  request->goal.header.stamp = start.header.stamp;
  request->goal.pose.position.x = cosine * dx + sine * dy;
  request->goal.pose.position.y = -sine * dx + cosine * dy;
  request->goal.pose.orientation.w = 1.0;

  auto future = inference_client_->async_send_request(request);
  const auto status = future.wait_for(
    std::chrono::duration<double>(inference_timeout_sec_));
  if (status != std::future_status::ready) {
    inference_client_->remove_pending_request(future);
    throw nav2_core::PlannerException("PointGoal inference timed out.");
  }
  const auto response = future.get();
  if (response->plan.poses.size() < static_cast<std::size_t>(minimum_prefix_points_)) {
    throw nav2_core::PlannerException(
            "PointGoal inference returned an empty or undersized prefix.");
  }
  if (response->plan.header.frame_id != robot_frame_) {
    throw nav2_core::PlannerException(
            "PointGoal inference returned an unexpected path frame.");
  }
  geometry_msgs::msg::PoseStamped origin;
  origin.pose.orientation.w = 1.0;
  if (poseDistance(response->plan.poses.front(), origin) >
    maximum_prefix_start_error_m_)
  {
    throw nav2_core::PlannerException(
            "PointGoal path does not begin near the robot origin.");
  }
  return response->plan;
}

bool PointGoalGlobalPlanner::poseIsCollisionFree(
  const double x, const double y, const double yaw,
  nav2_costmap_2d::FootprintCollisionChecker<
    nav2_costmap_2d::Costmap2D *> & checker) const
{
  unsigned int map_x = 0;
  unsigned int map_y = 0;
  if (!costmap_->worldToMap(x, y, map_x, map_y)) {
    return false;
  }
  if (costmap_->getCost(map_x, map_y) >= collision_cost_threshold_) {
    return false;
  }
  const double footprint_cost = checker.footprintCostAtPose(
    x, y, yaw, costmap_ros_->getRobotFootprint());
  return footprint_cost >= 0.0 && footprint_cost < collision_cost_threshold_;
}

nav_msgs::msg::Path PointGoalGlobalPlanner::collisionFreePrefix(
  const nav_msgs::msg::Path & path) const
{
  nav2_costmap_2d::FootprintCollisionChecker<
    nav2_costmap_2d::Costmap2D *> checker(costmap_);
  return vint_nav2_plugins::collisionFreePrefix(
    path, collision_check_step_m_,
    [this, &checker](const double x, const double y, const double yaw) {
      return poseIsCollisionFree(x, y, yaw, checker);
    });
}

nav_msgs::msg::Path PointGoalGlobalPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  if (!active_) {
    throw nav2_core::PlannerException("PointGoal planner is inactive.");
  }
  if (start.header.frame_id != global_frame_ || goal.header.frame_id != global_frame_) {
    throw nav2_core::PlannerException(
            "PointGoal planner start and goal must use the global costmap frame.");
  }

  const auto local_prefix = requestPrefix(start, goal);
  auto predicted_prefix = transformRobotPathToMap(
    local_prefix, start, global_frame_);
  predicted_prefix.header.stamp = node_->now();
  for (auto & pose : predicted_prefix.poses) {
    pose.header = predicted_prefix.header;
  }
  auto prefix = collisionFreePrefix(predicted_prefix);
  if (prefix.poses.size() < static_cast<std::size_t>(minimum_prefix_points_)) {
    throw nav2_core::PlannerException(
            "PointGoal prefix has too few collision-free poses.");
  }
  if (prefix.poses.size() < predicted_prefix.poses.size()) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000,
      "Truncated PointGoal prefix from %zu to %zu poses at the first collision.",
      predicted_prefix.poses.size(), prefix.poses.size());
  }
  const double start_distance = poseDistance(start, goal);
  const double join_distance = poseDistance(prefix.poses.back(), goal);
  if (start_distance - join_distance < minimum_goal_progress_m_) {
    throw nav2_core::PlannerException(
            "PointGoal prefix does not make sufficient progress toward the goal.");
  }

  auto join = prefix.poses.back();
  join.header = prefix.header;
  const auto tail = tail_planner_->createPlan(join, goal);
  if (tail.poses.empty()) {
    throw nav2_core::PlannerException(
            "Conventional tail planner could not reach the final goal.");
  }
  auto complete = joinPaths(prefix, tail, duplicate_tolerance_m_);
  complete.header.frame_id = global_frame_;
  complete.header.stamp = node_->now();
  for (auto & pose : complete.poses) {
    pose.header = complete.header;
  }
  complete.poses.back().pose.orientation = goal.pose.orientation;
  RCLCPP_DEBUG(
    node_->get_logger(),
    "PointGoal plan: %.2f m learned prefix, %.2f m total path.",
    pathLength(prefix), pathLength(complete));
  return complete;
}

}  // namespace vint_nav2_plugins

PLUGINLIB_EXPORT_CLASS(
  vint_nav2_plugins::PointGoalGlobalPlanner,
  nav2_core::GlobalPlanner)
