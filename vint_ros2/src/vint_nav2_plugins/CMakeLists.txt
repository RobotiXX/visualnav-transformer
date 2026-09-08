cmake_minimum_required(VERSION 3.8)
project(vint_nav2_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_theta_star_planner REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(dependencies
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav2_theta_star_planner
  nav2_util
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  tf2
  tf2_geometry_msgs
  tf2_ros
)

add_library(${PROJECT_NAME} SHARED
  src/controller_geometry.cpp
  src/image_goal_controller.cpp
  src/planner_geometry.cpp
  src/pointgoal_global_planner.cpp
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

pluginlib_export_plugin_description_file(nav2_core vint_nav2_plugins.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(FILES vint_nav2_plugins.xml README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_planner_geometry test/test_planner_geometry.cpp)
  if(TARGET test_planner_geometry)
    target_link_libraries(test_planner_geometry ${PROJECT_NAME})
    ament_target_dependencies(test_planner_geometry geometry_msgs nav_msgs tf2)
  endif()
  ament_add_gtest(test_controller_geometry test/test_controller_geometry.cpp)
  if(TARGET test_controller_geometry)
    target_link_libraries(test_controller_geometry ${PROJECT_NAME})
    ament_target_dependencies(test_controller_geometry geometry_msgs)
  endif()
  ament_add_gtest(test_plugin_loading test/test_plugin_loading.cpp)
  if(TARGET test_plugin_loading)
    ament_target_dependencies(test_plugin_loading nav2_core pluginlib rclcpp)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(
    test/test_image_goal_controller_launch.py
    TIMEOUT 60
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})
ament_package()
