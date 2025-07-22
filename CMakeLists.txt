cmake_minimum_required(VERSION 3.5)
project(crisp_controllers)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache")
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()


# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Enable precompiled headers
option(USE_PRECOMPILED_HEADERS "Use precompiled headers" ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(CHECK_TIDY "Adds clang-tidy tests" OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(controller_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(generate_parameter_library REQUIRED)

generate_parameter_library(
  cartesian_impedance_controller_parameters
  src/cartesian_impedance_controller.yaml
)

generate_parameter_library(
  pose_broadcaster_parameters
  src/pose_broadcaster.yaml
)

generate_parameter_library(
  torque_feedback_controller_parameters
  src/torque_feedback_controller.yaml
)

generate_parameter_library(
  wrench_feedback_controller_parameters
  src/wrench_feedback_controller.yaml
)

add_library(
        ${PROJECT_NAME}
        SHARED
        src/cartesian_impedance_controller.cpp
        src/pose_broadcaster.cpp
        src/torque_feedback_controller.cpp
        src/wrench_feedback_controller.cpp
        src/utils/kinematics_utils.cpp
        src/utils/nullspace_control.cpp
        src/utils/torque_utils.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(${PROJECT_NAME} 
        PRIVATE
        include/crisp_controllers/pch.hpp
    )
endif()

# Link parameter libraries to the main library
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    cartesian_impedance_controller_parameters
    pose_broadcaster_parameters
    torque_feedback_controller_parameters
    wrench_feedback_controller_parameters
)


ament_target_dependencies(${PROJECT_NAME}
  PUBLIC
controller_interface
hardware_interface
pluginlib
rclcpp
rclcpp_lifecycle
pinocchio
generate_parameter_library
realtime_tools
)

pluginlib_export_plugin_description_file(
        controller_interface crisp_controllers.xml)

install(
        TARGETS
        ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

install(
        DIRECTORY include/
        DESTINATION include
)

ament_export_include_directories(
        include
)
ament_export_libraries(
        ${PROJECT_NAME}
)
ament_export_dependencies(
        controller_interface
        pluginlib
        rclcpp
        rclcpp_lifecycle
        hardware_interface
        Eigen3
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_torque_rate_saturation
    tests/test_torque_rate_saturation.cpp
  )
  target_include_directories(test_torque_rate_saturation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_torque_rate_saturation
    Eigen3::Eigen
  )
  
  ament_add_gtest(test_pseudo_inverse
    tests/test_pseudo_inverse.cpp
  )
  target_include_directories(test_pseudo_inverse PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_pseudo_inverse
    Eigen3::Eigen
  )

  ament_add_gtest(test_filters
    tests/test_filters.cpp
  )
  target_include_directories(test_filters PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_filters
    Eigen3::Eigen
  )

  # Test for torque feedback controller (simplified to avoid linking issues)
  ament_add_gtest(test_torque_feedback_controller
    tests/test_torque_feedback_controller.cpp
  )
  target_include_directories(test_torque_feedback_controller PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_torque_feedback_controller
    Eigen3::Eigen
  )

  # Test for feedback control core utilities
  ament_add_gtest(test_feedback_control_core
    tests/test_feedback_control_core.cpp
  )
  target_include_directories(test_feedback_control_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_feedback_control_core
    Eigen3::Eigen
  )

  # Test for nullspace controller utilities
  ament_add_gtest(test_nullspace_controller
    tests/test_nullspace_controller.cpp
  )
  target_include_directories(test_nullspace_controller PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_nullspace_controller
    Eigen3::Eigen
  )

  # Test for wrench feedback controller
  ament_add_gtest(test_wrench_feedback_controller
    tests/test_wrench_feedback_controller.cpp
  )
  target_include_directories(test_wrench_feedback_controller PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_wrench_feedback_controller
    Eigen3::Eigen
    geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
  )
endif()

ament_package()
