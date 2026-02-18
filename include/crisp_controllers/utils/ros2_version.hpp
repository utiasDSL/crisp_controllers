#pragma once

// ROS2 version checking utilities for conditional compilation
// Used to handle API differences between ROS2 distributions

#include <rclcpp/version.h>

// ============================================================================
// ROS2 Distro version mapping (RCLCPP_VERSION_MAJOR)
// Humble  = 16
// Iron    = 21
// Jazzy   = 28
// Kilted  = 29+
// ============================================================================

#define ROS2_DISTRO_HUMBLE 16
#define ROS2_DISTRO_IRON 21
#define ROS2_DISTRO_JAZZY 28

// Convenience macros for version checking (strictly greater than)
#define ROS2_VERSION_ABOVE_HUMBLE (RCLCPP_VERSION_MAJOR > ROS2_DISTRO_HUMBLE)
#define ROS2_VERSION_ABOVE_IRON (RCLCPP_VERSION_MAJOR > ROS2_DISTRO_IRON)
#define ROS2_VERSION_ABOVE_JAZZY (RCLCPP_VERSION_MAJOR > ROS2_DISTRO_JAZZY)

// ============================================================================
// Feature-specific macros
// ============================================================================

// REGISTER_ROS2_CONTROL_INTROSPECTION was added in ros2_control 4.27.0
// Available in Jazzy (backported) and Kilted/Rolling
#if __has_include(<hardware_interface/version.h>)
#include <hardware_interface/version.h>
#if HARDWARE_INTERFACE_VERSION_GTE(4, 27, 0)
#define HAS_ROS2_CONTROL_INTROSPECTION 1
#else
#define HAS_ROS2_CONTROL_INTROSPECTION 0
#endif
#else
#define HAS_ROS2_CONTROL_INTROSPECTION 0
#endif
