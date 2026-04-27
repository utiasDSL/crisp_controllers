#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include <controller_interface/controller_interface.hpp>
#include <crisp_controllers/utils/ros2_version.hpp>

#if ROS2_VERSION_ABOVE_HUMBLE
#include <crisp_controllers/twist_broadcaster_parameters.hpp>
#else
#include <twist_broadcaster_parameters.hpp>
#endif

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>

#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace crisp_controllers {

class TwistBroadcaster : public controller_interface::ControllerInterface {
public:
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;
  controller_interface::return_type
  update(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

private:
  std::shared_ptr<twist_broadcaster::ParamListener> params_listener_;
  twist_broadcaster::Params params_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::TwistStamped>>
    realtime_twist_publisher_;

  std::string end_effector_frame_;
  int end_effector_frame_id;

  pinocchio::Model model_;
  pinocchio::Data data_;

  /** @brief Allowed type of joints **/
  const std::unordered_set<std::basic_string<char>> allowed_joint_types = {
    "JointModelRX",
    "JointModelRY",
    "JointModelRZ",
    "JointModelRevoluteUnaligned",
    "JointModelRUBX",
    "JointModelRUBY",
    "JointModelRUBZ",
  };
  /** @brief Continous joint types that should be considered separetly. **/
  const std::unordered_set<std::basic_string<char>> continous_joint_types = {
    "JointModelRUBX", "JointModelRUBY", "JointModelRUBZ"};

  Eigen::VectorXd q;
  Eigen::VectorXd q_dot;

  rclcpp::Duration publish_elapsed_{0, 0};
  rclcpp::Duration publish_interval_{0, 0};
};

}  // namespace crisp_controllers
