#pragma once

/**
 * @file cartesian_controller.hpp
 * @brief Cartesian controller implementation for robot manipulation (supports impedance and OSC)
 * @author Your Organization
 */

#include <Eigen/Dense>

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>
#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <rclcpp/rclcpp.hpp>

#include "realtime_tools/realtime_buffer.hpp"
#include <cartesian_impedance_controller_parameters.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>
#include <unordered_set>

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace crisp_controllers {

/**
 * @brief Controller implementing Cartesian control
 *
 * This controller implements Cartesian control for robotic manipulation,
 * supporting both impedance control and operational space control (OSC),
 * allowing for compliant interaction with the environment while maintaining
 * desired position and orientation targets.
 */
class CartesianController
    : public controller_interface::ControllerInterface {
public:
  /**
   * @brief Get the command interface configuration
   * @return Interface configuration specifying required command interfaces
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  /**
   * @brief Get the state interface configuration
   * @return Interface configuration specifying required state interfaces
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  /**
   * @brief Update function called periodically
   * @param time Current time
   * @param period Time since last update
   * @return Success/failure of update
   */
  controller_interface::return_type
  update(const rclcpp::Time &time, const rclcpp::Duration &period) override;

  /**
   * @brief Initialize the controller
   * @return Success/failure of initialization
   */
  CallbackReturn on_init() override;

  /**
   * @brief Configure the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of configuration
   */
  CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;

  /**
   * @brief Activate the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of activation
   */
  CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;

  /**
   * @brief Deactivate the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of deactivation
   */
  CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  /*CartesianImpedanceController();*/

private:
  /** @brief Subscription for target pose messages */
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  /** @brief Subscription for target joint state messages */
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  /** @brief Subscription for target wrench messages */
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;

  /** @brief Flag to indicate if multiple publishers detected */
  bool multiple_publishers_detected_;

  /** @brief Expected maximum number of publishers per topic */
  size_t max_allowed_publishers_;

  /**
   * @brief Set the stiffness and damping matrices based on parameters
   */
  void setStiffnessAndDamping();

  /**
   * @brief Reads the target pose in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_pose_();

  /**
   * @brief Reads the target joint in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_joint_();

  /**
   * @brief Reads the target wrench in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_wrench_();

  bool new_target_pose_;
  bool new_target_joint_;
  bool new_target_wrench_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::PoseStamped>>
    target_pose_buffer_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
    target_joint_buffer_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::WrenchStamped>>
    target_wrench_buffer_;

  /** @brief Target position in Cartesian space */
  Eigen::Vector3d target_position_;
  /** @brief Target orientation as quaternion */
  Eigen::Quaterniond target_orientation_;
  /** @brief Target wrench in task space */
  Eigen::VectorXd target_wrench_;
  /** @brief Combined target pose as SE3 transformation */
  pinocchio::SE3 target_pose_;

  /** @brief Parameter listener for dynamic parameter updates */
  std::shared_ptr<cartesian_impedance_controller::ParamListener>
      params_listener_;
  /** @brief Current parameter values */
  cartesian_impedance_controller::Params params_;

  /** @brief Frame ID of the end effector in the robot model */
  int end_effector_frame_id;

  /** @brief Pinocchio robot model */
  pinocchio::Model model_;
  /** @brief Pinocchio data for computations */
  pinocchio::Data data_;

  /** @brief Cartesian stiffness matrix (6x6) */
  Eigen::MatrixXd stiffness = Eigen::MatrixXd::Zero(6, 6);
  /** @brief Cartesian damping matrix (6x6) */
  Eigen::MatrixXd damping = Eigen::MatrixXd::Zero(6, 6);

  /** @brief Nullspace stiffness matrix for posture control */
  Eigen::MatrixXd nullspace_stiffness;
  /** @brief Nullspace damping matrix for posture control */
  Eigen::MatrixXd nullspace_damping;

  /** @brief Current joint positions with dimension nv. */
  Eigen::VectorXd q;
  /** @brief Current joint positions with dimension nq.
   This is size might be different than the actuated dimension of the joint type is different! 
   Check https://github.com/stack-of-tasks/pinocchio/issues/1127
  */
  Eigen::VectorXd q_pin;
  /** @brief Current joint velocities */
  Eigen::VectorXd dq;
  /** @brief Reference joint positions for posture task */
  Eigen::VectorXd q_ref;
  /** @brief Reference joint velocities */
  Eigen::VectorXd dq_ref;

  /** @brief Previously computed torque */
  Eigen::VectorXd tau_previous;

  /** @brief Current end effector pose */
  pinocchio::SE3 end_effector_pose;
  /** @brief End effector Jacobian matrix */
  pinocchio::Data::Matrix6x J;

  /** @brief Friction parameters 1 of size nv */
  Eigen::VectorXd fp1;
  /** @brief Friction parameters 2 of size nv */
  Eigen::VectorXd fp2;
  /** @brief Friction parameters 3 of size nv */
  Eigen::VectorXd fp3;


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
  const std::unordered_set<std::basic_string<char>> continous_joint_types =
    {"JointModelRUBX", "JointModelRUBY", "JointModelRUBZ"};

  /** @brief Maximum allowed delta values for error clipping */
  Eigen::VectorXd max_delta_ = Eigen::VectorXd::Zero(6);

  /** @brief Nullspace projection matrix */
  Eigen::MatrixXd nullspace_projection;

  /** @brief Task space error vector (6x1) */
  Eigen::VectorXd error = Eigen::VectorXd::Zero(6);

  /** @brief Task space control torque */
  Eigen::VectorXd tau_task;
  /** @brief Joint limit avoidance torque */
  Eigen::VectorXd tau_joint_limits;
  /** @brief Secondary task torque before nullspace projection */
  Eigen::VectorXd tau_secondary;
  /** @brief Nullspace projected secondary task torque */
  Eigen::VectorXd tau_nullspace;
  /** @brief Friction compensation torque */
  Eigen::VectorXd tau_friction;
  /** @brief Coriolis compensation torque */
  Eigen::VectorXd tau_coriolis;
  /** @brief Gravity compensation torque */
  Eigen::VectorXd tau_gravity;
  /** @brief External wrench compensation torque */
  Eigen::VectorXd tau_wrench;
  /** @brief Final desired torque command */
  Eigen::VectorXd tau_d;

  /**
   * @brief Log debug information based on parameter settings
   * @param time Current time for throttling logs
   */
  void log_debug_info(const rclcpp::Time& time);

  /**
   * @brief Check publisher count for a specific topic
   * @param topic_name Name of the topic to check
   * @return true if publisher count is safe (<=1), false otherwise
   */
  bool check_topic_publisher_count(const std::string& topic_name);

  /**
   * @brief End-effector pose expressed in the base frame (base_frame)
   */
  pinocchio::SE3 end_effector_pose_b;
  /**
   * @brief Pinocchio frame ID for params_.base_frame
   */
  int base_frame_id;
};

} // namespace crisp_controllers
