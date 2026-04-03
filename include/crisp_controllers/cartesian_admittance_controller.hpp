#pragma once

/**
 * @file cartesian_admittance_controller.hpp
 * @brief Cartesian admittance controller with impedance outer loop for compliant force-based interaction
 *
 * Maintains an internal virtual mass-spring-damper (MSD) state that evolves based on external
 * F/T sensor readings. The MSD output becomes the desired pose for an impedance (Cartesian)
 * outer control loop. This is a separate class that contains equivalent impedance logic
 * to CartesianController, not derived from it.
 */

#include <memory>
#include <string>
#include <unordered_set>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <rclcpp/rclcpp.hpp>

#include <crisp_controllers/utils/ros2_version.hpp>

#if ROS2_VERSION_ABOVE_HUMBLE
#include <crisp_controllers/cartesian_admittance_controller_parameters.hpp>
#else
#include <cartesian_admittance_controller_parameters.hpp>
#endif

#include <sensor_msgs/msg/joint_state.hpp>
#include "realtime_tools/realtime_buffer.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace crisp_controllers {

/**
 * @brief Cartesian admittance controller with impedance outer loop
 *
 * This controller implements a two-layer control scheme:
 * 1. Inner admittance loop: A virtual mass-spring-damper system driven by external F/T sensor
 *    readings, producing a desired pose that tracks the commanded pose while being compliant
 *    to external forces.
 * 2. Outer impedance loop: Standard Cartesian impedance (or OSC) control that tracks the
 *    inner loop's output pose, with all the same compensation torques as CartesianController.
 */
class CartesianAdmittanceController : public controller_interface::ControllerInterface {
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
  update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

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
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Activate the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of activation
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Deactivate the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of deactivation
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

private:
  // ---- Subscriptions (same as CartesianController) ----

  /** @brief Subscription for target pose messages */
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  /** @brief Subscription for target joint state messages */
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  /** @brief Subscription for target wrench messages */
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;
  /** @brief Subscription for variable impedance stiffness messages */
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr stiffness_sub_;

  /** @brief Subscription for F/T sensor messages */
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_sensor_sub_;
  /** @brief Subscription for variable admittance stiffness messages */
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr adm_stiffness_sub_;

  /** @brief Flag to indicate if multiple publishers detected */
  bool multiple_publishers_detected_;
  /** @brief Expected maximum number of publishers per topic */
  size_t max_allowed_publishers_;

  // ---- Methods ----

  /**
   * @brief Set the impedance stiffness and damping matrices based on parameters
   */
  void setStiffnessAndDamping();

  /**
   * @brief Set the admittance mass, stiffness, and damping matrices from parameters
   */
  void setAdmittanceParameters();

  /**
   * @brief Get the current state of the robot from hardware interfaces and update internal variables
   * @param initialize If set to true, initialize the exponential moving average filter with the current state
   */
  void updateCurrentState(bool initialize = false);

  /**
   * @brief Reads the target pose in realtime loop from the buffer and parses it
   */
  void parse_target_pose_();

  /**
   * @brief Reads the target joint in realtime loop from the buffer and parses it
   */
  void parse_target_joint_();

  /**
   * @brief Reads the target wrench in realtime loop from the buffer and parses it
   */
  void parse_target_wrench_();

  /**
   * @brief Reads the target impedance stiffness in realtime loop from the buffer and parses it
   */
  void parse_target_stiffness_();

  /**
   * @brief Reads the F/T sensor data from the realtime buffer and updates the internal wrench
   */
  void parse_ft_sensor_();

  /**
   * @brief Reads the target admittance stiffness from the realtime buffer
   */
  void parse_target_adm_stiffness_();

  // ---- Flags for new data ----
  bool new_target_pose_;
  bool new_target_joint_;
  bool new_target_wrench_;
  bool new_target_stiffness_ = false;
  bool use_topic_stiffness_ = false;
  bool new_ft_sensor_ = false;
  bool new_target_adm_stiffness_ = false;
  bool use_topic_adm_stiffness_ = false;

  // ---- Realtime buffers (same as CartesianController) ----
  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::PoseStamped>>
    target_pose_buffer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
    target_joint_buffer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::WrenchStamped>>
    target_wrench_buffer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::Float64MultiArray>>
    target_stiffness_buffer_;

  // ---- Admittance-specific realtime buffers ----
  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::WrenchStamped>>
    ft_sensor_buffer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::Float64MultiArray>>
    target_adm_stiffness_buffer_;

  // ---- Target / desired state ----

  /** @brief Target position in Cartesian space */
  Eigen::Vector3d target_position_;
  /** @brief Target orientation as quaternion */
  Eigen::Quaterniond target_orientation_;
  /** @brief Target wrench in task space */
  Eigen::VectorXd target_wrench_;
  /** @brief Desired target position in Cartesian space after applying filtering */
  Eigen::Vector3d desired_position_;
  /** @brief Desired target orientation as quaternion after applying filtering */
  Eigen::Quaterniond desired_orientation_;

  // ---- Parameters ----

  /** @brief Parameter listener for dynamic parameter updates */
  std::shared_ptr<cartesian_admittance_controller::ParamListener> params_listener_;
  /** @brief Current parameter values */
  cartesian_admittance_controller::Params params_;

  // ---- Pinocchio model ----

  /** @brief Frame ID of the end effector in the robot model */
  int end_effector_frame_id;
  /** @brief Frame ID of the F/T sensor measurement frame in the robot model */
  int ft_sensor_frame_id;
  /** @brief Pinocchio robot model */
  pinocchio::Model model_;
  /** @brief Pinocchio data for computations */
  pinocchio::Data data_;

  // ---- Impedance outer loop matrices ----

  /** @brief Cartesian stiffness matrix (6x6) */
  Eigen::MatrixXd stiffness = Eigen::MatrixXd::Zero(6, 6);
  /** @brief Cartesian damping matrix (6x6) */
  Eigen::MatrixXd damping = Eigen::MatrixXd::Zero(6, 6);
  /** @brief Topic-provided stiffness matrix (6x6 diagonal) */
  Eigen::Matrix<double, 6, 6> topic_stiffness_ = Eigen::Matrix<double, 6, 6>::Zero();

  /** @brief Nullspace stiffness matrix for posture control */
  Eigen::MatrixXd nullspace_stiffness;
  /** @brief Nullspace damping matrix for posture control */
  Eigen::MatrixXd nullspace_damping;

  // ---- Admittance inner loop state ----

  /** @brief Internal MSD pose state */
  pinocchio::SE3 inner_SE3_;
  /** @brief Internal MSD velocity state (6D: linear + angular) */
  Eigen::Vector<double, 6> inner_motion_;
  /** @brief First-cycle initialization flag for admittance state */
  bool admittance_initialized_;
  /** @brief Admittance virtual mass matrix (6x6 diagonal) */
  Eigen::Matrix<double, 6, 6> adm_mass_;
  /** @brief Inverse of admittance virtual mass matrix */
  Eigen::Matrix<double, 6, 6> adm_mass_inv_;
  /** @brief Admittance stiffness matrix (6x6 diagonal) */
  Eigen::Matrix<double, 6, 6> adm_stiffness_;
  /** @brief Admittance damping matrix (6x6 diagonal) */
  Eigen::Matrix<double, 6, 6> adm_damping_;
  /** @brief F/T sensor reading (6D wrench) */
  Eigen::VectorXd ft_wrench_;
  /** @brief Topic-provided admittance stiffness (6x6 diagonal) */
  Eigen::Matrix<double, 6, 6> topic_adm_stiffness_;

  // ---- Joint state vectors ----

  /** @brief Current joint positions with dimension nv */
  Eigen::VectorXd q;
  /** @brief Current joint positions with dimension nq (may differ for continuous joints) */
  Eigen::VectorXd q_pin;
  /** @brief Current joint velocities */
  Eigen::VectorXd dq;
  /** @brief Reference joint positions for posture task */
  Eigen::VectorXd q_ref;
  /** @brief Reference joint velocities */
  Eigen::VectorXd dq_ref;
  /** @brief Target joint positions for posture task */
  Eigen::VectorXd q_target;

  /** @brief Previously computed torque */
  Eigen::VectorXd tau_previous;

  /** @brief Current end effector pose */
  pinocchio::SE3 end_effector_pose;
  /** @brief End effector Jacobian matrix */
  pinocchio::Data::Matrix6x J;
  /** @brief End effector Jacobian matrix pseudoinverse */
  Eigen::MatrixXd J_pinv;
  /** @brief Joint-space identity matrix */
  Eigen::MatrixXd Id_nv;

  /** @brief Friction parameters 1 of size nv */
  Eigen::VectorXd fp1;
  /** @brief Friction parameters 2 of size nv */
  Eigen::VectorXd fp2;
  /** @brief Friction parameters 3 of size nv */
  Eigen::VectorXd fp3;

  /** @brief Allowed type of joints */
  const std::unordered_set<std::basic_string<char>> allowed_joint_types = {
    "JointModelRX",
    "JointModelRY",
    "JointModelRZ",
    "JointModelRevoluteUnaligned",
    "JointModelRUBX",
    "JointModelRUBY",
    "JointModelRUBZ",
  };
  /** @brief Continuous joint types that should be considered separately */
  const std::unordered_set<std::basic_string<char>> continous_joint_types = {
    "JointModelRUBX", "JointModelRUBY", "JointModelRUBZ"};

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

  /** @brief Inverse of the manipulator joint mass projected in Cartesian space (6x6) */
  Eigen::Matrix<double, 6, 6> Mx_inv = Eigen::Matrix<double, 6, 6>::Zero();
  /** @brief the manipulator joint mass projected in Cartesian space (6x6) */
  Eigen::Matrix<double, 6, 6> Mx = Eigen::Matrix<double, 6, 6>::Zero();

  /**
   * @brief Log debug information based on parameter settings
   * @param time Current time for throttling logs
   */
  void log_debug_info(const rclcpp::Time & time);

  /**
   * @brief Check publisher count for a specific topic
   * @param topic_name Name of the topic to check
   * @return true if publisher count is safe (<=1), false otherwise
   */
  bool check_topic_publisher_count(const std::string & topic_name);
};

}  // namespace crisp_controllers
