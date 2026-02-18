#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include <controller_interface/controller_interface.hpp>
#include <crisp_controllers/utils/ros2_version.hpp>

#if ROS2_VERSION_ABOVE_HUMBLE
#include <crisp_controllers/torque_feedback_controller_parameters.hpp>
#else
#include <torque_feedback_controller_parameters.hpp>
#endif
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace crisp_controllers {

/**
 * @brief Torque feedback controller for robotic manipulators
 * 
 * This controller implements torque-based feedback control that responds to external
 * torques applied to the robot. It uses a proportional-derivative (PD) control scheme
 * with friction compensation to generate joint torque commands.
 * 
 * The controller applies a threshold to the external torque vector magnitude to
 * filter out noise and small disturbances. Only when the total torque exceeds
 * the threshold are the torques used for feedback control.
 * 
 * Control law: tau_d = -kp_fb * tau_ext_filtered - kd * dq + tau_friction + tau_nullspace
 * 
 * Where:
 * - tau_ext_filtered: External torques after threshold filtering
 * - kp_fb: Proportional feedback gain
 * - kd: Derivative (damping) gain
 * - dq: Joint velocities
 * - tau_friction: Friction compensation torques
 * - tau_nullspace: Nullspace control torques to maintain initial joint positions
 */
class TorqueFeedbackController : public controller_interface::ControllerInterface {
public:
  /**
   * @brief Configure command interfaces for the controller
   * @return Configuration specifying joint effort interfaces
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  /**
   * @brief Configure state interfaces for the controller
   * @return Configuration specifying joint position, velocity, and effort interfaces
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  /**
   * @brief Main control update loop
   * 
   * Reads joint states, applies torque threshold filtering, computes control
   * torques using PD control with friction compensation, and sends commands
   * to the robot joints.
   * 
   * @param time Current time
   * @param period Control period
   * @return Control execution status
   */
  controller_interface::return_type
  update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  /**
   * @brief Initialize controller parameters and subscribers
   * @return Initialization status
   */
  CallbackReturn on_init() override;

  /**
   * @brief Configure controller (set collision behavior if using real hardware)
   * @param previous_state Previous lifecycle state
   * @return Configuration status
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Activate controller and initialize joint states
   * @param previous_state Previous lifecycle state
   * @return Activation status
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Deactivate controller
   * @param previous_state Previous lifecycle state
   * @return Deactivation status
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

private:
  /// Parameter listener for dynamic parameter updates
  std::shared_ptr<torque_feedback_controller::ParamListener> params_listener_;
  /// Current controller parameters
  torque_feedback_controller::Params params_;

  /// Subscriber for external torque commands
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;

  /// Publisher for commanded wrench
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_pub_;

  /// Timer for 200Hz wrench publishing
  rclcpp::TimerBase::SharedPtr wrench_timer_;

  /**
   * @brief Callback for target joint state messages
   *
   * Updates the external torques from joint state effort values.
   * Only effort values are used from the incoming message.
   * @param msg Joint state message containing external torques in effort field
   */
  void target_joint_callback_(const sensor_msgs::msg::JointState::SharedPtr msg);

  /**
   * @brief Callback for wrench publishing timer
   *
   * Computes and publishes the commanded wrench using the Jacobian transpose
   */
  void publish_wrench_callback_();

  /// Names of controlled joints
  std::vector<std::string> joint_names_;
  /// Number of controlled joints
  int num_joints_;
  /// Current joint positions
  Eigen::VectorXd q_;
  /// Current joint velocities
  Eigen::VectorXd dq_;
  /// Current joint torques
  Eigen::VectorXd tau_;
  /// Commanded joint torques (for wrench computation)
  Eigen::VectorXd tau_commanded_;

  /// External torques received from subscriber
  Eigen::VectorXd tau_ext_;

  /// Initial joint positions recorded on activation (nullspace target)
  Eigen::VectorXd q_init_;

  /// Nullspace weights (computed from parameters)
  Eigen::VectorXd nullspace_weights_;

  /// Friction parameters as Eigen vectors
  Eigen::VectorXd friction_fp1_;
  Eigen::VectorXd friction_fp2_;
  Eigen::VectorXd friction_fp3_;

  /// Nullspace projection matrix
  Eigen::MatrixXd nullspace_projection_;

  /// Pinocchio model and data for dynamics computations
  pinocchio::Model model_;
  pinocchio::Data data_;

  /// End-effector frame ID for jacobian computation
  pinocchio::FrameIndex end_effector_frame_id_;

  /// Jacobian matrix for nullspace projection
  Eigen::MatrixXd J_;
};

}  // namespace crisp_controllers
