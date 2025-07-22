#pragma once
#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <crisp_controllers/wrench_feedback_controller_parameters.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace crisp_controllers {

/**
 * @brief Wrench feedback controller for robotic manipulators
 * 
 * This controller implements Cartesian wrench-based feedback control that responds to external
 * wrenches applied to the robot end-effector. It converts wrench commands to joint torques
 * using the Jacobian transpose and applies PD control with friction compensation.
 * 
 * The controller applies a threshold to the external wrench vector magnitude to
 * filter out noise and small disturbances. Only when the total wrench exceeds
 * the threshold are the wrenches used for feedback control.
 * 
 * Control law: tau_d = -J^T * kp_fb * wrench_ext_filtered - kd * dq + tau_friction + tau_nullspace
 * 
 * Where:
 * - wrench_ext_filtered: External wrenches after threshold filtering
 * - J^T: Jacobian transpose for wrench-to-torque conversion
 * - kp_fb: Proportional feedback gain
 * - kd: Derivative (damping) gain
 * - dq: Joint velocities
 * - tau_friction: Friction compensation torques
 * - tau_nullspace: Nullspace control torques to maintain initial joint positions
 */
class WrenchFeedbackController
    : public controller_interface::ControllerInterface {
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
   * Reads joint states, applies wrench threshold filtering, computes control
   * torques using PD control with friction compensation, and sends commands
   * to the robot joints.
   * 
   * @param time Current time
   * @param period Control period
   * @return Control execution status
   */
  controller_interface::return_type
  update(const rclcpp::Time &time, const rclcpp::Duration &period) override;
  
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
  CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  
  /**
   * @brief Activate controller and initialize joint states
   * @param previous_state Previous lifecycle state
   * @return Activation status
   */
  CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;
  
  /**
   * @brief Deactivate controller
   * @param previous_state Previous lifecycle state
   * @return Deactivation status
   */
  CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

private:
  /// Parameter listener for dynamic parameter updates
  std::shared_ptr<wrench_feedback_controller::ParamListener> params_listener_;
  /// Current controller parameters
  wrench_feedback_controller::Params params_;

  /// Subscriber for external wrench commands
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;

  /**
   * @brief Callback for target wrench messages
   *
   * Updates the external wrenches from wrench stamped messages.
   * @param msg Wrench stamped message containing external wrench
   */
  void
  target_wrench_callback_(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);

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
  /// Commanded joint torques
  Eigen::VectorXd tau_commanded_;

  /// External wrench received from subscriber (6D: force + torque)
  Eigen::VectorXd wrench_ext_;
  
  /// Initial joint positions recorded on activation (nullspace target)
  Eigen::VectorXd q_init_;
  
  /// Nullspace weights (computed from parameters)
  Eigen::VectorXd nullspace_weights_;
  
  /// Friction parameters as Eigen vectors
  Eigen::VectorXd friction_fp1_;
  Eigen::VectorXd friction_fp2_;
  Eigen::VectorXd friction_fp3_;
  
  /// Pinocchio model and data for dynamics computations
  pinocchio::Model model_;
  pinocchio::Data data_;
  
  /// End-effector frame ID for jacobian computation
  pinocchio::FrameIndex end_effector_frame_id_;
  
  /// Jacobian matrix for wrench-to-torque conversion
  Eigen::MatrixXd J_;
};

} // namespace crisp_controllers