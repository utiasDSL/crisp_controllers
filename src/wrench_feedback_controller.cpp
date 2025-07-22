#include <crisp_controllers/wrench_feedback_controller.hpp>
#include <crisp_controllers/utils/friction_model.hpp>
#include <crisp_controllers/utils/kinematics_utils.hpp>
#include <crisp_controllers/utils/nullspace_control.hpp>
#include <crisp_controllers/utils/torque_utils.hpp>

#include <cmath>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>

namespace crisp_controllers {

controller_interface::InterfaceConfiguration
WrenchFeedbackController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto &joint_name : joint_names_) {
    config.names.push_back(joint_name + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
WrenchFeedbackController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto &joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
  }
  for (const auto &joint_name : joint_names_) {
    config.names.push_back(joint_name + "/velocity");
  }
  for (const auto &joint_name : joint_names_) {
    config.names.push_back(joint_name + "/effort");
  }
  return config;
}

controller_interface::return_type
WrenchFeedbackController::update(const rclcpp::Time & /*time*/,
                                 const rclcpp::Duration & /*period*/) {
  // Update joint states
  for (int i = 0; i < num_joints_; i++) {
    q_[i] = state_interfaces_[i].get_value();
    dq_[i] = state_interfaces_[num_joints_ + i].get_value();
  }

  // Compute forward kinematics and jacobian using utility
  kinematics_utils::computeForwardKinematicsAndJacobian(
      model_, data_, q_, dq_, J_, end_effector_frame_id_);

  // Apply threshold to external wrench based on vector magnitude
  Eigen::VectorXd wrench_ext_thresholded = torque_utils::applyTorqueThreshold(
      wrench_ext_, params_.wrench_threshold);

  // Convert wrench to joint torques using Jacobian transpose: tau = J^T * wrench
  Eigen::VectorXd tau_from_wrench = J_.transpose() * wrench_ext_thresholded;

  // Create nullspace parameters structure
  nullspace_control::NullspaceParams ns_params;
  ns_params.stiffness = params_.nullspace.stiffness;
  ns_params.damping = params_.nullspace.damping;
  ns_params.max_tau = params_.nullspace.max_tau;
  ns_params.regularization = params_.nullspace.regularization;
  ns_params.projector_type = params_.nullspace.projector_type;

  // Compute nullspace control using utility
  Eigen::VectorXd tau_nullspace = nullspace_control::computeNullspaceControl(
      q_, dq_, q_init_, J_, model_, data_, ns_params, nullspace_weights_,
      get_node()->get_logger());

  // Compute feedback and friction torques
  auto tau_d = -params_.k_fb * tau_from_wrench - params_.kd * dq_;
  auto tau_f = get_friction(dq_, friction_fp1_, friction_fp2_, friction_fp3_);

  // Total commanded torques
  tau_commanded_ = tau_d + tau_f + tau_nullspace;

  // Send commands to joints
  for (int i = 0; i < num_joints_; i++) {
    command_interfaces_[i].set_value(tau_commanded_[i]);
  }

  // Update dynamic parameters
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();
  
  // Update nullspace weights
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    nullspace_weights_[i] = params_.nullspace.weights.joints_map.at(joint_names_[i]).value;
  }
  
  return controller_interface::return_type::OK;
}

CallbackReturn WrenchFeedbackController::on_init() {
  // Initialize parameters
  params_listener_ =
      std::make_shared<wrench_feedback_controller::ParamListener>(get_node());
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();

  // Set basic parameters
  joint_names_ = params_.joints;
  num_joints_ = joint_names_.size();

  // Initialize state vectors
  q_ = Eigen::VectorXd::Zero(num_joints_);
  dq_ = Eigen::VectorXd::Zero(num_joints_);
  tau_commanded_ = Eigen::VectorXd::Zero(num_joints_);
  q_init_ = Eigen::VectorXd::Zero(num_joints_);

  // Initialize 6D wrench vector (3 forces + 3 torques)
  wrench_ext_ = Eigen::VectorXd::Zero(6);

  nullspace_weights_ = Eigen::VectorXd::Ones(num_joints_);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    nullspace_weights_[i] = params_.nullspace.weights.joints_map.at(joint_names_[i]).value;
  }

  // Initialize friction parameters
  friction_fp1_ = Eigen::Map<const Eigen::VectorXd>(params_.friction.fp1.data(), params_.friction.fp1.size());
  friction_fp2_ = Eigen::Map<const Eigen::VectorXd>(params_.friction.fp2.data(), params_.friction.fp2.size());
  friction_fp3_ = Eigen::Map<const Eigen::VectorXd>(params_.friction.fp3.data(), params_.friction.fp3.size());

  // Create wrench subscriber
  wrench_sub_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
      params_.wrench_source_topic,
      rclcpp::QoS(1).best_effort().keep_last(1).durability_volatile(),
      std::bind(&WrenchFeedbackController::target_wrench_callback_, this,
                std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn WrenchFeedbackController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  
  // Get robot description for pinocchio model
  auto parameters_client = std::make_shared<rclcpp::AsyncParametersClient>(
      get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();

  std::string robot_description_;
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to get robot_description parameter.");
    return CallbackReturn::ERROR;
  }

  // Setup pinocchio model using utility
  if (!kinematics_utils::setupPinocchioModel(
          robot_description_, joint_names_, model_, data_,
          get_node()->get_logger())) {
    return CallbackReturn::ERROR;
  }

  // Validate joint types using utility
  if (!kinematics_utils::validateJointTypes(model_, get_node()->get_logger())) {
    return CallbackReturn::ERROR;
  }

  // Get end-effector frame ID
  end_effector_frame_id_ = model_.getFrameId(params_.end_effector_frame);
  
  // Initialize jacobian matrix (6x nv for 6D wrench)
  J_ = Eigen::MatrixXd::Zero(6, model_.nv);

  return CallbackReturn::SUCCESS;
}

CallbackReturn WrenchFeedbackController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  // Initialize joint states and record initial positions
  for (int i = 0; i < num_joints_; i++) {
    q_[i] = state_interfaces_[i].get_value();
    dq_[i] = state_interfaces_[num_joints_ + i].get_value();
    q_init_[i] = q_[i];
  }

  // Initialize external wrench to zero
  wrench_ext_.setZero();

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn WrenchFeedbackController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

void WrenchFeedbackController::target_wrench_callback_(
    const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
  // Convert geometry_msgs::WrenchStamped to 6D Eigen vector
  wrench_ext_[0] = msg->wrench.force.x;
  wrench_ext_[1] = msg->wrench.force.y;
  wrench_ext_[2] = msg->wrench.force.z;
  wrench_ext_[3] = msg->wrench.torque.x;
  wrench_ext_[4] = msg->wrench.torque.y;
  wrench_ext_[5] = msg->wrench.torque.z;
}

} // namespace crisp_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(crisp_controllers::WrenchFeedbackController,
                       controller_interface::ControllerInterface)
