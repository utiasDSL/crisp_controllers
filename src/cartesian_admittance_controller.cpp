
#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/src/Core/Matrix.h>  // NOLINT(build/include_order)
#include <fmt/format.h>
#include <controller_interface/controller_interface_base.hpp>
#include <rclcpp/logging.hpp>

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/fwd.hpp>
#include <pinocchio/algorithm/utils/force.hpp>

#include "pinocchio/algorithm/model.hpp"

#include <std_msgs/msg/float64_multi_array.hpp>

#include <crisp_controllers/cartesian_admittance_controller.hpp>
#include <crisp_controllers/pch.hpp>
#include <crisp_controllers/utils/friction_model.hpp>
#include <crisp_controllers/utils/joint_limits.hpp>
#include <crisp_controllers/utils/pseudo_inverse.hpp>
#include "crisp_controllers/utils/fiters.hpp"
#include "crisp_controllers/utils/torque_rate_saturation.hpp"

#include "crisp_controllers/utils/fiters.hpp"
#include "crisp_controllers/utils/ros2_version.hpp"
#include "crisp_controllers/utils/torque_rate_saturation.hpp"

#if HAS_ROS2_CONTROL_INTROSPECTION
#include <hardware_interface/introspection.hpp>
#endif

namespace crisp_controllers {

controller_interface::InterfaceConfiguration
CartesianAdmittanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint_name : params_.joints) {
    config.names.push_back(joint_name + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianAdmittanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto & joint_name : params_.joints) {
    config.names.push_back(joint_name + "/position");
  }
  for (const auto & joint_name : params_.joints) {
    config.names.push_back(joint_name + "/velocity");
  }
  return config;
}

controller_interface::return_type
CartesianAdmittanceController::update(const rclcpp::Time & time, const rclcpp::Duration & period) {

  // 1. Update current state information with EMA filtered values
  updateCurrentState();

  // 2. Check if new targets available (pose, joint, wrench -- same as CartesianController)
  if (new_target_pose_) {
    parse_target_pose_();
    new_target_pose_ = false;
  }
  if (new_target_joint_) {
    parse_target_joint_();
    new_target_joint_ = false;
  }
  if (new_target_wrench_) {
    parse_target_wrench_();
    new_target_wrench_ = false;
  }
  if (new_target_stiffness_) {
    parse_target_stiffness_();
    new_target_stiffness_ = false;
    setStiffnessAndDamping();
  }

  // 3. Parse F/T sensor data
  if (new_ft_sensor_) {
    parse_ft_sensor_();
    new_ft_sensor_ = false;
  }

  // 4. Parse variable admittance stiffness
  if (new_target_adm_stiffness_) {
    parse_target_adm_stiffness_();
    new_target_adm_stiffness_ = false;
    setAdmittanceParameters();
  }

  // 5. Forward kinematics
  pinocchio::forwardKinematics(model_, data_, q_pin, dq);
  pinocchio::updateFramePlacements(model_, data_);
  end_effector_pose = data_.oMf[end_effector_frame_id];

  // 6. Initialize admittance state on first cycle
  if (!admittance_initialized_) {
    inner_SE3_ = end_effector_pose;
    inner_motion_.setZero();
    admittance_initialized_ = true;
  }

  // 7. Compute admittance target pose from desired (same EMA filtering as CartesianController)
  desired_position_ =
    exponential_moving_average(desired_position_, target_position_, params_.filter.target_pose);
  desired_orientation_ =
    target_orientation_.slerp(params_.filter.target_pose, desired_orientation_);

  // 8. Compute SE3 error between inner state and desired pose (on manifold)
  // Build desired SE3 from filtered position and orientation
  pinocchio::SE3 desired_SE3;
  desired_SE3.translation() = desired_position_;
  desired_SE3.rotation() = desired_orientation_.toRotationMatrix();

  // Position error: desired - inner
  Eigen::Vector3d adm_pos_error = desired_SE3.translation() - inner_SE3_.translation();

  // Orientation error: SO(3) logarithmic map (world frame)
  Eigen::Vector3d adm_rot_error =
    pinocchio::log3(desired_SE3.rotation() * inner_SE3_.rotation().transpose());

  Eigen::Vector<double, 6> adm_error;
  adm_error << adm_pos_error, adm_rot_error;

  // 9. Transform F/T wrench from sensor (LOCAL) frame to LOCAL_WORLD_ALIGNED frame
  pinocchio::SE3 ft_sensor_pose = data_.oMf[ft_sensor_frame_id];
  pinocchio::Force ft_local(ft_wrench_.head(3), ft_wrench_.tail(3));
  pinocchio::Force ft_world = pinocchio::Force::Zero();
  pinocchio::changeReferenceFrame(
    ft_sensor_pose, ft_local, pinocchio::LOCAL, pinocchio::LOCAL_WORLD_ALIGNED, ft_world);
  Eigen::Vector<double, 6> ft_wrench_world = ft_world.toVector();

  // 10. Admittance dynamics: accel = M_inv * (F_ext - D * vel + K * error)
  // Spring term +K*(desired - inner) is a restoring force toward the desired pose
  Eigen::Matrix<double, 6, 6> K_adm = use_topic_adm_stiffness_ ? topic_adm_stiffness_ : adm_stiffness_;
  Eigen::Vector<double, 6> adm_force = ft_wrench_world - adm_damping_ * inner_motion_ + K_adm * adm_error;
  Eigen::Vector<double, 6> accel = adm_mass_inv_ * adm_force;

  // 10. Semi-implicit Euler integration
  double dt = period.seconds();
  inner_motion_ += accel * dt;
  // Update inner_SE3_ using exponential map
  pinocchio::SE3 delta = pinocchio::exp6(pinocchio::Motion(inner_motion_ * dt));
  inner_SE3_ = delta * inner_SE3_;

  // 11. Use inner_SE3_ as the desired for impedance outer loop
  // Compute impedance error (same as CartesianController but using inner_SE3_ as target)
  if (params_.use_local_jacobian) {
    error.head(3) = end_effector_pose.rotation().transpose() *
      (inner_SE3_.translation() - end_effector_pose.translation());
    error.tail(3) =
      pinocchio::log3(end_effector_pose.rotation().transpose() * inner_SE3_.rotation());
  } else {
    error.head(3) = inner_SE3_.translation() - end_effector_pose.translation();
    error.tail(3) =
      pinocchio::log3(inner_SE3_.rotation() * end_effector_pose.rotation().transpose());
  }

  // 12. Error clipping (same as CartesianController)
  if (params_.limit_error) {
    max_delta_ << params_.task.error_clip.x, params_.task.error_clip.y, params_.task.error_clip.z,
      params_.task.error_clip.rx, params_.task.error_clip.ry, params_.task.error_clip.rz;
    error = error.cwiseMax(-max_delta_).cwiseMin(max_delta_);
  }

  // 13. Jacobian, pseudo-inverse, nullspace, Mx computation (same as CartesianController)
  J.setZero();
  auto reference_frame = params_.use_local_jacobian
    ? pinocchio::ReferenceFrame::LOCAL
    : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  pinocchio::computeFrameJacobian(model_, data_, q_pin, end_effector_frame_id, reference_frame, J);

  J_pinv = pseudo_inverse(J, params_.nullspace.regularization);
  if (params_.nullspace.projector_type == "dynamic" || params_.use_operational_space) {
    pinocchio::computeMinverse(model_, data_, q_pin);
    data_.Minv.triangularView<Eigen::StrictlyLower>() =
      data_.Minv.transpose().triangularView<Eigen::StrictlyLower>();
    Mx_inv = J * data_.Minv * J.transpose();
    Mx = pseudo_inverse(Mx_inv, params_.operational_space_regularization);
  }

  if (params_.nullspace.projector_type == "dynamic") {
    auto J_bar = data_.Minv * J.transpose() * Mx;
    nullspace_projection = Id_nv - J.transpose() * J_bar.transpose();
  } else if (params_.nullspace.projector_type == "kinematic") {
    nullspace_projection = Id_nv - J_pinv * J;
  } else if (params_.nullspace.projector_type == "none") {
    nullspace_projection = Eigen::MatrixXd::Identity(model_.nv, model_.nv);
  } else {
    RCLCPP_ERROR_STREAM_ONCE(
      get_node()->get_logger(),
      "Unknown nullspace projector type: " << params_.nullspace.projector_type);
    return controller_interface::return_type::ERROR;
  }

  // 14. Task torque (impedance): tau_task = J^T * (K * error - D * J * dq) or J^T * Mx * (...)
  if (params_.use_operational_space) {
    tau_task << J.transpose() * Mx * (stiffness * error - damping * (J * dq));
  } else {
    tau_task << J.transpose() * (stiffness * error - damping * (J * dq));
  }

  // 15. All compensation torques (same as CartesianController)
  if (model_.nq != model_.nv) {
    tau_joint_limits = Eigen::VectorXd::Zero(model_.nv);
  } else {
    tau_joint_limits = get_joint_limit_torque(
      q,
      model_.lowerPositionLimit,
      model_.upperPositionLimit,
      params_.joint_limit_repulsion.safe_range,
      params_.joint_limit_repulsion.max_torque);
  }

  tau_secondary << nullspace_stiffness * (q_ref - q) + nullspace_damping * (dq_ref - dq);

  tau_nullspace << nullspace_projection * tau_secondary;
  tau_nullspace =
    tau_nullspace.cwiseMin(params_.nullspace.max_tau).cwiseMax(-params_.nullspace.max_tau);

  tau_friction =
    params_.use_friction ? get_friction(dq, fp1, fp2, fp3) : Eigen::VectorXd::Zero(model_.nv);

  if (params_.use_coriolis_compensation) {
    pinocchio::computeAllTerms(model_, data_, q_pin, dq);
    tau_coriolis = pinocchio::computeCoriolisMatrix(model_, data_, q_pin, dq) * dq;
  } else {
    tau_coriolis = Eigen::VectorXd::Zero(model_.nv);
  }

  tau_gravity = params_.use_gravity_compensation
    ? pinocchio::computeGeneralizedGravity(model_, data_, q_pin)
    : Eigen::VectorXd::Zero(model_.nv);

  tau_wrench << J.transpose() * target_wrench_;

  tau_d << tau_task + tau_nullspace + tau_friction + tau_coriolis + tau_gravity + tau_joint_limits +
      tau_wrench;

  // 16. Torque saturation and filtering
  if (params_.limit_torques) {
    tau_d = saturateTorqueRate(tau_d, tau_previous, params_.max_delta_tau);
  }
  tau_d = exponential_moving_average(tau_d, tau_previous, params_.filter.output_torque);

  // 17. Command interfaces
  if (!params_.stop_commands) {
    for (size_t i = 0; i < params_.joints.size(); ++i) {
#if ROS2_VERSION_ABOVE_HUMBLE
      (void)command_interfaces_[i].set_value(tau_d[i]);
#else
      command_interfaces_[i].set_value(tau_d[i]);
#endif
    }
  }

  tau_previous = tau_d;

  // 18. Refresh parameters
  params_listener_->refresh_dynamic_parameters();
  if (params_listener_->is_old(params_)) {
    params_ = params_listener_->get_params();
    setStiffnessAndDamping();
    setAdmittanceParameters();
  }

  log_debug_info(time);

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianAdmittanceController::on_init() {
  params_listener_ = std::make_shared<cartesian_admittance_controller::ParamListener>(get_node());
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();

  return CallbackReturn::SUCCESS;
}

CallbackReturn
CartesianAdmittanceController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {
  auto parameters_client =
    std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();

  std::string robot_description_;
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
    return CallbackReturn::ERROR;
  }

  pinocchio::Model raw_model_;
  pinocchio::urdf::buildModelFromXML(robot_description_, raw_model_);

  RCLCPP_INFO(get_node()->get_logger(), "Checking available joints in model:");
  for (int joint_id = 0; joint_id < raw_model_.njoints; joint_id++) {
    RCLCPP_INFO_STREAM(
      get_node()->get_logger(),
      "Joint " << joint_id << " with name " << raw_model_.names[joint_id] << " is of type "
               << raw_model_.joints[joint_id].shortname());
  }

  // Check that the passed joints exist in the kinematic tree
  for (auto & joint : params_.joints) {
    if (!raw_model_.existJointName(joint)) {
      RCLCPP_ERROR_STREAM(
        get_node()->get_logger(),
        "Failed to configure because " << joint
                                       << " is not part of the kinematic tree but it "
                                          "has been passed in the parameters.");
      return CallbackReturn::ERROR;
    }
  }
  RCLCPP_INFO(
    get_node()->get_logger(),
    "All joints passed in the parameters exist in the kinematic tree "
    "of the URDF.");
  RCLCPP_INFO_STREAM(
    get_node()->get_logger(), "Removing the rest of the joints that are not used: ");
  // Fix all joints that are not referenced in the tree
  std::vector<pinocchio::JointIndex> list_of_joints_to_lock_by_id;
  for (auto & joint : raw_model_.names) {
    if (
      std::find(params_.joints.begin(), params_.joints.end(), joint) == params_.joints.end() &&
      joint != "universe") {
      RCLCPP_INFO_STREAM(
        get_node()->get_logger(), "Joint " << joint << " is not used, removing it from the model.");
      list_of_joints_to_lock_by_id.push_back(raw_model_.getJointId(joint));
    }
  }

  Eigen::VectorXd q_locked = Eigen::VectorXd::Zero(raw_model_.nq);
  model_ = pinocchio::buildReducedModel(raw_model_, list_of_joints_to_lock_by_id, q_locked);
  data_ = pinocchio::Data(model_);

  for (int joint_id = 0; joint_id < model_.njoints; joint_id++) {
    if (model_.names[joint_id] == "universe") {
      continue;
    }
    if (!allowed_joint_types.count(model_.joints[joint_id].shortname())) {
      RCLCPP_ERROR_STREAM(
        get_node()->get_logger(),
        "Joint type " << model_.joints[joint_id].shortname() << " is unsupported ("
                      << model_.names[joint_id]
                      << "), only revolute/continous like joints can be used.");
      return CallbackReturn::ERROR;
    }
  }

  // Preallocate the matrices and vectors that will be used in the control loop
  end_effector_frame_id = model_.getFrameId(params_.end_effector_frame);
  if (params_.ft_sensor.frame.empty()) {
    ft_sensor_frame_id = end_effector_frame_id;
    RCLCPP_WARN(get_node()->get_logger(),
      "ft_sensor.frame is not set, using end_effector_frame for F/T wrench transformation. "
      "Set ft_sensor.frame to the actual sensor measurement frame for correct results.");
  } else {
    if (!model_.existFrame(params_.ft_sensor.frame)) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "F/T sensor frame '%s' does not exist in the robot model. Please check the "
        "ft_sensor.frame parameter and the URDF frames.",
        params_.ft_sensor.frame.c_str());
      return CallbackReturn::ERROR;
    }
    ft_sensor_frame_id = model_.getFrameId(params_.ft_sensor.frame);
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Using F/T sensor frame: %s (id=%d)",
      params_.ft_sensor.frame.c_str(),
      ft_sensor_frame_id);
  }
  q = Eigen::VectorXd::Zero(model_.nv);
  q_pin = Eigen::VectorXd::Zero(model_.nq);
  dq = Eigen::VectorXd::Zero(model_.nv);
  q_ref = Eigen::VectorXd::Zero(model_.nv);
  q_target = Eigen::VectorXd::Zero(model_.nv);
  dq_ref = Eigen::VectorXd::Zero(model_.nv);
  tau_previous = Eigen::VectorXd::Zero(model_.nv);
  J = Eigen::MatrixXd::Zero(6, model_.nv);
  J_pinv = Eigen::MatrixXd::Zero(model_.nv, 6);
  Id_nv = Eigen::MatrixXd::Identity(model_.nv, model_.nv);

  // Map the friction parameters to Eigen vectors
  fp1 = Eigen::Map<Eigen::VectorXd>(params_.friction.fp1.data(), model_.nv);
  fp2 = Eigen::Map<Eigen::VectorXd>(params_.friction.fp2.data(), model_.nv);
  fp3 = Eigen::Map<Eigen::VectorXd>(params_.friction.fp3.data(), model_.nv);

  nullspace_stiffness = Eigen::MatrixXd::Zero(model_.nv, model_.nv);
  nullspace_damping = Eigen::MatrixXd::Zero(model_.nv, model_.nv);

  setStiffnessAndDamping();
  setAdmittanceParameters();

  new_target_pose_ = false;
  new_target_joint_ = false;
  new_target_wrench_ = false;
  new_target_stiffness_ = false;
  use_topic_stiffness_ = false;
  new_ft_sensor_ = false;
  new_target_adm_stiffness_ = false;
  use_topic_adm_stiffness_ = false;
  admittance_initialized_ = false;

  multiple_publishers_detected_ = false;
  max_allowed_publishers_ = 1;

  // Initialize admittance inner loop state
  inner_SE3_ = pinocchio::SE3::Identity();
  inner_motion_.setZero();
  ft_wrench_ = Eigen::VectorXd::Zero(6);
  topic_adm_stiffness_ = Eigen::Matrix<double, 6, 6>::Zero();

  // --- Subscriptions (same as CartesianController) ---
  auto target_pose_callback =
    [this](const std::shared_ptr<geometry_msgs::msg::PoseStamped> msg) -> void {
    if (!check_topic_publisher_count("target_pose")) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring target_pose message due to multiple publishers detected!");
      return;
    }
    target_pose_buffer_.writeFromNonRT(msg);
    new_target_pose_ = true;
  };

  auto target_joint_callback =
    [this](const std::shared_ptr<sensor_msgs::msg::JointState> msg) -> void {
    if (!check_topic_publisher_count("target_joint")) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring target_joint message due to multiple publishers detected!");
      return;
    }
    target_joint_buffer_.writeFromNonRT(msg);
    new_target_joint_ = true;
  };

  auto target_wrench_callback =
    [this](const std::shared_ptr<geometry_msgs::msg::WrenchStamped> msg) -> void {
    if (!check_topic_publisher_count("target_wrench")) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring target_wrench message due to multiple publishers detected!");
      return;
    }
    target_wrench_buffer_.writeFromNonRT(msg);
    new_target_wrench_ = true;
  };

  pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
    "target_pose", rclcpp::QoS(1), target_pose_callback);

  joint_sub_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
    "target_joint", rclcpp::QoS(1), target_joint_callback);

  wrench_sub_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
    "target_wrench", rclcpp::QoS(1), target_wrench_callback);

  if (params_.variable_stiffness.enabled) {
    auto target_stiffness_callback =
      [this](const std::shared_ptr<std_msgs::msg::Float64MultiArray> msg) -> void {
      if (!check_topic_publisher_count(params_.variable_stiffness.topic)) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(),
          *get_node()->get_clock(),
          1000,
          "Ignoring target_stiffness message due to multiple publishers detected!");
        return;
      }
      target_stiffness_buffer_.writeFromNonRT(msg);
      new_target_stiffness_ = true;
    };

    stiffness_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      params_.variable_stiffness.topic, rclcpp::QoS(1), target_stiffness_callback);

    RCLCPP_INFO(get_node()->get_logger(), "Variable impedance stiffness enabled on topic: %s",
      params_.variable_stiffness.topic.c_str());
  }

  // --- Admittance-specific subscriptions ---

  // F/T sensor subscription
  auto ft_sensor_callback =
    [this](const std::shared_ptr<geometry_msgs::msg::WrenchStamped> msg) -> void {
    ft_sensor_buffer_.writeFromNonRT(msg);
    new_ft_sensor_ = true;
  };

  ft_sensor_sub_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
    params_.ft_sensor.topic, rclcpp::SensorDataQoS(), ft_sensor_callback);

  RCLCPP_INFO(get_node()->get_logger(), "F/T sensor subscription on topic: %s",
    params_.ft_sensor.topic.c_str());

  // Variable admittance stiffness subscription
  if (params_.variable_admittance_stiffness.enabled) {
    auto target_adm_stiffness_callback =
      [this](const std::shared_ptr<std_msgs::msg::Float64MultiArray> msg) -> void {
      target_adm_stiffness_buffer_.writeFromNonRT(msg);
      new_target_adm_stiffness_ = true;
    };

    adm_stiffness_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      params_.variable_admittance_stiffness.topic, rclcpp::QoS(1), target_adm_stiffness_callback);

    RCLCPP_INFO(get_node()->get_logger(), "Variable admittance stiffness enabled on topic: %s",
      params_.variable_admittance_stiffness.topic.c_str());
  }

  // Initialize all control vectors with appropriate dimensions
  tau_task = Eigen::VectorXd::Zero(model_.nv);
  tau_joint_limits = Eigen::VectorXd::Zero(model_.nv);
  tau_secondary = Eigen::VectorXd::Zero(model_.nv);
  tau_nullspace = Eigen::VectorXd::Zero(model_.nv);
  tau_friction = Eigen::VectorXd::Zero(model_.nv);
  tau_coriolis = Eigen::VectorXd::Zero(model_.nv);
  tau_gravity = Eigen::VectorXd::Zero(model_.nv);
  tau_wrench = Eigen::VectorXd::Zero(model_.nv);
  tau_d = Eigen::VectorXd::Zero(model_.nv);

  // Initialize target state vectors
  target_position_ = Eigen::Vector3d::Zero();
  target_orientation_ = Eigen::Quaterniond::Identity();
  target_wrench_ = Eigen::VectorXd::Zero(6);
  desired_position_ = Eigen::Vector3d::Zero();
  desired_orientation_ = Eigen::Quaterniond::Identity();

  // Initialize error vector
  error = Eigen::VectorXd::Zero(6);
  max_delta_ = Eigen::VectorXd::Zero(6);

  // Initialize nullspace projection matrix
  nullspace_projection = Eigen::MatrixXd::Identity(model_.nv, model_.nv);

#if HAS_ROS2_CONTROL_INTROSPECTION
  if (params_.enable_introspection) {
    RCLCPP_INFO(get_node()->get_logger(), "Enabling ROS2 Control Introspection for debugging.");
    this->enable_introspection(true);
    for (int i = 0; i < tau_task.size(); ++i) {
      REGISTER_ROS2_CONTROL_INTROSPECTION("tau_task_" + std::to_string(i), &tau_task[i]);
      REGISTER_ROS2_CONTROL_INTROSPECTION("tau_desired_" + std::to_string(i), &tau_d[i]);
    }
    for (int i = 0; i < error.size(); ++i) {
      REGISTER_ROS2_CONTROL_INTROSPECTION("error_" + std::to_string(i), &error[i]);
    }
    REGISTER_ROS2_CONTROL_INTROSPECTION("target_position_x", &target_position_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("target_position_y", &target_position_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("target_position_z", &target_position_[2]);
    for (int i = 0; i < target_orientation_.coeffs().size(); ++i) {
      REGISTER_ROS2_CONTROL_INTROSPECTION(
        "target_orientation_" + std::to_string(i), &target_orientation_.coeffs()[i]);
    }
    for (int i = 0; i < q_target.size(); ++i) {
      REGISTER_ROS2_CONTROL_INTROSPECTION("q_target_" + std::to_string(i), &q_target[i]);
      REGISTER_ROS2_CONTROL_INTROSPECTION("q_ref_" + std::to_string(i), &q_ref[i]);
    }
  } else {
    RCLCPP_INFO(get_node()->get_logger(), "ROS2 Control Introspection is disabled.");
  }
#endif

  RCLCPP_INFO(get_node()->get_logger(), "Admittance controller configured successfully.");

  return CallbackReturn::SUCCESS;
}

void CartesianAdmittanceController::setStiffnessAndDamping() {
  if (use_topic_stiffness_) {
    stiffness = topic_stiffness_;
  } else {
    stiffness.setZero();
    stiffness.diagonal() << params_.task.k_pos_x, params_.task.k_pos_y, params_.task.k_pos_z,
      params_.task.k_rot_x, params_.task.k_rot_y, params_.task.k_rot_z;
  }

  // Clamp stiffness to [0, max_stiffness]
  const double max_k_trans = params_.variable_max_impedance_stiffness.translational;
  const double max_k_rot = params_.variable_max_impedance_stiffness.rotational;
  for (int i = 0; i < 3; ++i) {
    if (stiffness(i, i) < 0.0 || stiffness(i, i) > max_k_trans) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Translational stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, stiffness(i, i), max_k_trans);
      stiffness(i, i) = std::clamp(stiffness(i, i), 0.0, max_k_trans);
    }
  }
  for (int i = 3; i < 6; ++i) {
    if (stiffness(i, i) < 0.0 || stiffness(i, i) > max_k_rot) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Rotational stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, stiffness(i, i), max_k_rot);
      stiffness(i, i) = std::clamp(stiffness(i, i), 0.0, max_k_rot);
    }
  }

  damping.setZero();
  // For each axis, use explicit damping if > 0, otherwise compute from stiffness
  damping.diagonal()
    << (params_.task.d_pos_x > 0 ? params_.task.d_pos_x : 2.0 * std::sqrt(params_.task.k_pos_x)),
    (params_.task.d_pos_y > 0 ? params_.task.d_pos_y : 2.0 * std::sqrt(params_.task.k_pos_y)),
    (params_.task.d_pos_z > 0 ? params_.task.d_pos_z : 2.0 * std::sqrt(params_.task.k_pos_z)),
    (params_.task.d_rot_x > 0 ? params_.task.d_rot_x : 2.0 * std::sqrt(params_.task.k_rot_x)),
    (params_.task.d_rot_y > 0 ? params_.task.d_rot_y : 2.0 * std::sqrt(params_.task.k_rot_y)),
    (params_.task.d_rot_z > 0 ? params_.task.d_rot_z : 2.0 * std::sqrt(params_.task.k_rot_z));

  nullspace_stiffness.setZero();
  nullspace_damping.setZero();

  auto weights = Eigen::VectorXd(model_.nv);
  for (size_t i = 0; i < params_.joints.size(); ++i) {
    weights[i] = params_.nullspace.weights.joints_map.at(params_.joints.at(i)).value;
  }
  nullspace_stiffness.diagonal() << params_.nullspace.stiffness * weights;
  nullspace_damping.diagonal() << 2.0 * nullspace_stiffness.diagonal().cwiseSqrt();

  if (params_.nullspace.damping >= 0.0) {
    nullspace_damping.diagonal() = params_.nullspace.damping * weights;
  } else {
    nullspace_damping.diagonal() = 2.0 * nullspace_stiffness.diagonal().cwiseSqrt();
  }
}

void CartesianAdmittanceController::setAdmittanceParameters() {
  adm_mass_.setZero();
  adm_mass_.diagonal() << params_.admittance.mass_x, params_.admittance.mass_y,
    params_.admittance.mass_z, params_.admittance.mass_rx, params_.admittance.mass_ry,
    params_.admittance.mass_rz;
  adm_mass_inv_ = adm_mass_.inverse();

  if (use_topic_adm_stiffness_) {
    adm_stiffness_ = topic_adm_stiffness_;
  } else {
    adm_stiffness_.setZero();
    adm_stiffness_.diagonal() << params_.admittance.stiffness_x, params_.admittance.stiffness_y,
      params_.admittance.stiffness_z, params_.admittance.stiffness_rx,
      params_.admittance.stiffness_ry, params_.admittance.stiffness_rz;
  }

  // Clamp admittance stiffness to [0, max_admittance_stiffness]
  const double max_ak_trans = params_.variable_max_admittance_stiffness.translational;
  const double max_ak_rot = params_.variable_max_admittance_stiffness.rotational;
  for (int i = 0; i < 3; ++i) {
    if (adm_stiffness_(i, i) < 0.0 || adm_stiffness_(i, i) > max_ak_trans) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Admittance translational stiffness[%d]=%.1f out of [0, %.1f], clamping.",
        i, adm_stiffness_(i, i), max_ak_trans);
      adm_stiffness_(i, i) = std::clamp(adm_stiffness_(i, i), 0.0, max_ak_trans);
    }
  }
  for (int i = 3; i < 6; ++i) {
    if (adm_stiffness_(i, i) < 0.0 || adm_stiffness_(i, i) > max_ak_rot) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Admittance rotational stiffness[%d]=%.1f out of [0, %.1f], clamping.",
        i, adm_stiffness_(i, i), max_ak_rot);
      adm_stiffness_(i, i) = std::clamp(adm_stiffness_(i, i), 0.0, max_ak_rot);
    }
  }

  adm_damping_.setZero();
  adm_damping_.diagonal() << params_.admittance.damping_x, params_.admittance.damping_y,
    params_.admittance.damping_z, params_.admittance.damping_rx, params_.admittance.damping_ry,
    params_.admittance.damping_rz;
}

void CartesianAdmittanceController::updateCurrentState(bool initialize) {
  auto num_joints = params_.joints.size();
  for (size_t i = 0; i < num_joints; i++) {
    auto joint_name = params_.joints[i];
    auto joint_id = model_.getJointId(joint_name);
    auto joint = model_.joints[joint_id];

#if ROS2_VERSION_ABOVE_HUMBLE
    double q_meas = state_interfaces_[i].get_optional().value_or(q[i]);
    double dq_meas = state_interfaces_[num_joints + i].get_optional().value_or(dq[i]);
#else
    double q_meas = state_interfaces_[i].get_value();
    double dq_meas = state_interfaces_[num_joints + i].get_value();
#endif

    q_ref[i] = initialize
      ? q_meas
      : exponential_moving_average(q_ref[i], q_target[i], params_.filter.q_ref);

    q[i] = initialize
      ? q_meas
      : exponential_moving_average(q[i], q_meas, params_.filter.q);

    if (continous_joint_types.count(joint.shortname())) {
      q_pin[joint.idx_q()] = std::cos(q[i]);
      q_pin[joint.idx_q() + 1] = std::sin(q[i]);
    } else {
      q_pin[joint.idx_q()] = q[i];
    }

    dq[i] = initialize
      ? dq_meas
      : exponential_moving_average(dq[i], dq_meas, params_.filter.dq);

    q_target[i] = initialize ? q_meas : q_target[i];
  }
}

CallbackReturn
CartesianAdmittanceController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {

  // Update the current state with initial measurements (no EMA filtering)
  updateCurrentState(true);

  pinocchio::forwardKinematics(model_, data_, q_pin, dq);
  pinocchio::updateFramePlacements(model_, data_);

  end_effector_pose = data_.oMf[end_effector_frame_id];

  target_position_ = end_effector_pose.translation();
  target_orientation_ = Eigen::Quaterniond(end_effector_pose.rotation());
  desired_position_ = target_position_;
  desired_orientation_ = target_orientation_;

  // Reset admittance state so it reinitializes on first update cycle
  admittance_initialized_ = false;
  inner_motion_.setZero();
  ft_wrench_.setZero();

  RCLCPP_INFO(get_node()->get_logger(), "Admittance controller activated.");
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
CartesianAdmittanceController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

void CartesianAdmittanceController::parse_target_pose_() {
  auto msg = *target_pose_buffer_.readFromRT();
  target_position_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  target_orientation_ = Eigen::Quaterniond(
    msg->pose.orientation.w,
    msg->pose.orientation.x,
    msg->pose.orientation.y,
    msg->pose.orientation.z);
}

void CartesianAdmittanceController::parse_target_joint_() {
  auto msg = *target_joint_buffer_.readFromRT();
  size_t num_joints = static_cast<size_t>(model_.nv);
  if (msg->position.size()) {
    for (size_t i = 0; i < std::min(msg->position.size(), num_joints); ++i) {
      q_target[i] = msg->position[i];
    }
  }
  if (msg->velocity.size()) {
    for (size_t i = 0; i < std::min(msg->velocity.size(), num_joints); ++i) {
      dq_ref[i] = msg->velocity[i];
    }
  }
}

void CartesianAdmittanceController::parse_target_wrench_() {
  auto msg = *target_wrench_buffer_.readFromRT();
  target_wrench_ << msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z,
    msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z;
}

void CartesianAdmittanceController::parse_target_stiffness_() {
  auto msg = *target_stiffness_buffer_.readFromRT();
  if (msg->data.size() != 6) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "Variable stiffness message must have exactly 6 elements, got %zu. Ignoring.",
      msg->data.size());
    return;
  }
  const double max_k_trans = params_.variable_max_impedance_stiffness.translational;
  const double max_k_rot = params_.variable_max_impedance_stiffness.rotational;
  std::array<double, 6> vals = {msg->data[0], msg->data[1], msg->data[2],
                                msg->data[3], msg->data[4], msg->data[5]};
  for (int i = 0; i < 3; ++i) {
    if (vals[i] < 0.0 || vals[i] > max_k_trans) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Topic impedance stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, vals[i], max_k_trans);
      vals[i] = std::clamp(vals[i], 0.0, max_k_trans);
    }
  }
  for (int i = 3; i < 6; ++i) {
    if (vals[i] < 0.0 || vals[i] > max_k_rot) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Topic impedance stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, vals[i], max_k_rot);
      vals[i] = std::clamp(vals[i], 0.0, max_k_rot);
    }
  }
  topic_stiffness_.setZero();
  topic_stiffness_.diagonal() << vals[0], vals[1], vals[2], vals[3], vals[4], vals[5];
  use_topic_stiffness_ = true;
  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
    "Variable impedance stiffness received: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
    vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
}

void CartesianAdmittanceController::parse_ft_sensor_() {
  auto msg = *ft_sensor_buffer_.readFromRT();
  ft_wrench_ << msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z,
    msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z;
}

void CartesianAdmittanceController::parse_target_adm_stiffness_() {
  auto msg = *target_adm_stiffness_buffer_.readFromRT();
  if (msg->data.size() != 6) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "Variable admittance stiffness must have 6 elements, got %zu. Ignoring.",
      msg->data.size());
    return;
  }
  const double max_ak_trans = params_.variable_max_admittance_stiffness.translational;
  const double max_ak_rot = params_.variable_max_admittance_stiffness.rotational;
  std::array<double, 6> avals = {msg->data[0], msg->data[1], msg->data[2],
                                 msg->data[3], msg->data[4], msg->data[5]};
  for (int i = 0; i < 3; ++i) {
    if (avals[i] < 0.0 || avals[i] > max_ak_trans) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Topic admittance stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, avals[i], max_ak_trans);
      avals[i] = std::clamp(avals[i], 0.0, max_ak_trans);
    }
  }
  for (int i = 3; i < 6; ++i) {
    if (avals[i] < 0.0 || avals[i] > max_ak_rot) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Topic admittance stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, avals[i], max_ak_rot);
      avals[i] = std::clamp(avals[i], 0.0, max_ak_rot);
    }
  }
  topic_adm_stiffness_.setZero();
  topic_adm_stiffness_.diagonal() << avals[0], avals[1], avals[2], avals[3], avals[4], avals[5];
  use_topic_adm_stiffness_ = true;
  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
    "Variable admittance stiffness received: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
    avals[0], avals[1], avals[2], avals[3], avals[4], avals[5]);
}

void CartesianAdmittanceController::log_debug_info(const rclcpp::Time & time) {
  if (!params_.log.enabled) {
    return;
  }
  if (params_.log.robot_state) {
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "nq: " << model_.nq << ", nv: " << model_.nv);

    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "end_effector_pos" << end_effector_pose.translation());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "q: " << q.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "q_pin: " << q_pin.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "dq: " << dq.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "J: " << J);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "inner_SE3 pos: " << inner_SE3_.translation().transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "inner_motion: " << inner_motion_.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "ft_wrench: " << ft_wrench_.transpose());
  }

  if (params_.log.control_values) {
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "error: " << error.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "max_delta: " << max_delta_.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "q_ref: " << q_ref.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "dq_ref: " << dq_ref.transpose());
  }

  if (params_.log.controller_parameters) {
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "stiffness: " << stiffness);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "damping: " << damping);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "nullspace_stiffness: " << nullspace_stiffness);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "nullspace_damping: " << nullspace_damping);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "adm_mass: " << adm_mass_.diagonal().transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "adm_stiffness: " << adm_stiffness_.diagonal().transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "adm_damping: " << adm_damping_.diagonal().transpose());
  }

  if (params_.log.limits) {
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "joint_limits: " << model_.lowerPositionLimit.transpose() << ", "
                       << model_.upperPositionLimit.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "velocity_limits: " << model_.velocityLimit);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "effort_limits: " << model_.effortLimit);
  }

  if (params_.log.computed_torques) {
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "tau_task: " << tau_task.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "tau_joint_limits: " << tau_joint_limits.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "tau_nullspace: " << tau_nullspace.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "tau_friction: " << tau_friction.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "tau_coriolis: " << tau_coriolis.transpose());
  }

  if (params_.log.dynamic_params) {
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "M: " << data_.M);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "Minv: " << data_.Minv);
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "nullspace projector: " << nullspace_projection);
  }

  if (params_.log.timing) {
    auto t_end = get_node()->get_clock()->now();
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      2000,
      "Control loop needed: " << (t_end.nanoseconds() - time.nanoseconds()) * 1e-6 << " ms");
  }
}

bool CartesianAdmittanceController::check_topic_publisher_count(const std::string & topic_name) {
  std::vector<rclcpp::TopicEndpointInfo> topic_info;
  try {
    topic_info = get_node()->get_publishers_info_by_topic(topic_name);
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      5000,
      "Failed to get publisher info for topic '%s': %s", topic_name.c_str(), e.what());
    return true;  // Allow message through if check fails
  }
  size_t publisher_count = topic_info.size();

  if (publisher_count > max_allowed_publishers_) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      2000,
      "Topic '%s' has %zu publishers (expected max: %zu). Multiple command sources detected!",
      topic_name.c_str(),
      publisher_count,
      max_allowed_publishers_);

    if (!multiple_publishers_detected_) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "SAFETY WARNING: Multiple publishers detected on topic '%s'! "
        "Ignoring commands from this topic to prevent conflicting control signals.",
        topic_name.c_str());
      multiple_publishers_detected_ = true;
    }
    return false;
  }

  if (multiple_publishers_detected_ && publisher_count <= max_allowed_publishers_) {
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Publisher conflict resolved on topic '%s'. Resuming message processing.",
      topic_name.c_str());
    multiple_publishers_detected_ = false;
  }

  return true;
}

}  // namespace crisp_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  crisp_controllers::CartesianAdmittanceController, controller_interface::ControllerInterface)
