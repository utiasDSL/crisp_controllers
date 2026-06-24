#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include <crisp_controllers/state_broadcaster.hpp>
#include <crisp_controllers/utils/fiters.hpp>
#include "crisp_controllers/utils/ros2_version.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/logging.hpp>

namespace crisp_controllers {

controller_interface::InterfaceConfiguration
StateBroadcaster::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::InterfaceConfiguration
StateBroadcaster::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::ALL;
  return config;
}

controller_interface::return_type
StateBroadcaster::update(const rclcpp::Time & time, const rclcpp::Duration & period) {

  // Handle position-interface-based broadcasting
  read_position_interfaces();
  update_pinocchio_positions();
  pinocchio::forwardKinematics(model_, data_, q_pin_);
  pinocchio::updateFramePlacements(model_, data_);

  const auto & current_pose = data_.oMf[end_effector_frame_id_];
  const Eigen::Quaterniond current_quaternion(current_pose.rotation());

  if (should_publish(period, pose_timer_) && realtime_pose_publisher_) {
    pose_msg_.header.stamp = time;
    pose_msg_.pose.position.x = current_pose.translation()[0];
    pose_msg_.pose.position.y = current_pose.translation()[1];
    pose_msg_.pose.position.z = current_pose.translation()[2];
    pose_msg_.pose.orientation.x = current_quaternion.x();
    pose_msg_.pose.orientation.y = current_quaternion.y();
    pose_msg_.pose.orientation.z = current_quaternion.z();
    pose_msg_.pose.orientation.w = current_quaternion.w();

#if REALTIME_TOOLS_NEW_API
    realtime_pose_publisher_->try_publish(pose_msg_);
#else
    if (realtime_pose_publisher_->trylock()) {
      realtime_pose_publisher_->msg_ = pose_msg_;
      realtime_pose_publisher_->unlockAndPublish();
    }
#endif
  }

  for (size_t i = 0; i < params_.joints.size(); ++i) {
    joint_state_msg_.position[i] = q_[i];
  }

  // Handle velocity-interface-based broadcasting
  if (has_velocity_interfaces_) {
    read_velocity_interfaces();
    pinocchio::forwardKinematics(model_, data_, q_pin_, dq_);
    pinocchio::updateFramePlacements(model_, data_);

    const auto current_velocity =
      pinocchio::getFrameVelocity(model_, data_, end_effector_frame_id_, twist_reference_frame_);
    if (should_publish(period, twist_timer_) && realtime_twist_publisher_) {
      twist_msg_.header.stamp = time;
      twist_msg_.twist.linear.x = current_velocity.linear()[0];
      twist_msg_.twist.linear.y = current_velocity.linear()[1];
      twist_msg_.twist.linear.z = current_velocity.linear()[2];
      twist_msg_.twist.angular.x = current_velocity.angular()[0];
      twist_msg_.twist.angular.y = current_velocity.angular()[1];
      twist_msg_.twist.angular.z = current_velocity.angular()[2];

#if REALTIME_TOOLS_NEW_API
      realtime_twist_publisher_->try_publish(twist_msg_);
#else
      if (realtime_twist_publisher_->trylock()) {
        realtime_twist_publisher_->msg_ = twist_msg_;
        realtime_twist_publisher_->unlockAndPublish();
      }
#endif
    }

    for (size_t i = 0; i < params_.joints.size(); ++i) {
      joint_state_msg_.velocity[i] = dq_[joint_v_indices_[i]];
    }
  }

  // Handle effort-interface-based broadcasting
  if (has_effort_interfaces_) {
    read_effort_interfaces();
    for (size_t i = 0; i < params_.joints.size(); ++i) {
      joint_state_msg_.effort[i] = tau_measured_[joint_v_indices_[i]];
    }

    compute_wrench(tau_measured_, raw_wrench_reference_frame_, raw_wrench_);
    if (should_publish(period, raw_wrench_timer_) && realtime_raw_wrench_publisher_) {
      raw_wrench_msg_.header.stamp = time;
      raw_wrench_msg_.wrench.force.x = raw_wrench_[0];
      raw_wrench_msg_.wrench.force.y = raw_wrench_[1];
      raw_wrench_msg_.wrench.force.z = raw_wrench_[2];
      raw_wrench_msg_.wrench.torque.x = raw_wrench_[3];
      raw_wrench_msg_.wrench.torque.y = raw_wrench_[4];
      raw_wrench_msg_.wrench.torque.z = raw_wrench_[5];

#if REALTIME_TOOLS_NEW_API
      realtime_raw_wrench_publisher_->try_publish(raw_wrench_msg_);
#else
      if (realtime_raw_wrench_publisher_->trylock()) {
        realtime_raw_wrench_publisher_->msg_ = raw_wrench_msg_;
        realtime_raw_wrench_publisher_->unlockAndPublish();
      }
#endif
    }
  }

  // External joint effort and external wrench derivation require velocity available
  if (has_velocity_interfaces_ && has_effort_interfaces_) {
    if (params_.use_coriolis_compensation) {
      pinocchio::computeAllTerms(model_, data_, q_pin_, dq_);
      tau_coriolis_.noalias() =
        pinocchio::computeCoriolisMatrix(model_, data_, q_pin_, dq_) * dq_;
    } else {
      tau_coriolis_.setZero();
    }

    if (params_.use_gravity_compensation) {
      tau_gravity_.noalias() = pinocchio::computeGeneralizedGravity(model_, data_, q_pin_);
    } else {
      tau_gravity_.setZero();
    }

    if (params_.use_inertial_compensation) {
      const double period_seconds = period.seconds();
      if (!has_previous_velocity_) {
        dq_previous_ = dq_;
        tau_inertia_.setZero();
        has_previous_velocity_ = true;
      } else if (period_seconds > 0.0) {
        ddq_estimated_.noalias() = dq_ - dq_previous_;
        ddq_estimated_ /= period_seconds;
        for (Eigen::Index i = 0; i < ddq_filtered_.size(); ++i) {
          ddq_filtered_[i] = exponential_moving_average(
            ddq_filtered_[i], ddq_estimated_[i], params_.acceleration_filter_alpha);
        }

        pinocchio::crba(model_, data_, q_pin_);
        data_.M.triangularView<Eigen::StrictlyLower>() =
          data_.M.transpose().triangularView<Eigen::StrictlyLower>();
        tau_inertia_.noalias() = data_.M * ddq_filtered_;
        dq_previous_ = dq_;
      } else {
        tau_inertia_.setZero();
        dq_previous_ = dq_;
      }
    } else {
      tau_inertia_.setZero();
      has_previous_velocity_ = false;
    }

    tau_residual_.noalias() = tau_measured_;
    tau_residual_ -= tau_coriolis_;
    tau_residual_ -= tau_gravity_;
    tau_residual_ -= tau_inertia_;
    update_external_effort_message();

    if (should_publish(period, external_effort_timer_) && realtime_external_effort_publisher_) {
#if REALTIME_TOOLS_NEW_API
      realtime_external_effort_publisher_->try_publish(external_effort_msg_);
#else
      if (realtime_external_effort_publisher_->trylock()) {
        realtime_external_effort_publisher_->msg_ = external_effort_msg_;
        realtime_external_effort_publisher_->unlockAndPublish();
      }
#endif
    }

    compute_wrench(tau_residual_, external_wrench_reference_frame_, external_wrench_);
    if (should_publish(period, external_wrench_timer_) && realtime_external_wrench_publisher_) {
      external_wrench_msg_.header.stamp = time;
      external_wrench_msg_.wrench.force.x = external_wrench_[0];
      external_wrench_msg_.wrench.force.y = external_wrench_[1];
      external_wrench_msg_.wrench.force.z = external_wrench_[2];
      external_wrench_msg_.wrench.torque.x = external_wrench_[3];
      external_wrench_msg_.wrench.torque.y = external_wrench_[4];
      external_wrench_msg_.wrench.torque.z = external_wrench_[5];

#if REALTIME_TOOLS_NEW_API
      realtime_external_wrench_publisher_->try_publish(external_wrench_msg_);
#else
      if (realtime_external_wrench_publisher_->trylock()) {
        realtime_external_wrench_publisher_->msg_ = external_wrench_msg_;
        realtime_external_wrench_publisher_->unlockAndPublish();
      }
#endif
    }
  }

  if (should_publish(period, joint_state_timer_) && realtime_joint_state_publisher_) {
    joint_state_msg_.header.stamp = time;
#if REALTIME_TOOLS_NEW_API
    realtime_joint_state_publisher_->try_publish(joint_state_msg_);
#else
    if (realtime_joint_state_publisher_->trylock()) {
      realtime_joint_state_publisher_->msg_ = joint_state_msg_;
      realtime_joint_state_publisher_->unlockAndPublish();
    }
#endif
  }

  return controller_interface::return_type::OK;
}

CallbackReturn StateBroadcaster::on_init() {
  params_listener_ = std::make_shared<state_broadcaster::ParamListener>(get_node());
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();

  return CallbackReturn::SUCCESS;
}

CallbackReturn
StateBroadcaster::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {
  warn_legacy_parameters();

  if (!validate_common_parameters() || !build_model() || !cache_joint_model_indices()) {
    return CallbackReturn::ERROR;
  }

  std::string raw_wrench_frame;
  std::string external_wrench_frame;
  if (
    !configure_reference_frame(
      "twist", params_.twist.reference_frame, twist_reference_frame_, twist_msg_.header.frame_id) ||
    !configure_reference_frame(
      "wrench.raw", params_.wrench.raw.reference_frame, raw_wrench_reference_frame_,
      raw_wrench_frame) ||
    !configure_reference_frame(
      "wrench.external", params_.wrench.external.reference_frame, external_wrench_reference_frame_,
      external_wrench_frame)) {
    return CallbackReturn::ERROR;
  }

  pose_msg_.header.frame_id =
    params_.pose.frame.empty() ? params_.base_frame : params_.pose.frame;
  twist_msg_.header.frame_id =
    params_.twist.frame.empty() ? twist_msg_.header.frame_id : params_.twist.frame;
  raw_wrench_msg_.header.frame_id =
    params_.wrench.raw.frame.empty() ? raw_wrench_frame : params_.wrench.raw.frame;
  external_wrench_msg_.header.frame_id =
    params_.wrench.external.frame.empty() ? external_wrench_frame : params_.wrench.external.frame;

  if (
    !configure_publish_interval("pose", params_.pose.publish_frequency, pose_timer_) ||
    !configure_publish_interval("twist", params_.twist.publish_frequency, twist_timer_) ||
    !configure_publish_interval(
      "wrench.raw", params_.wrench.raw.publish_frequency, raw_wrench_timer_) ||
    !configure_publish_interval(
      "wrench.external", params_.wrench.external.publish_frequency, external_wrench_timer_) ||
    !configure_publish_interval(
      "effort.external", params_.effort.external.publish_frequency, external_effort_timer_) ||
    !configure_publish_interval(
      "joint_states", params_.joint_states.publish_frequency, joint_state_timer_)) {
    return CallbackReturn::ERROR;
  }

  pose_publisher_ = get_node()->create_publisher<geometry_msgs::msg::PoseStamped>(
    params_.pose.topic, rclcpp::SystemDefaultsQoS());
  realtime_pose_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>>(
      pose_publisher_);

  twist_publisher_ = get_node()->create_publisher<geometry_msgs::msg::TwistStamped>(
    params_.twist.topic, rclcpp::SystemDefaultsQoS());
  realtime_twist_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::TwistStamped>>(
      twist_publisher_);

  raw_wrench_publisher_ = get_node()->create_publisher<geometry_msgs::msg::WrenchStamped>(
    params_.wrench.raw.topic, rclcpp::SystemDefaultsQoS());
  realtime_raw_wrench_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>(
      raw_wrench_publisher_);

  external_wrench_publisher_ = get_node()->create_publisher<geometry_msgs::msg::WrenchStamped>(
    params_.wrench.external.topic, rclcpp::SystemDefaultsQoS());
  realtime_external_wrench_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>(
      external_wrench_publisher_);

  external_effort_publisher_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    params_.effort.external.topic, rclcpp::SystemDefaultsQoS());
  realtime_external_effort_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(
      external_effort_publisher_);

  joint_state_publisher_ = get_node()->create_publisher<sensor_msgs::msg::JointState>(
    params_.joint_states.topic, rclcpp::SystemDefaultsQoS());
  realtime_joint_state_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
      joint_state_publisher_);

#if !REALTIME_TOOLS_NEW_API
  realtime_pose_publisher_->msg_.header.frame_id = pose_msg_.header.frame_id;
  realtime_twist_publisher_->msg_.header.frame_id = twist_msg_.header.frame_id;
  realtime_raw_wrench_publisher_->msg_.header.frame_id = raw_wrench_msg_.header.frame_id;
  realtime_external_wrench_publisher_->msg_.header.frame_id =
    external_wrench_msg_.header.frame_id;
#endif

  return CallbackReturn::SUCCESS;
}

CallbackReturn StateBroadcaster::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
  if (!cache_state_interface_indices()) {
    return CallbackReturn::ERROR;
  }

  pose_timer_.elapsed = rclcpp::Duration(0, 0);
  twist_timer_.elapsed = rclcpp::Duration(0, 0);
  raw_wrench_timer_.elapsed = rclcpp::Duration(0, 0);
  external_wrench_timer_.elapsed = rclcpp::Duration(0, 0);
  external_effort_timer_.elapsed = rclcpp::Duration(0, 0);
  joint_state_timer_.elapsed = rclcpp::Duration(0, 0);
  has_previous_velocity_ = false;
  ddq_estimated_.setZero();
  ddq_filtered_.setZero();
  tau_inertia_.setZero();

  joint_state_msg_.name = params_.joints;
  joint_state_msg_.position.assign(params_.joints.size(), 0.0);
  joint_state_msg_.velocity.clear();
  joint_state_msg_.effort.clear();
  if (has_velocity_interfaces_) {
    joint_state_msg_.velocity.assign(params_.joints.size(), 0.0);
  }
  if (has_effort_interfaces_) {
    joint_state_msg_.effort.assign(params_.joints.size(), 0.0);
  }

  external_effort_msg_.data.assign(params_.joints.size(), 0.0);

#if !REALTIME_TOOLS_NEW_API
  realtime_joint_state_publisher_->msg_ = joint_state_msg_;
  realtime_external_effort_publisher_->msg_ = external_effort_msg_;
#endif

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
StateBroadcaster::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

bool StateBroadcaster::configure_reference_frame(
  const std::string & output_name, const std::string & reference_frame,
  pinocchio::ReferenceFrame & parsed_reference_frame, std::string & published_frame) const {
  if (reference_frame == "local") {
    parsed_reference_frame = pinocchio::ReferenceFrame::LOCAL;
    published_frame = params_.end_effector_frame;
    return true;
  }
  if (reference_frame == "local_world_aligned") {
    if (params_.base_frame.empty()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to configure because base_frame is empty for %s local_world_aligned output.",
        output_name.c_str());
      return false;
    }
    parsed_reference_frame = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    published_frame = params_.base_frame;
    return true;
  }

  RCLCPP_ERROR(
    get_node()->get_logger(),
    "Failed to configure because %s reference_frame '%s' is unsupported.", output_name.c_str(),
    reference_frame.c_str());
  return false;
}

bool StateBroadcaster::configure_publish_interval(
  const std::string & output_name, double publish_frequency, PublishTimer & timer) const {
  if (publish_frequency < 0.0) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to configure because %s publish_frequency must be non-negative.",
      output_name.c_str());
    return false;
  }

  timer.interval = publish_frequency > 0.0 ? rclcpp::Duration::from_seconds(1.0 / publish_frequency)
                                           : rclcpp::Duration(0, 0);
  return true;
}

void StateBroadcaster::warn_legacy_parameters() const {
  const auto & overrides = get_node()->get_node_parameters_interface()->get_parameter_overrides();
  const bool has_legacy_topic = overrides.find("topic") != overrides.end();
  const bool has_legacy_publish_frequency = overrides.find("publish_frequency") != overrides.end();

  if (!has_legacy_topic && !has_legacy_publish_frequency) {
    return;
  }

  const char * legacy_params = has_legacy_topic && has_legacy_publish_frequency
    ? "'topic' and 'publish_frequency' were"
    : has_legacy_topic ? "'topic' was" : "'publish_frequency' was";

  RCLCPP_WARN(
    get_node()->get_logger(),
    "Deprecated flat StateBroadcaster parameter%s %s supplied and ignored. "
    "Use pose.topic/pose.publish_frequency or twist.topic/twist.publish_frequency instead. "
    "Configured nested values, or their defaults, will be used.",
    has_legacy_topic && has_legacy_publish_frequency ? "s" : "", legacy_params);
}

bool StateBroadcaster::validate_common_parameters() const {
  if (params_.joints.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to configure because joints is empty.");
    return false;
  }
  if (params_.end_effector_frame.empty()) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Failed to configure because end_effector_frame is empty.");
    return false;
  }
  if (params_.base_frame.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to configure because base_frame is empty.");
    return false;
  }

  const std::array<std::pair<const char *, std::string>, 6> topics = {{
    {"pose.topic", params_.pose.topic},
    {"twist.topic", params_.twist.topic},
    {"wrench.raw.topic", params_.wrench.raw.topic},
    {"wrench.external.topic", params_.wrench.external.topic},
    {"effort.external.topic", params_.effort.external.topic},
    {"joint_states.topic", params_.joint_states.topic},
  }};
  for (const auto & [name, topic] : topics) {
    if (topic.empty()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to configure because %s is empty.", name);
      return false;
    }
  }

  return true;
}

bool StateBroadcaster::build_model() {
  constexpr auto robot_state_publisher_node = "robot_state_publisher";
  constexpr auto robot_description_parameter = "robot_description";
  constexpr auto parameter_service_timeout = std::chrono::seconds(2);

  auto parameters_client =
    std::make_shared<rclcpp::AsyncParametersClient>(get_node(), robot_state_publisher_node);
  if (!parameters_client->wait_for_service(parameter_service_timeout)) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to configure because %s did not provide its parameter service within 2 seconds; "
      "cannot read %s.",
      robot_state_publisher_node, robot_description_parameter);
    return false;
  }

  auto parameters_future = parameters_client->get_parameters({robot_description_parameter});
  if (parameters_future.wait_for(parameter_service_timeout) != std::future_status::ready) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to configure because %s did not return %s within 2 seconds.",
      robot_state_publisher_node, robot_description_parameter);
    return false;
  }

  const auto result = parameters_future.get();

  std::string robot_description;
  if (
    !result.empty() && result[0].get_name() == robot_description_parameter &&
    result[0].get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
    robot_description = result[0].as_string();
  } else {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to configure because %s did not return a string %s parameter.",
      robot_state_publisher_node, robot_description_parameter);
    return false;
  }

  if (robot_description.empty()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to configure because %s returned an empty %s parameter.",
      robot_state_publisher_node, robot_description_parameter);
    return false;
  }

  pinocchio::Model raw_model;
  pinocchio::urdf::buildModelFromXML(robot_description, raw_model);

  for (const auto & joint : params_.joints) {
    if (!raw_model.existJointName(joint)) {
      RCLCPP_ERROR_STREAM(
        get_node()->get_logger(),
        "Failed to configure because "
          << joint
          << " is not part of the kinematic tree but it has been passed in the parameters.");
      return false;
    }
  }

  std::vector<pinocchio::JointIndex> list_of_joints_to_lock_by_id;
  for (const auto & joint : raw_model.names) {
    if (
      std::find(params_.joints.begin(), params_.joints.end(), joint) == params_.joints.end() &&
      joint != "universe") {
      list_of_joints_to_lock_by_id.push_back(raw_model.getJointId(joint));
    }
  }

  Eigen::VectorXd q_locked = Eigen::VectorXd::Zero(raw_model.nq);
  model_ = pinocchio::buildReducedModel(raw_model, list_of_joints_to_lock_by_id, q_locked);
  data_ = pinocchio::Data(model_);

  for (int joint_id = 0; joint_id < model_.njoints; joint_id++) {
    if (model_.names[joint_id] == "universe") {
      continue;
    }
    if (!allowed_joint_types_.count(model_.joints[joint_id].shortname())) {
      RCLCPP_ERROR_STREAM(
        get_node()->get_logger(),
        "Joint type " << model_.joints[joint_id].shortname() << " is unsupported ("
                      << model_.names[joint_id]
                      << "), only revolute/continuous-like joints can be used.");
      return false;
    }
  }

  if (!model_.existFrame(params_.end_effector_frame)) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to configure because end_effector_frame '%s' is not part of the reduced model.",
      params_.end_effector_frame.c_str());
    return false;
  }

  end_effector_frame_id_ = model_.getFrameId(params_.end_effector_frame);

  q_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(params_.joints.size()));
  q_pin_ = Eigen::VectorXd::Zero(model_.nq);
  dq_ = Eigen::VectorXd::Zero(model_.nv);
  dq_previous_ = Eigen::VectorXd::Zero(model_.nv);
  ddq_estimated_ = Eigen::VectorXd::Zero(model_.nv);
  ddq_filtered_ = Eigen::VectorXd::Zero(model_.nv);
  tau_measured_ = Eigen::VectorXd::Zero(model_.nv);
  tau_coriolis_ = Eigen::VectorXd::Zero(model_.nv);
  tau_gravity_ = Eigen::VectorXd::Zero(model_.nv);
  tau_inertia_ = Eigen::VectorXd::Zero(model_.nv);
  tau_residual_ = Eigen::VectorXd::Zero(model_.nv);
  J_ = pinocchio::Data::Matrix6x::Zero(6, model_.nv);
  wrench_system_.setZero();
  wrench_rhs_.setZero();
  raw_wrench_.setZero();
  external_wrench_.setZero();

  return true;
}

bool StateBroadcaster::cache_joint_model_indices() {
  joint_q_indices_.clear();
  joint_v_indices_.clear();
  joint_is_continuous_.clear();
  joint_q_indices_.reserve(params_.joints.size());
  joint_v_indices_.reserve(params_.joints.size());
  joint_is_continuous_.reserve(params_.joints.size());

  for (const auto & joint_name : params_.joints) {
    const auto joint_id = model_.getJointId(joint_name);
    const auto & joint = model_.joints[joint_id];
    joint_q_indices_.push_back(joint.idx_q());
    joint_v_indices_.push_back(joint.idx_v());
    joint_is_continuous_.push_back(continuous_joint_types_.count(joint.shortname()) > 0);
  }

  return true;
}

bool StateBroadcaster::cache_state_interface_indices() {
  position_interface_indices_.assign(params_.joints.size(), kMissingInterface);
  velocity_interface_indices_.assign(params_.joints.size(), kMissingInterface);
  effort_interface_indices_.assign(params_.joints.size(), kMissingInterface);

  for (size_t interface_index = 0; interface_index < state_interfaces_.size(); ++interface_index) {
    const auto joint_name = state_interfaces_[interface_index].get_prefix_name();
    const auto interface_name = state_interfaces_[interface_index].get_interface_name();
    for (size_t joint_index = 0; joint_index < params_.joints.size(); ++joint_index) {
      if (joint_name != params_.joints[joint_index]) {
        continue;
      }
      if (interface_name == "position") {
        position_interface_indices_[joint_index] = interface_index;
      } else if (interface_name == "velocity") {
        velocity_interface_indices_[joint_index] = interface_index;
      } else if (interface_name == "effort") {
        effort_interface_indices_[joint_index] = interface_index;
      }
    }
  }

  for (size_t i = 0; i < params_.joints.size(); ++i) {
    if (position_interface_indices_[i] == kMissingInterface) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to activate because joint '%s' does not provide a position state interface.",
        params_.joints[i].c_str());
      return false;
    }
  }

  has_velocity_interfaces_ = std::all_of(
    velocity_interface_indices_.begin(), velocity_interface_indices_.end(),
    [](size_t index) { return index != kMissingInterface; });
  has_effort_interfaces_ = std::all_of(
    effort_interface_indices_.begin(), effort_interface_indices_.end(),
    [](size_t index) { return index != kMissingInterface; });

  if (!has_velocity_interfaces_) {
    RCLCPP_INFO(
      get_node()->get_logger(),
      "StateBroadcaster activated without a complete velocity interface group; twist and external "
      "effort outputs are disabled.");
  }
  if (!has_effort_interfaces_) {
    RCLCPP_INFO(
      get_node()->get_logger(),
      "StateBroadcaster activated without a complete effort interface group; wrench and external "
      "effort outputs are disabled.");
  }

  return true;
}

void StateBroadcaster::read_position_interfaces() {
  for (size_t i = 0; i < params_.joints.size(); ++i) {
#if ROS2_VERSION_ABOVE_HUMBLE
    q_[i] = state_interfaces_[position_interface_indices_[i]].get_optional().value_or(q_[i]);
#else
    q_[i] = state_interfaces_[position_interface_indices_[i]].get_value();
#endif
  }
}

void StateBroadcaster::read_velocity_interfaces() {
  for (size_t i = 0; i < params_.joints.size(); ++i) {
#if ROS2_VERSION_ABOVE_HUMBLE
    dq_[joint_v_indices_[i]] = state_interfaces_[velocity_interface_indices_[i]]
                                 .get_optional()
                                 .value_or(dq_[joint_v_indices_[i]]);
#else
    dq_[joint_v_indices_[i]] = state_interfaces_[velocity_interface_indices_[i]].get_value();
#endif
  }
}

void StateBroadcaster::read_effort_interfaces() {
  for (size_t i = 0; i < params_.joints.size(); ++i) {
#if ROS2_VERSION_ABOVE_HUMBLE
    tau_measured_[joint_v_indices_[i]] = state_interfaces_[effort_interface_indices_[i]]
                                           .get_optional()
                                           .value_or(tau_measured_[joint_v_indices_[i]]);
#else
    tau_measured_[joint_v_indices_[i]] =
      state_interfaces_[effort_interface_indices_[i]].get_value();
#endif
  }
}

void StateBroadcaster::update_pinocchio_positions() {
  q_pin_.setZero();
  for (size_t i = 0; i < params_.joints.size(); ++i) {
    if (joint_is_continuous_[i]) {
      q_pin_[joint_q_indices_[i]] = std::cos(q_[i]);
      q_pin_[joint_q_indices_[i] + 1] = std::sin(q_[i]);
    } else {
      q_pin_[joint_q_indices_[i]] = q_[i];
    }
  }
}

void StateBroadcaster::compute_wrench(
  const Eigen::VectorXd & effort, pinocchio::ReferenceFrame reference_frame,
  Eigen::Matrix<double, 6, 1> & wrench) {
  J_.setZero();
  pinocchio::computeFrameJacobian(
    model_, data_, q_pin_, end_effector_frame_id_, reference_frame, J_);
  wrench_system_.noalias() = J_ * J_.transpose();
  wrench_system_.diagonal().array() += params_.wrench.regularization;
  wrench_rhs_.noalias() = J_ * effort;
  wrench = wrench_system_.ldlt().solve(wrench_rhs_);
}

void StateBroadcaster::update_external_effort_message() {
  for (size_t i = 0; i < params_.joints.size(); ++i) {
    external_effort_msg_.data[i] = tau_residual_[joint_v_indices_[i]];
  }
}

bool StateBroadcaster::should_publish(const rclcpp::Duration & period, PublishTimer & timer) {
  timer.elapsed = timer.elapsed + period;
  const bool should_publish =
    (timer.elapsed >= timer.interval) || (timer.interval.nanoseconds() == 0);
  if (should_publish) {
    timer.elapsed = timer.elapsed - timer.interval;
    timer.elapsed = std::min(timer.elapsed, timer.interval);
  }
  return should_publish;
}

}  // namespace crisp_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  crisp_controllers::StateBroadcaster, controller_interface::ControllerInterface)
