
#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/src/Core/Matrix.h>  // NOLINT(build/include_order)
#include <fmt/format.h>
#include <controller_interface/controller_interface_base.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/fwd.hpp>

#include "pinocchio/algorithm/model.hpp"

#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <franka_spine_msgs/srv/get_position.hpp>

#include <crisp_controllers/cartesian_controller.hpp>
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
CartesianController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint_name : params_.joints) {
    config.names.push_back(joint_name + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianController::state_interface_configuration() const {
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
CartesianController::update(const rclcpp::Time & time, const rclcpp::Duration & /*period*/) {
  
  // Update current state information with EMA filtered values
  updateCurrentState();

  // Check if new targets available
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

  pinocchio::forwardKinematics(model_, data_, q_pin, dq);
  pinocchio::updateFramePlacements(model_, data_);

  desired_position_ =
    exponential_moving_average(desired_position_, target_position_, params_.filter.target_pose);
  desired_orientation_ =
    target_orientation_.slerp(params_.filter.target_pose, desired_orientation_);

  /*target_pose_ = pinocchio::SE3(target_orientation_.toRotationMatrix(),
   * target_position_);*/
  // Express the EE pose relative to base_frame (matches TF/streamed targets),
  // falling back to the model world root if base_frame is unset. See on_configure.
  end_effector_pose = (base_frame_id_ >= 0)
    ? data_.oMf[base_frame_id_].inverse() * data_.oMf[end_effector_frame_id]
    : data_.oMf[end_effector_frame_id];

  if (!end_effector_pose.translation().allFinite() ||
      !end_effector_pose.rotation().allFinite()) {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(),
      *get_node()->get_clock(),
      1000,
      "end_effector_pose contains NaN/Inf. "
      "Holding previous torque command this cycle.");
    if (!params_.stop_commands) {
      for (size_t i = 0; i < params_.joints.size(); ++i) {
#if ROS2_VERSION_ABOVE_HUMBLE
        (void)command_interfaces_[i].set_value(tau_previous[i]);
#else
        command_interfaces_[i].set_value(tau_previous[i]);
#endif
      }
    }
    return controller_interface::return_type::OK;
  }

  // We consider translation and rotation separately to avoid unatural screw
  // motions
  if (params_.use_local_jacobian) {
    error.head(3) = end_effector_pose.rotation().transpose() *
      (desired_position_ - end_effector_pose.translation());
    error.tail(3) =
      pinocchio::log3(end_effector_pose.rotation().transpose() * desired_orientation_);
  } else {
    error.head(3) = desired_position_ - end_effector_pose.translation();
    error.tail(3) =
      pinocchio::log3(desired_orientation_ * end_effector_pose.rotation().transpose());
  }

  if (params_.limit_error) {
    max_delta_ << params_.task.error_clip.x, params_.task.error_clip.y, params_.task.error_clip.z,
      params_.task.error_clip.rx, params_.task.error_clip.ry, params_.task.error_clip.rz;
    error = error.cwiseMax(-max_delta_).cwiseMin(max_delta_);
  }

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

  if (params_.use_operational_space) {
    tau_task << J.transpose() * Mx * (stiffness * error - damping * (J * dq));
  } else {
    tau_task << J.transpose() * (stiffness * error - damping * (J * dq));
  }

  if (model_.nq != model_.nv) {
    // TODO(placeholder): Then we have some continouts joints, not being handled for now
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

  if (params_.limit_torques) {
    tau_d = saturateTorqueRate(tau_d, tau_previous, params_.max_delta_tau);
  }
  tau_d = exponential_moving_average(tau_d, tau_previous, params_.filter.output_torque);

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

  params_listener_->refresh_dynamic_parameters();
  if (params_listener_->is_old(params_)) {
    params_ = params_listener_->get_params();
    setStiffnessAndDamping();
  }

  log_debug_info(time);

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianController::on_init() {
  params_listener_ = std::make_shared<cartesian_controller::ParamListener>(get_node());
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();

  return CallbackReturn::SUCCESS;
}

CallbackReturn
CartesianController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {
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

  // First we check that the passed joints exist in the kineatic tree
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
  // Now we fix all joints that are not referenced in the tree
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

  // -------------------------------------------------------------------------
  // Populate the LOCKED-joint configuration with the ROBOT'S REAL values instead
  // of zeros. If left at zero, pinocchio builds the reduced model as if the spine
  // is fully lowered and the base/torso joints are at 0 -- so the arm's base frame
  // (and thus every EE pose / Jacobian) is offset and rotated from reality. On the
  // Mobile FR3 Duo the spine is raised ~0.44 m and the base steer joints are
  // non-zero (they set each arm's mounting orientation), so locking them at 0
  // produces a wrong model -> frame-wrong commands and reflexes.
  //
  // Sources (verified on robot-0009-thor; see docs/SPINE_JOINT_STATE_BUG.md):
  //   * base / torso joints: /joint_states (correct there).
  //   * spine (franka_spine_vertical_joint): NOT correct in /joint_states (a
  //     joint_state_publisher default of 0). Read the true value from the spine
  //     driver service /franka_spine_node/get_position.
  std::map<std::string, double> locked_values;

  // A DEDICATED node + executor for the one-shot reads below. We must NOT spin
  // get_node(): the controller's node is already owned by the controller_manager
  // executor, so spinning it here throws "Node has already been added to an
  // executor" (fails on_configure). A separate node on the same ROS graph reads
  // joint_states + the spine service without touching the controller node.
  auto io_node = std::make_shared<rclcpp::Node>(
    std::string(get_node()->get_name()) + "_locked_cfg_io");
  rclcpp::executors::SingleThreadedExecutor io_exec;
  io_exec.add_node(io_node);

  // 1) one-shot read of joint_states for base/torso joint positions.
  {
    auto js = std::make_shared<sensor_msgs::msg::JointState>();
    bool got = false;
    auto sub = io_node->create_subscription<sensor_msgs::msg::JointState>(
      params_.locked_joints.joint_states_topic, rclcpp::QoS(10),
      [&js, &got](const sensor_msgs::msg::JointState::SharedPtr msg) {
        *js = *msg;
        got = true;
      });
    rclcpp::Time t0 = io_node->now();
    while (!got && (io_node->now() - t0).seconds() < 3.0) {
      io_exec.spin_some();
      rclcpp::sleep_for(std::chrono::milliseconds(20));
    }
    if (got) {
      for (size_t i = 0; i < js->name.size() && i < js->position.size(); ++i) {
        locked_values[js->name[i]] = js->position[i];
      }
      RCLCPP_INFO(get_node()->get_logger(),
        "Read %zu joint positions from joint_states for locked-joint config.",
        js->name.size());
    } else {
      RCLCPP_WARN(get_node()->get_logger(),
        "No joint_states within 3s; locked base/torso joints default to 0.");
    }
  }

  // 2) authoritative spine position from the spine driver service (overrides the
  //    bogus 0 that joint_states carries for the spine). ALWAYS read.
  {
    auto spine_cli =
      io_node->create_client<franka_spine_msgs::srv::GetPosition>(
        params_.locked_joints.spine_service);
    if (spine_cli->wait_for_service(std::chrono::seconds(2))) {
      auto req = std::make_shared<franka_spine_msgs::srv::GetPosition::Request>();
      auto fut = spine_cli->async_send_request(req);
      if (io_exec.spin_until_future_complete(fut, std::chrono::seconds(3)) ==
          rclcpp::FutureReturnCode::SUCCESS) {
        auto resp = fut.get();
        if (resp->success) {
          locked_values[params_.locked_joints.spine_joint] = resp->position;
          RCLCPP_INFO(get_node()->get_logger(),
            "Spine position from driver service: %.4f m (overrides joint_states).",
            resp->position);
        } else {
          RCLCPP_WARN(get_node()->get_logger(),
            "Spine get_position returned success=false; spine stays at prior value.");
        }
      } else {
        RCLCPP_WARN(get_node()->get_logger(),
          "Spine get_position call timed out; spine stays at prior value.");
      }
    } else {
      RCLCPP_WARN(get_node()->get_logger(),
        "Spine service '%s' unavailable; spine locked-value may be wrong.",
        params_.locked_joints.spine_service.c_str());
    }
  }
  io_exec.remove_node(io_node);

  // 3) build q_locked: for each joint being locked, if we have a real value for it,
  //    write it at that joint's configuration index (idx_q). Continuous joints
  //    (idx_q spans 2 for cos/sin) are handled explicitly.
  Eigen::VectorXd q_locked = Eigen::VectorXd::Zero(raw_model_.nq);
  for (pinocchio::JointIndex jid : list_of_joints_to_lock_by_id) {
    const std::string & jname = raw_model_.names[jid];
    auto it = locked_values.find(jname);
    if (it == locked_values.end()) {
      continue;  // no real value -> leave at 0
    }
    const auto & joint = raw_model_.joints[jid];
    int iq = joint.idx_q();
    if (continous_joint_types.count(joint.shortname())) {
      q_locked[iq] = std::cos(it->second);
      q_locked[iq + 1] = std::sin(it->second);
    } else {
      q_locked[iq] = it->second;
    }
    RCLCPP_INFO_STREAM(get_node()->get_logger(),
      "Locked joint '" << jname << "' set to real value " << it->second
      << " (was 0).");
  }
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
  if (!model_.existFrame(params_.end_effector_frame)) {
    RCLCPP_ERROR_STREAM(
      get_node()->get_logger(),
      "end_effector_frame '" << params_.end_effector_frame
        << "' is not present in the robot model. Refusing to configure: "
           "activating with an invalid frame results in undefined behavior "
           "(out-of-bounds access into pinocchio::Data, manifesting as a "
           "segfault or NaN/Inf in computed torques).");
    return CallbackReturn::ERROR;
  }
  end_effector_frame_id = model_.getFrameId(params_.end_effector_frame);
  // Resolve base_frame so the reported/controlled EE pose can be expressed
  // relative to it (matches TF), rather than in the pinocchio model world root.
  if (!params_.base_frame.empty() && model_.existFrame(params_.base_frame)) {
    base_frame_id_ = static_cast<int>(model_.getFrameId(params_.base_frame));
    RCLCPP_INFO_STREAM(get_node()->get_logger(),
      "EE pose will be expressed relative to base_frame '" << params_.base_frame
      << "' (id " << base_frame_id_ << ").");
  } else {
    base_frame_id_ = -1;
    RCLCPP_WARN_STREAM(get_node()->get_logger(),
      "base_frame '" << params_.base_frame << "' not set/found; EE pose stays in the "
      "model world root (may not match TF).");
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

  new_target_pose_ = false;
  new_target_joint_ = false;
  new_target_wrench_ = false;
  new_target_stiffness_ = false;
  use_topic_stiffness_ = false;

  multiple_publishers_detected_ = false;
  max_allowed_publishers_ = 1;

  const auto target_pose_topic =
    params_.topics.target_pose.empty() ? "target_pose" : params_.topics.target_pose;
  const auto target_joint_topic =
    params_.topics.target_joint.empty() ? "target_joint" : params_.topics.target_joint;
  const auto target_wrench_topic =
    params_.topics.target_wrench.empty() ? "target_wrench" : params_.topics.target_wrench;

  auto target_pose_callback =
    [this, target_pose_topic](const std::shared_ptr<geometry_msgs::msg::PoseStamped> msg) -> void {
    if (!check_topic_publisher_count(target_pose_topic)) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring message on '%s' due to multiple publishers detected!",
        target_pose_topic.c_str());
      return;
    }
    target_pose_buffer_.writeFromNonRT(msg);
    new_target_pose_ = true;
  };

  auto target_joint_callback =
    [this, target_joint_topic](const std::shared_ptr<sensor_msgs::msg::JointState> msg) -> void {
    if (!check_topic_publisher_count(target_joint_topic)) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring message on '%s' due to multiple publishers detected!",
        target_joint_topic.c_str());
      return;
    }
    target_joint_buffer_.writeFromNonRT(msg);
    new_target_joint_ = true;
  };

  auto target_wrench_callback =
    [this, target_wrench_topic](
      const std::shared_ptr<geometry_msgs::msg::WrenchStamped> msg) -> void {
    if (!check_topic_publisher_count(target_wrench_topic)) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        1000,
        "Ignoring message on '%s' due to multiple publishers detected!",
        target_wrench_topic.c_str());
      return;
    }
    target_wrench_buffer_.writeFromNonRT(msg);
    new_target_wrench_ = true;
  };

  pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
    target_pose_topic, rclcpp::QoS(1), target_pose_callback);

  joint_sub_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
    target_joint_topic, rclcpp::QoS(1), target_joint_callback);

  wrench_sub_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
    target_wrench_topic, rclcpp::QoS(1), target_wrench_callback);

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

  RCLCPP_INFO(get_node()->get_logger(), "Variable stiffness topic: %s",
    params_.variable_stiffness.topic.c_str());

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

  RCLCPP_INFO(get_node()->get_logger(), "State interfaces and control vectors initialized.");

  return CallbackReturn::SUCCESS;
}

void CartesianController::setStiffnessAndDamping() {
  if (use_topic_stiffness_) {
    stiffness = topic_stiffness_;
  } else {
    stiffness.setZero();
    stiffness.diagonal() << params_.task.k_pos_x, params_.task.k_pos_y, params_.task.k_pos_z,
      params_.task.k_rot_x, params_.task.k_rot_y, params_.task.k_rot_z;
  }

  // Clamp stiffness to [0, max_stiffness]
  const double max_k_trans = params_.variable_max_stiffness.translational;
  const double max_k_rot = params_.variable_max_stiffness.rotational;
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

  // For nullspace, use explicit damping if > 0, otherwise compute from stiffness
  if (params_.nullspace.damping > 0) {
    nullspace_damping.diagonal() = params_.nullspace.damping * weights;
  } else {
    nullspace_damping.diagonal() = 2.0 * nullspace_stiffness.diagonal().cwiseSqrt();
  }
}

void CartesianController::updateCurrentState(bool initialize) {
  auto num_joints = params_.joints.size();
  for (size_t i = 0; i < num_joints; i++) {
    auto joint_name = params_.joints[i];
    auto joint_id = model_.getJointId(joint_name);  // pinocchio joind id might be different
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
    
    if (continous_joint_types.count(joint.shortname())) { // Then we are handling a continous
                                                          // joint that is SO(2)
      q_pin[joint.idx_q()] = std::cos(q[i]);
      q_pin[joint.idx_q() + 1] = std::sin(q[i]);
    } else {  // simple revolute joint case
      q_pin[joint.idx_q()] = q[i];
    }

    dq[i] = initialize
      ? dq_meas
      : exponential_moving_average(dq[i], dq_meas, params_.filter.dq);

    q_target[i] = initialize ? q_meas : q_target[i];
  }
}

CallbackReturn
CartesianController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
  
  // Update the current state with initial measurements (no EMA filtering)
  // to avoid large initial errors
  updateCurrentState(true);

  pinocchio::forwardKinematics(model_, data_, q_pin, dq);
  pinocchio::updateFramePlacements(model_, data_);

  // Seed relative to base_frame (matches TF), see on_configure / update().
  end_effector_pose = (base_frame_id_ >= 0)
    ? data_.oMf[base_frame_id_].inverse() * data_.oMf[end_effector_frame_id]
    : data_.oMf[end_effector_frame_id];

  target_position_ = end_effector_pose.translation();
  target_orientation_ = Eigen::Quaterniond(end_effector_pose.rotation());
  desired_position_ = target_position_;
  desired_orientation_ = target_orientation_;

  RCLCPP_INFO(get_node()->get_logger(), "Controller activated.");
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
CartesianController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

void CartesianController::parse_target_pose_() {
  auto msg = *target_pose_buffer_.readFromRT();
  target_position_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  target_orientation_ = Eigen::Quaterniond(
    msg->pose.orientation.w,
    msg->pose.orientation.x,
    msg->pose.orientation.y,
    msg->pose.orientation.z);
}

void CartesianController::parse_target_joint_() {
  auto msg = *target_joint_buffer_.readFromRT();
  if (msg->position.size()) {
    for (size_t i = 0; i < msg->position.size(); ++i) {
      q_target[i] = msg->position[i];
    }
    /*filterJointValues(msg->name, msg->position, params_.joints, q_ref);*/
  }
  if (msg->velocity.size()) {
    for (size_t i = 0; i < msg->position.size(); ++i) {
      dq_ref[i] = msg->velocity[i];
    }
    /*filterJointValues(msg->name, msg->velocity, params_.joints, dq_ref);*/
  }
}

void CartesianController::parse_target_wrench_() {
  auto msg = *target_wrench_buffer_.readFromRT();
  target_wrench_ << msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z,
    msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z;
}

void CartesianController::parse_target_stiffness_() {
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
  const double max_k_trans = params_.variable_max_stiffness.translational;
  const double max_k_rot = params_.variable_max_stiffness.rotational;
  std::array<double, 6> vals = {msg->data[0], msg->data[1], msg->data[2],
                                msg->data[3], msg->data[4], msg->data[5]};
  for (int i = 0; i < 3; ++i) {
    if (vals[i] < 0.0 || vals[i] > max_k_trans) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Topic stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, vals[i], max_k_trans);
      vals[i] = std::clamp(vals[i], 0.0, max_k_trans);
    }
  }
  for (int i = 3; i < 6; ++i) {
    if (vals[i] < 0.0 || vals[i] > max_k_rot) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
        "Topic stiffness[%d]=%.1f out of [0, %.1f], clamping.", i, vals[i], max_k_rot);
      vals[i] = std::clamp(vals[i], 0.0, max_k_rot);
    }
  }
  topic_stiffness_.setZero();
  topic_stiffness_.diagonal() << vals[0], vals[1], vals[2], vals[3], vals[4], vals[5];
  use_topic_stiffness_ = true;
  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
    "Variable stiffness received: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
    vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
}

void CartesianController::log_debug_info(const rclcpp::Time & time) {
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
    /*RCLCPP_INFO_STREAM_THROTTLE(get_node()->get_logger(),
       * *get_node()->get_clock(),*/
    /*                            1000, "end_effector_rot" <<
       * end_effector_pose.rotation());*/
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "q: " << q.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "q_pin: " << q_pin.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "dq: " << dq.transpose());
    RCLCPP_INFO_STREAM_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "J: " << J);
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
    /*RCLCPP_INFO_STREAM_THROTTLE(get_node()->get_logger(),
       * *get_node()->get_clock(),*/
    /*                            1000, "Mx: " << Mx);*/
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

bool CartesianController::check_topic_publisher_count(const std::string & topic_name) {
  auto topic_info = get_node()->get_publishers_info_by_topic(topic_name);
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
  crisp_controllers::CartesianController, controller_interface::ControllerInterface)
