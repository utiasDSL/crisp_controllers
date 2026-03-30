#include <binders.h>
#include <cassert>
#include <cmath>
#include <memory>

#include <Eigen/src/Core/Matrix.h>  // NOLINT(build/include_order)
#include <crisp_controllers/pose_broadcaster.hpp>
#include "crisp_controllers/utils/ros2_version.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/logging.hpp>

using namespace std::chrono_literals;

namespace crisp_controllers {

controller_interface::InterfaceConfiguration
PoseBroadcaster::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::InterfaceConfiguration
PoseBroadcaster::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint_name : params_.joints) {
    config.names.push_back(joint_name + "/position");
  }
  return config;
}

controller_interface::return_type
PoseBroadcaster::update(const rclcpp::Time & time, const rclcpp::Duration & period) {
  size_t num_joints = params_.joints.size();
  Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);

  for (size_t i = 0; i < num_joints; i++) {
    auto joint_name = params_.joints[i];
    auto joint_id = model_.getJointId(joint_name);
    auto joint = model_.joints[joint_id];

#if ROS2_VERSION_ABOVE_HUMBLE
    q[i] = state_interfaces_[i].get_optional().value_or(q[i]);
#else
    q[i] = state_interfaces_[i].get_value();
#endif
    if (continous_joint_types.count(
          joint.shortname())) {  // Then we are handling a continous joint that is SO(2)
      q_pin[joint.idx_q()] = std::cos(q[i]);
      q_pin[joint.idx_q() + 1] = std::sin(q[i]);
    } else {
      q_pin[joint.idx_q()] = q[i];
    }
  }

  pinocchio::forwardKinematics(model_, data_, q_pin);
  pinocchio::updateFramePlacements(model_, data_);

  auto current_pose = data_.oMf[end_effector_frame_id];
  auto current_quaternion = Eigen::Quaterniond(current_pose.rotation());

  // Decide whether to publish the pose or not
  publish_elapsed_ = publish_elapsed_ + period;
  bool should_publish =
    (publish_elapsed_ >= publish_interval_) || (publish_interval_.nanoseconds() == 0);
  if (should_publish && realtime_pose_publisher_) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = time;
    pose_msg.header.frame_id = params_.base_frame;
    pose_msg.pose.position.x = current_pose.translation()[0];
    pose_msg.pose.position.y = current_pose.translation()[1];
    pose_msg.pose.position.z = current_pose.translation()[2];
    pose_msg.pose.orientation.x = current_quaternion.x();
    pose_msg.pose.orientation.y = current_quaternion.y();
    pose_msg.pose.orientation.z = current_quaternion.z();
    pose_msg.pose.orientation.w = current_quaternion.w();

#if REALTIME_TOOLS_NEW_API
    if (realtime_pose_publisher_->try_publish(pose_msg)) {
#else
    if (realtime_pose_publisher_->trylock()) {
      realtime_pose_publisher_->msg_ = pose_msg;
      realtime_pose_publisher_->unlockAndPublish();
#endif
      publish_elapsed_ = publish_elapsed_ - publish_interval_;
      // clamp to publish only 1 time even if missed multiple intervals
      publish_elapsed_ = std::min(publish_elapsed_, publish_interval_);
    }
  }

  return controller_interface::return_type::OK;
}

CallbackReturn PoseBroadcaster::on_init() {
  // Initialize parameters
  params_listener_ = std::make_shared<pose_broadcaster::ParamListener>(get_node());
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();

  return CallbackReturn::SUCCESS;
}

CallbackReturn PoseBroadcaster::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {
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
        "Failed to configure because "
          << joint
          << " is not part of the kinematic tree but it has been passed in the parameters.");
      return CallbackReturn::ERROR;
    }
  }
  RCLCPP_INFO(
    get_node()->get_logger(),
    "All joints passed in the parameters exist in the kinematic tree of the URDF.");
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

  end_effector_frame_id = model_.getFrameId(params_.end_effector_frame);
  q = Eigen::VectorXd::Zero(model_.nv);

  pose_publisher_ = get_node()->create_publisher<geometry_msgs::msg::PoseStamped>(
    "current_pose", rclcpp::SystemDefaultsQoS());
  realtime_pose_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>>(
      pose_publisher_);

  if (params_.publish_frequency > 0.0) {
    publish_interval_ = rclcpp::Duration::from_seconds(1.0 / params_.publish_frequency);
  } else {
    publish_interval_ = rclcpp::Duration(0, 0);  // publish every cycle
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn PoseBroadcaster::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
  // reset publish time accumulation
  publish_elapsed_ = rclcpp::Duration(0, 0);
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
PoseBroadcaster::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

}  // namespace crisp_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  crisp_controllers::PoseBroadcaster, controller_interface::ControllerInterface)
