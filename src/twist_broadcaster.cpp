#include <binders.h>
#include <cassert>
#include <cmath>
#include <memory>

#include <Eigen/src/Core/Matrix.h>  // NOLINT(build/include_order)
#include <crisp_controllers/twist_broadcaster.hpp>
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
TwistBroadcaster::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::InterfaceConfiguration
TwistBroadcaster::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint_name : params_.joints) {
    config.names.push_back(joint_name + "/position");
    config.names.push_back(joint_name + "/velocity");
  }
  return config;
}

controller_interface::return_type
TwistBroadcaster::update(const rclcpp::Time & time, const rclcpp::Duration & period) {
  size_t num_joints = params_.joints.size();
  Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
  Eigen::VectorXd q_dot_pin = Eigen::VectorXd::Zero(model_.nv);

  for (size_t i = 0; i < num_joints; i++) {
    auto joint_name = params_.joints[i];
    auto joint_id = model_.getJointId(joint_name);
    auto joint = model_.joints[joint_id];

#if ROS2_VERSION_ABOVE_HUMBLE
    q[i] = state_interfaces_[i * 2].get_optional().value_or(q[i]);
    q_dot[i] = state_interfaces_[i * 2 + 1].get_optional().value_or(q_dot[i]);
#else
    q[i] = state_interfaces_[i * 2].get_value();
    q_dot[i] = state_interfaces_[i * 2 + 1].get_value();
#endif

    if (continous_joint_types.count(
          joint.shortname())) {  // Then we are handling a continous joint that is SO(2)
      q_pin[joint.idx_q()] = std::cos(q[i]);
      q_pin[joint.idx_q() + 1] = std::sin(q[i]);
      q_dot_pin[joint.idx_v()] = q_dot[i];
    } else {
      q_pin[joint.idx_q()] = q[i];
      q_dot_pin[joint.idx_v()] = q_dot[i];
    }
  }

  pinocchio::forwardKinematics(model_, data_, q_pin, q_dot_pin);
  pinocchio::updateFramePlacements(model_, data_);

  auto current_velocity = pinocchio::getFrameVelocity(model_, data_, end_effector_frame_id);

  // Decide whether to publish the twist or not
  publish_elapsed_ = publish_elapsed_ + period;
  bool should_publish =
    (publish_elapsed_ >= publish_interval_) || (publish_interval_.nanoseconds() == 0);

  if (should_publish && realtime_twist_publisher_) {
    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.stamp = time;
    twist_msg.header.frame_id = params_.end_effector_frame;
    twist_msg.twist.linear.x = current_velocity.linear()[0];
    twist_msg.twist.linear.y = current_velocity.linear()[1];
    twist_msg.twist.linear.z = current_velocity.linear()[2];
    twist_msg.twist.angular.x = current_velocity.angular()[0];
    twist_msg.twist.angular.y = current_velocity.angular()[1];
    twist_msg.twist.angular.z = current_velocity.angular()[2];

#if REALTIME_TOOLS_NEW_API
    if (realtime_twist_publisher_->try_publish(twist_msg)) {
#else
    if (realtime_twist_publisher_->trylock()) {
      realtime_twist_publisher_->msg_ = twist_msg;
      realtime_twist_publisher_->unlockAndPublish();
#endif
      publish_elapsed_ = publish_elapsed_ - publish_interval_;
      // clamp to publish only 1 time even if missed multiple intervals
      publish_elapsed_ = std::min(publish_elapsed_, publish_interval_);
    }
  }

  return controller_interface::return_type::OK;
}

CallbackReturn TwistBroadcaster::on_init() {
  // Initialize parameters
  params_listener_ = std::make_shared<twist_broadcaster::ParamListener>(get_node());
  params_listener_->refresh_dynamic_parameters();
  params_ = params_listener_->get_params();

  return CallbackReturn::SUCCESS;
}

CallbackReturn TwistBroadcaster::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {
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
  q_dot = Eigen::VectorXd::Zero(model_.nv);

  twist_publisher_ = get_node()->create_publisher<geometry_msgs::msg::TwistStamped>(
    "current_twist", rclcpp::SystemDefaultsQoS());
  realtime_twist_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::TwistStamped>>(
      twist_publisher_);

  if (params_.publish_frequency > 0.0) {
    publish_interval_ = rclcpp::Duration::from_seconds(1.0 / params_.publish_frequency);
  } else {
    publish_interval_ = rclcpp::Duration(0, 0);  // publish every cycle
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn TwistBroadcaster::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
  // reset publish time accumulation
  publish_elapsed_ = rclcpp::Duration(0, 0);
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
TwistBroadcaster::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/) {
  return CallbackReturn::SUCCESS;
}

}  // namespace crisp_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  crisp_controllers::TwistBroadcaster, controller_interface::ControllerInterface)
