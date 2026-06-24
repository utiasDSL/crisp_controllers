#pragma once

/**
 * @file state_broadcaster.hpp
 * @brief State-only ROS 2 controller that publishes robot pose, twist, wrench, residual effort,
 * and joint state from hardware state interfaces.
 */

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include <controller_interface/controller_interface.hpp>
#include <crisp_controllers/utils/ros2_version.hpp>

#if ROS2_VERSION_ABOVE_HUMBLE
#include <crisp_controllers/state_broadcaster_parameters.hpp>
#else
#include <state_broadcaster_parameters.hpp>
#endif

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace crisp_controllers {

/**
 * @brief ROS 2 controller_interface plugin for broadcasting robot state estimates.
 *
 * StateBroadcaster consumes state interfaces only and publishes end-effector pose, optional twist,
 * optional raw wrench estimates, optional residual effort after subtracting enabled model terms,
 * optional wrench estimates projected from residual effort, and joint state messages. Pinocchio is
 * used for kinematics, frame Jacobians, model terms, and wrench estimation. Outputs that require
 * velocity or effort state interfaces are enabled only when the complete corresponding interface
 * group is available.
 */
class StateBroadcaster : public controller_interface::ControllerInterface {
public:
  /**
   * @brief Declare that this broadcaster does not claim command interfaces.
   * @return Interface configuration with type NONE.
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  /**
   * @brief Declare that this broadcaster reads available state interfaces.
   * @return Interface configuration with type ALL.
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  /**
   * @brief Read state interfaces, update Pinocchio state, and publish configured outputs.
   * @param time Current controller manager time used for message stamps.
   * @param period Time elapsed since the previous update.
   * @return OK when the update completes.
   */
  controller_interface::return_type
  update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  /**
   * @brief Initialize generated parameter handling.
   * @return Lifecycle callback result.
   */
  CallbackReturn on_init() override;

  /**
   * @brief Validate parameters, build the reduced Pinocchio model, and create publishers.
   * @param previous_state Previous lifecycle state.
   * @return Lifecycle callback result.
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Cache state interface indices and initialize publication state.
   * @param previous_state Previous lifecycle state.
   * @return Lifecycle callback result.
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Deactivate the broadcaster.
   * @param previous_state Previous lifecycle state.
   * @return Lifecycle callback result.
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

private:
  /** @brief Sentinel index used when a requested state interface is unavailable. */
  static constexpr size_t kMissingInterface = static_cast<size_t>(-1);

  /** @brief Accumulated elapsed time and configured interval for a periodic output. */
  struct PublishTimer {
    /** @brief Time accumulated since this output last published. */
    rclcpp::Duration elapsed{0, 0};
    /** @brief Desired interval between publications; zero means publish every update. */
    rclcpp::Duration interval{0, 0};
  };

  /** @brief Generated parameter listener for state broadcaster parameters. */
  std::shared_ptr<state_broadcaster::ParamListener> params_listener_;
  /** @brief Cached parameter values used by the broadcaster. */
  state_broadcaster::Params params_;

  /** @brief Publisher for end-effector pose messages. */
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  /** @brief Realtime publisher for end-effector pose messages. */
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>>
    realtime_pose_publisher_;
  /** @brief Preallocated end-effector pose message. */
  geometry_msgs::msg::PoseStamped pose_msg_;

  /** @brief Publisher for end-effector twist messages. */
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_publisher_;
  /** @brief Realtime publisher for end-effector twist messages. */
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::TwistStamped>>
    realtime_twist_publisher_;
  /** @brief Preallocated end-effector twist message. */
  geometry_msgs::msg::TwistStamped twist_msg_;

  /** @brief Publisher for wrench estimated directly from measured joint effort. */
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr raw_wrench_publisher_;
  /** @brief Realtime publisher for wrench estimated directly from measured joint effort. */
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
    realtime_raw_wrench_publisher_;
  /** @brief Preallocated raw wrench message. */
  geometry_msgs::msg::WrenchStamped raw_wrench_msg_;

  /** @brief Publisher for wrench estimated from residual effort. */
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr external_wrench_publisher_;
  /** @brief Realtime publisher for wrench estimated from residual effort. */
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
    realtime_external_wrench_publisher_;
  /** @brief Preallocated external wrench message. */
  geometry_msgs::msg::WrenchStamped external_wrench_msg_;

  /** @brief Publisher for residual joint effort after subtracting enabled model terms. */
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr external_effort_publisher_;
  /** @brief Realtime publisher for residual joint effort after subtracting enabled model terms. */
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>
    realtime_external_effort_publisher_;
  /** @brief Preallocated external effort message. */
  std_msgs::msg::Float64MultiArray external_effort_msg_;

  /** @brief Publisher for joint state messages assembled from available state interfaces. */
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
  /** @brief Realtime publisher for joint state messages. */
  std::shared_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>
    realtime_joint_state_publisher_;
  /** @brief Preallocated joint state message. */
  sensor_msgs::msg::JointState joint_state_msg_;

  /** @brief Pinocchio frame index for the configured end-effector frame. */
  pinocchio::FrameIndex end_effector_frame_id_;
  /** @brief Reference frame used when computing and publishing end-effector twist. */
  pinocchio::ReferenceFrame twist_reference_frame_{pinocchio::ReferenceFrame::LOCAL};
  /** @brief Reference frame used when estimating raw wrench. */
  pinocchio::ReferenceFrame raw_wrench_reference_frame_{
    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED};
  /** @brief Reference frame used when estimating external wrench. */
  pinocchio::ReferenceFrame external_wrench_reference_frame_{
    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED};

  /** @brief Reduced Pinocchio model containing the configured joints. */
  pinocchio::Model model_;
  /** @brief Pinocchio data workspace for kinematics, dynamics, and Jacobian computations. */
  pinocchio::Data data_;

  /** @brief Pinocchio joint model short names accepted by this broadcaster. */
  const std::unordered_set<std::basic_string<char>> allowed_joint_types_ = {
    "JointModelRX",
    "JointModelRY",
    "JointModelRZ",
    "JointModelRevoluteUnaligned",
    "JointModelRUBX",
    "JointModelRUBY",
    "JointModelRUBZ",
  };
  /** @brief Pinocchio continuous joint model short names represented with cosine/sine positions. */
  const std::unordered_set<std::basic_string<char>> continuous_joint_types_ = {
    "JointModelRUBX", "JointModelRUBY", "JointModelRUBZ"};

  /** @brief Pinocchio q-vector indices for configured joints. */
  std::vector<Eigen::Index> joint_q_indices_;
  /** @brief Pinocchio velocity and effort vector indices for configured joints. */
  std::vector<Eigen::Index> joint_v_indices_;
  /** @brief Flags indicating which configured joints use cosine/sine position representation. */
  std::vector<bool> joint_is_continuous_;
  /** @brief Controller state interface indices for joint position values. */
  std::vector<size_t> position_interface_indices_;
  /** @brief Controller state interface indices for joint velocity values. */
  std::vector<size_t> velocity_interface_indices_;
  /** @brief Controller state interface indices for joint effort values. */
  std::vector<size_t> effort_interface_indices_;
  /** @brief True when every configured joint has a velocity state interface. */
  bool has_velocity_interfaces_{false};
  /** @brief True when every configured joint has an effort state interface. */
  bool has_effort_interfaces_{false};

  /** @brief Current configured joint positions in parameter order. */
  Eigen::VectorXd q_;
  /** @brief Current Pinocchio configuration vector. */
  Eigen::VectorXd q_pin_;
  /** @brief Current Pinocchio velocity vector. */
  Eigen::VectorXd dq_;
  /** @brief Previous Pinocchio velocity vector used for acceleration estimation. */
  Eigen::VectorXd dq_previous_;
  /** @brief Estimated joint acceleration before filtering. */
  Eigen::VectorXd ddq_estimated_;
  /** @brief Filtered joint acceleration used for inertial compensation. */
  Eigen::VectorXd ddq_filtered_;
  /** @brief Measured joint effort vector read from state interfaces. */
  Eigen::VectorXd tau_measured_;
  /** @brief Coriolis effort compensation vector. */
  Eigen::VectorXd tau_coriolis_;
  /** @brief Gravity effort compensation vector. */
  Eigen::VectorXd tau_gravity_;
  /** @brief Inertial effort compensation vector. */
  Eigen::VectorXd tau_inertia_;
  /** @brief Residual effort after subtracting enabled model compensation terms. */
  Eigen::VectorXd tau_residual_;
  /** @brief End-effector frame Jacobian workspace. */
  pinocchio::Data::Matrix6x J_;
  /** @brief Regularized wrench least-squares system matrix. */
  Eigen::Matrix<double, 6, 6> wrench_system_;
  /** @brief Wrench least-squares right-hand side vector. */
  Eigen::Matrix<double, 6, 1> wrench_rhs_;
  /** @brief Wrench estimated from measured joint effort. */
  Eigen::Matrix<double, 6, 1> raw_wrench_;
  /** @brief Wrench estimated from residual joint effort. */
  Eigen::Matrix<double, 6, 1> external_wrench_;
  /** @brief True after the previous velocity vector has been initialized. */
  bool has_previous_velocity_{false};

  /** @brief Publish timer for pose output. */
  PublishTimer pose_timer_;
  /** @brief Publish timer for twist output. */
  PublishTimer twist_timer_;
  /** @brief Publish timer for raw wrench output. */
  PublishTimer raw_wrench_timer_;
  /** @brief Publish timer for external wrench output. */
  PublishTimer external_wrench_timer_;
  /** @brief Publish timer for external effort output. */
  PublishTimer external_effort_timer_;
  /** @brief Publish timer for joint state output. */
  PublishTimer joint_state_timer_;

  /**
   * @brief Parse an output reference-frame parameter into Pinocchio and ROS frame settings.
   * @param output_name Output name used in error messages.
   * @param reference_frame Parameter value to parse.
   * @param parsed_reference_frame Parsed Pinocchio reference frame.
   * @param published_frame Frame ID associated with the parsed reference frame.
   * @return True when the reference frame value is supported.
   */
  bool configure_reference_frame(
    const std::string & output_name, const std::string & reference_frame,
    pinocchio::ReferenceFrame & parsed_reference_frame, std::string & published_frame) const;

  /**
   * @brief Convert an output publish frequency parameter into a publish interval.
   * @param output_name Output name used in error messages.
   * @param publish_frequency Publish frequency in hertz; zero publishes every update.
   * @param timer Timer receiving the configured interval.
   * @return True when the frequency is valid.
   */
  bool configure_publish_interval(
    const std::string & output_name, double publish_frequency, PublishTimer & timer) const;

  /** @brief Warn when deprecated parameters are supplied. */
  void warn_legacy_parameters() const;

  /**
   * @brief Validate common parameters required before model and publisher setup.
   * @return True when required joints, frames, and topics are configured.
   */
  bool validate_common_parameters() const;

  /**
   * @brief Build the reduced Pinocchio model and allocate model-sized work buffers.
   * @return True when the robot description and configured frames/joints are valid.
   */
  bool build_model();

  /**
   * @brief Cache Pinocchio position and velocity indices for configured joints.
   * @return True when indices are cached.
   */
  bool cache_joint_model_indices();

  /**
   * @brief Cache controller state interface indices and detect optional interface groups.
   * @return True when all required position state interfaces are available.
   */
  bool cache_state_interface_indices();

  /** @brief Read joint position state interfaces into the configured joint vector. */
  void read_position_interfaces();

  /** @brief Read joint velocity state interfaces into the Pinocchio velocity vector. */
  void read_velocity_interfaces();

  /** @brief Read joint effort state interfaces into the measured effort vector. */
  void read_effort_interfaces();

  /** @brief Convert configured joint positions into Pinocchio configuration representation. */
  void update_pinocchio_positions();

  /**
   * @brief Estimate a spatial wrench from joint effort using the end-effector Jacobian.
   * @param effort Joint effort vector in Pinocchio velocity order.
   * @param reference_frame Pinocchio reference frame for the frame Jacobian.
   * @param wrench Output spatial wrench vector.
   */
  void compute_wrench(const Eigen::VectorXd & effort, pinocchio::ReferenceFrame reference_frame,
                      Eigen::Matrix<double, 6, 1> & wrench);

  /** @brief Copy residual efforts into the preallocated external effort message. */
  void update_external_effort_message();

  /**
   * @brief Advance a publish timer and report whether its output should publish this update.
   * @param period Time elapsed since the previous update.
   * @param timer Publish timer to update.
   * @return True when the output should publish.
   */
  bool should_publish(const rclcpp::Duration & period, PublishTimer & timer);
};

}  // namespace crisp_controllers
