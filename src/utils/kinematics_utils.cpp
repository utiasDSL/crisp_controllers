#include <crisp_controllers/utils/kinematics_utils.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <rclcpp/logging.hpp>
#include <algorithm>

namespace crisp_controllers {
namespace kinematics_utils {

void computeForwardKinematicsAndJacobian(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    Eigen::MatrixXd& jacobian,
    pinocchio::FrameIndex end_effector_frame_id) {
  
  pinocchio::forwardKinematics(model, data, q, dq);
  pinocchio::updateFramePlacements(model, data);

  jacobian.setZero();
  pinocchio::computeFrameJacobian(model, data, q, end_effector_frame_id,
                                  pinocchio::ReferenceFrame::LOCAL, jacobian);
}

bool setupPinocchioModel(
    const std::string& robot_description,
    const std::vector<std::string>& joint_names,
    pinocchio::Model& model,
    pinocchio::Data& data,
    rclcpp::Logger logger) {
  
  pinocchio::Model raw_model;
  pinocchio::urdf::buildModelFromXML(robot_description, raw_model);

  for (const auto& joint : joint_names) {
    if (!raw_model.existJointName(joint)) {
      RCLCPP_ERROR_STREAM(logger,
                          "Failed to configure because "
                              << joint
                              << " is not part of the kinematic tree but it "
                                 "has been passed in the parameters.");
      return false;
    }
  }

  std::vector<pinocchio::JointIndex> list_of_joints_to_lock_by_id;
  for (const auto& joint : raw_model.names) {
    if (std::find(joint_names.begin(), joint_names.end(), joint) ==
            joint_names.end() &&
        joint != "universe") {
      list_of_joints_to_lock_by_id.push_back(raw_model.getJointId(joint));
    }
  }

  Eigen::VectorXd q_locked = Eigen::VectorXd::Zero(raw_model.nq);
  model = pinocchio::buildReducedModel(raw_model,
                                       list_of_joints_to_lock_by_id, q_locked);
  data = pinocchio::Data(model);
  
  return true;
}

bool validateJointTypes(
    const pinocchio::Model& model,
    rclcpp::Logger logger) {
  
  std::set<std::string> allowed_joint_types = {"JointModelRZ", "JointModelRevoluteUnaligned", "JointModelRX", "JointModelRY"};

  for (int joint_id = 0; joint_id < model.njoints; joint_id++) {
    if (model.names[joint_id] == "universe") {
      continue;
    }
    if (!allowed_joint_types.count(model.joints[joint_id].shortname())) {
      RCLCPP_ERROR_STREAM(
          logger,
          "Joint type "
              << model.joints[joint_id].shortname() << " is unsupported ("
              << model.names[joint_id]
              << "). Continuous joints are not implemented yet for torque feedback controller. "
              << "Only revolute joints are supported.");
      return false;
    }
  }
  
  return true;
}

} // namespace kinematics_utils
} // namespace crisp_controllers
