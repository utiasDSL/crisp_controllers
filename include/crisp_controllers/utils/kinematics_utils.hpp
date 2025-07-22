#pragma once

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <rclcpp/rclcpp.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <Eigen/Dense>

namespace crisp_controllers {
namespace kinematics_utils {

void computeForwardKinematicsAndJacobian(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    Eigen::MatrixXd& jacobian,
    pinocchio::FrameIndex end_effector_frame_id);

bool setupPinocchioModel(
    const std::string& robot_description,
    const std::vector<std::string>& joint_names,
    pinocchio::Model& model,
    pinocchio::Data& data,
    rclcpp::Logger logger);

bool validateJointTypes(
    const pinocchio::Model& model,
    rclcpp::Logger logger);

} // namespace kinematics_utils
} // namespace crisp_controllers