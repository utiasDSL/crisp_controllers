#pragma once

#include <string>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

namespace crisp_controllers {
namespace nullspace_control {

struct NullspaceParams {
    double stiffness;
    double damping;
    double max_tau;
    double regularization;
    std::string projector_type;
};

Eigen::MatrixXd computeNullspaceProjection(
    const Eigen::MatrixXd& jacobian,
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const std::string& projector_type,
    double regularization,
    rclcpp::Logger logger);

Eigen::VectorXd computeNullspaceControl(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& q_init,
    const Eigen::MatrixXd& jacobian,
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const NullspaceParams& params,
    const Eigen::VectorXd& nullspace_weights,
    rclcpp::Logger logger);

} // namespace nullspace_control
} // namespace crisp_controllers