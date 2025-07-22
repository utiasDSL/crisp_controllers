#pragma once

#include <Eigen/Dense>

namespace crisp_controllers {
namespace torque_utils {

Eigen::VectorXd applyTorqueThreshold(
    const Eigen::VectorXd& tau_ext,
    double torque_threshold);

void limitTorques(
    Eigen::VectorXd& torques,
    double max_torque);

} // namespace torque_utils
} // namespace crisp_controllers