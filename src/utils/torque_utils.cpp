#include <crisp_controllers/utils/torque_utils.hpp>
#include <algorithm>

namespace crisp_controllers {
namespace torque_utils {

Eigen::VectorXd applyTorqueThreshold(
    const Eigen::VectorXd& tau_ext,
    double torque_threshold) {
  
  Eigen::VectorXd tau_ext_thresholded = Eigen::VectorXd::Zero(tau_ext.size());
  double tau_ext_magnitude = tau_ext.norm();
  if (tau_ext_magnitude > torque_threshold) {
    tau_ext_thresholded = tau_ext;
  }
  return tau_ext_thresholded;
}

void limitTorques(
    Eigen::VectorXd& torques,
    double max_torque) {
  
  for (int i = 0; i < torques.size(); i++) {
    torques[i] = std::max(-max_torque, std::min(max_torque, torques[i]));
  }
}

} // namespace torque_utils
} // namespace crisp_controllers