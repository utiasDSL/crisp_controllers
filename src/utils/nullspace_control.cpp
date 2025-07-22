#include <crisp_controllers/utils/nullspace_control.hpp>
#include <crisp_controllers/utils/pseudo_inverse.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <rclcpp/logging.hpp>
#include <cmath>
#include <algorithm>

namespace crisp_controllers {
namespace nullspace_control {

Eigen::MatrixXd computeNullspaceProjection(
    const Eigen::MatrixXd& jacobian,
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const std::string& projector_type,
    double regularization,
    rclcpp::Logger logger) {
  
  Eigen::MatrixXd Id_nv = Eigen::MatrixXd::Identity(model.nv, model.nv);

  if (projector_type == "dynamic") {
    pinocchio::computeMinverse(model, data, q);
    auto Mx_inv = jacobian * data.Minv * jacobian.transpose();
    auto Mx = pseudo_inverse(Mx_inv);
    auto J_bar = data.Minv * jacobian.transpose() * Mx;
    return Id_nv - jacobian.transpose() * J_bar.transpose();
  } else if (projector_type == "kinematic") {
    Eigen::MatrixXd J_pinv = pseudo_inverse(jacobian, regularization);
    return Id_nv - J_pinv * jacobian;
  } else if (projector_type == "none") {
    return Id_nv;
  } else {
    RCLCPP_ERROR_STREAM_ONCE(logger,
                        "Unknown nullspace projector type: "
                            << projector_type);
    return Id_nv;
  }
}

Eigen::VectorXd computeNullspaceControl(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& q_init,
    const Eigen::MatrixXd& jacobian,
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const NullspaceParams& params,
    const Eigen::VectorXd& nullspace_weights,
    rclcpp::Logger logger) {
  
  Eigen::VectorXd q_error = q - q_init;
  double nullspace_damping = params.damping > 0 ? params.damping : 2.0 * sqrt(params.stiffness);

  Eigen::VectorXd tau_secondary = -params.stiffness * nullspace_weights.cwiseProduct(q_error) - 
                                  nullspace_damping * nullspace_weights.cwiseProduct(dq);

  Eigen::MatrixXd nullspace_projection = computeNullspaceProjection(
      jacobian, model, data, q, params.projector_type, params.regularization, logger);
  
  Eigen::VectorXd tau_nullspace = nullspace_projection * tau_secondary;
  
  for (int i = 0; i < tau_nullspace.size(); i++) {
    tau_nullspace[i] = std::max(-params.max_tau, std::min(params.max_tau, tau_nullspace[i]));
  }
  
  return tau_nullspace;
}

} // namespace nullspace_control
} // namespace crisp_controllers