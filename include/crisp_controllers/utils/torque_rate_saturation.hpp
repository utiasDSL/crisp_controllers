// "Copyright [2026] <Analog Devices, Inc.>"
#pragma once

#include <Eigen/Core>

template <typename T>
inline Eigen::VectorXd saturateTorqueRate(
  const Eigen::VectorXd & tau_d_calculated,
  const Eigen::VectorXd & tau_J_d,
  const T & delta_tau_max) {
  Eigen::VectorXd difference = tau_d_calculated - tau_J_d;
  Eigen::VectorXd clamped_diff = difference.cwiseMin(delta_tau_max).cwiseMax(-delta_tau_max);
  return tau_J_d + clamped_diff;
}
