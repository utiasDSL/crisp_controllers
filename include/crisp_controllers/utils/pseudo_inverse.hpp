// "Copyright [2026] <Analog Devices, Inc.>"
#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace crisp_controllers {

inline Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd & matrix, double epsilon = 1e-4) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd singularValues = svd.singularValues();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().rows());
  for (int i = 0; i < singularValues.size(); ++i) {
    S(i, i) = singularValues[i] / (singularValues[i] * singularValues[i] + epsilon * epsilon);
  }

  return svd.matrixV() * S * svd.matrixU().transpose();
}

inline Eigen::MatrixXd
pseudo_inverse_moore_penrose(const Eigen::MatrixXd & matrix, double epsilon = 1e-6) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd singularValues = svd.singularValues();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().rows());
  for (int i = 0; i < singularValues.size(); ++i) {
    S(i, i) = singularValues[i] > epsilon ? 1.0 / singularValues[i] : 0.0;
  }

  return svd.matrixV() * S * svd.matrixU().transpose();
}

inline bool is_near_singular(const Eigen::MatrixXd & matrix, double epsilon = 1e-6) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd singularValues = svd.singularValues();
  return (singularValues.array() < epsilon).any();
}

}  // namespace crisp_controllers
