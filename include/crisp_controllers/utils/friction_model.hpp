// "Copyright [2026] <Analog Devices, Inc.>"
#pragma once
// Based in
// https://github.com/marcocognetti/FrankaEmikaPandaDynModel/tree/master

#include <Eigen/src/Core/GlobalFunctions.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/Core>

inline Eigen::VectorXd get_friction(
  const Eigen::VectorXd & dq, Eigen::VectorXd fp1, Eigen::VectorXd fp2, Eigen::VectorXd fp3) {
  return (fp1.array() /
            (Eigen::VectorXd::Ones(dq.size()).array() +
             (-fp2.array() * (dq.array() + fp3.array())).exp()) -
          fp1.array() /
            (Eigen::VectorXd::Ones(dq.size()).array() + (-fp2.array() * fp3.array()).exp()))
    .matrix();
}
