// "Copyright [2026] <Analog Devices, Inc.>"

#include <gtest/gtest.h>
#include "crisp_controllers/utils/torque_rate_saturation.hpp"

TEST(SaturateTorqueRateTest, ScalarDeltaTauMax) {
  Eigen::VectorXd tau_d(3), tau_J(3);
  tau_d << 2.0, -2.0, 0.5;
  tau_J << 0.0, 0.0, 0.0;

  double delta_tau_max = 1.0;
  Eigen::VectorXd result = saturateTorqueRate(tau_d, tau_J, delta_tau_max);

  Eigen::VectorXd expected(3);
  expected << 1.0, -1.0, 0.5;

  EXPECT_TRUE(result.isApprox(expected));
}

TEST(SaturateTorqueRateTest, VectorDeltaTauMax) {
  Eigen::VectorXd tau_d(3), tau_J(3), delta_tau_max(3);
  tau_d << 3.0, -3.0, 0.5;
  tau_J << 0.0, 0.0, 0.0;
  delta_tau_max << 2.0, 1.0, 1.0;

  Eigen::VectorXd result = saturateTorqueRate(tau_d, tau_J, delta_tau_max);

  Eigen::VectorXd expected(3);
  expected << 2.0, -1.0, 0.5;

  EXPECT_TRUE(result.isApprox(expected));
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
