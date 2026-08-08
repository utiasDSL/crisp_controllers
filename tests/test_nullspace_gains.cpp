#include <gtest/gtest.h>
#include "crisp_controllers/utils/nullspace_gains.hpp"

using crisp_controllers::expand_nullspace_gains;
using crisp_controllers::resolve_nullspace_damping;

TEST(ExpandNullspaceGainsTest, SingleValueIsBroadcast) {
  auto gains = expand_nullspace_gains({5.0}, 3);

  ASSERT_TRUE(gains.has_value());

  Eigen::VectorXd expected(3);
  expected << 5.0, 5.0, 5.0;
  EXPECT_TRUE(gains->isApprox(expected));
}

TEST(ExpandNullspaceGainsTest, OneValuePerDegreeOfFreedomIsKept) {
  auto gains = expand_nullspace_gains({5.0, 4.0, 3.0}, 3);

  ASSERT_TRUE(gains.has_value());

  Eigen::VectorXd expected(3);
  expected << 5.0, 4.0, 3.0;
  EXPECT_TRUE(gains->isApprox(expected));
}

TEST(ExpandNullspaceGainsTest, MismatchedSizeIsRejected) {
  EXPECT_FALSE(expand_nullspace_gains({5.0, 4.0}, 3).has_value());
  EXPECT_FALSE(expand_nullspace_gains({5.0, 4.0, 3.0, 2.0}, 3).has_value());
  EXPECT_FALSE(expand_nullspace_gains({}, 3).has_value());
}

TEST(ResolveNullspaceDampingTest, NegativeEntriesBecomeCriticalDamping) {
  Eigen::VectorXd damping(3), stiffness(3);
  damping << -1.0, 0.0, 3.0;
  stiffness << 25.0, 16.0, 9.0;

  Eigen::VectorXd expected(3);
  expected << 10.0, 0.0, 3.0;

  EXPECT_TRUE(resolve_nullspace_damping(damping, stiffness).isApprox(expected));
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
