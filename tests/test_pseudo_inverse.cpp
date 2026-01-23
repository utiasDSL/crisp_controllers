// "Copyright [2026] <Analog Devices, Inc.>"
#include <gtest/gtest.h>
#include "crisp_controllers/utils/pseudo_inverse.hpp"

TEST(PseudoInverseTest, SquareMatrixTest) {
  Eigen::MatrixXd A(2, 2);
  A << 4, 7, 2, 6;

  Eigen::MatrixXd A_pinv = crisp_controllers::pseudo_inverse(A);

  // Test if A * A_pinv * A ≈ A
  Eigen::MatrixXd result = A * A_pinv * A;
  EXPECT_TRUE(result.isApprox(A, 1e-5)) << "Square Matrix Test Failed!\n"
                                        << "Original matrix A:\n"
                                        << A << "\n"
                                        << "Pseudo-inverse A_pinv:\n"
                                        << A_pinv << "\n"
                                        << "Result (A * A_pinv * A):\n"
                                        << result << "\n"
                                        << "Expected (A):\n"
                                        << A << "\n"
                                        << "Difference:\n"
                                        << (result - A) << "\n";
}

TEST(PseudoInverseTest, RectangularMatrixTest) {
  Eigen::MatrixXd B(3, 2);
  B << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXd B_pinv = crisp_controllers::pseudo_inverse(B);

  Eigen::MatrixXd result = B * B_pinv * B;
  EXPECT_TRUE(result.isApprox(B, 1e-5)) << "Rectangular Matrix Test Failed!\n"
                                        << "Original matrix B:\n"
                                        << B << "\n"
                                        << "Pseudo-inverse B_pinv:\n"
                                        << B_pinv << "\n"
                                        << "Result (B * B_pinv * B):\n"
                                        << result << "\n"
                                        << "Expected (B):\n"
                                        << B << "\n"
                                        << "Difference:\n"
                                        << (result - B) << "\n";
}

TEST(PseudoInverseTest, MoorePenroseTest) {
  Eigen::MatrixXd C(2, 3);
  C << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXd C_pinv = crisp_controllers::pseudo_inverse_moore_penrose(C);

  // Test Moore-Penrose conditions
  // 1. C * C_pinv * C = C
  Eigen::MatrixXd condition1 = C * C_pinv * C;
  EXPECT_TRUE(condition1.isApprox(C, 1e-5)) << "Moore-Penrose Condition 1 Failed!\n"
                                            << "Original matrix C:\n"
                                            << C << "\n"
                                            << "Pseudo-inverse C_pinv:\n"
                                            << C_pinv << "\n"
                                            << "Result (C * C_pinv * C):\n"
                                            << condition1 << "\n"
                                            << "Expected (C):\n"
                                            << C << "\n"
                                            << "Difference:\n"
                                            << (condition1 - C) << "\n";

  // 2. C_pinv * C * C_pinv = C_pinv
  Eigen::MatrixXd condition2 = C_pinv * C * C_pinv;
  EXPECT_TRUE(condition2.isApprox(C_pinv, 1e-5)) << "Moore-Penrose Condition 2 Failed!\n"
                                                 << "Result (C_pinv * C * C_pinv):\n"
                                                 << condition2 << "\n"
                                                 << "Expected (C_pinv):\n"
                                                 << C_pinv << "\n"
                                                 << "Difference:\n"
                                                 << (condition2 - C_pinv) << "\n";

  // 3. (C * C_pinv)' = C * C_pinv
  Eigen::MatrixXd condition3 = C * C_pinv;
  Eigen::MatrixXd condition3_trans = condition3.transpose();
  EXPECT_TRUE(condition3_trans.isApprox(condition3, 1e-5))
    << "Moore-Penrose Condition 3 Failed!\n"
    << "C * C_pinv:\n"
    << condition3 << "\n"
    << "(C * C_pinv)':\n"
    << condition3_trans << "\n"
    << "Difference:\n"
    << (condition3_trans - condition3) << "\n";

  // 4. (C_pinv * C)' = C_pinv * C
  Eigen::MatrixXd condition4 = C_pinv * C;
  Eigen::MatrixXd condition4_trans = condition4.transpose();
  EXPECT_TRUE(condition4_trans.isApprox(condition4, 1e-5))
    << "Moore-Penrose Condition 4 Failed!\n"
    << "C_pinv * C:\n"
    << condition4 << "\n"
    << "(C_pinv * C)':\n"
    << condition4_trans << "\n"
    << "Difference:\n"
    << (condition4_trans - condition4) << "\n";
}

TEST(PseudoInverseTest, ZeroMatrixTest) {
  // Test with near-zero matrix
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2, 2);

  Eigen::MatrixXd D_pinv1 = crisp_controllers::pseudo_inverse(D);
  Eigen::MatrixXd D_pinv2 = crisp_controllers::pseudo_inverse_moore_penrose(D);

  // Both should return zero matrix
  EXPECT_TRUE(D_pinv1.isApprox(Eigen::MatrixXd::Zero(2, 2), 1e-5))
    << "Zero Matrix Test Failed (pseudoInverse)!\n"
    << "Result:\n"
    << D_pinv1 << "\n"
    << "Expected zero matrix:\n"
    << Eigen::MatrixXd::Zero(2, 2) << "\n"
    << "Difference:\n"
    << (D_pinv1 - Eigen::MatrixXd::Zero(2, 2)) << "\n";

  EXPECT_TRUE(D_pinv2.isApprox(Eigen::MatrixXd::Zero(2, 2), 1e-10))
    << "Zero Matrix Test Failed (pseudoInverseMoorePenrose)!\n"
    << "Result:\n"
    << D_pinv2 << "\n"
    << "Expected zero matrix:\n"
    << Eigen::MatrixXd::Zero(2, 2) << "\n"
    << "Difference:\n"
    << (D_pinv2 - Eigen::MatrixXd::Zero(2, 2)) << "\n";
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
