#include <gtest/gtest.h>
#include <Eigen/Dense>

class NullspaceControllerTest : public ::testing::Test {
protected:
  void SetUp() override {
    num_joints_ = 7;
    q_ = Eigen::VectorXd::Zero(num_joints_);
    dq_ = Eigen::VectorXd::Zero(num_joints_);
    q_init_ = Eigen::VectorXd::Zero(num_joints_);
    weights_ = Eigen::VectorXd::Ones(num_joints_);
    
    // Set some test joint positions and velocities
    q_ << 0.1, -0.2, 0.15, -0.05, 0.3, 0.1, -0.1;
    dq_ << 0.05, -0.02, 0.01, 0.0, 0.02, -0.01, 0.03;
    q_init_ << 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0;  // Different initial position
    
    // Create a simple 6x7 jacobian (end-effector wrench space to joint space)
    jacobian_ = Eigen::MatrixXd::Random(6, num_joints_);
    
    stiffness_ = 100.0;
    damping_ = 20.0;
    max_tau_ = 50.0; // Set higher to avoid limiting in basic tests
  }

  // This mimics the nullspace control logic from torque_feedback_controller
  Eigen::VectorXd compute_nullspace_torques(
      const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
      const Eigen::VectorXd& q_init, const Eigen::MatrixXd& jacobian,
      const Eigen::VectorXd& weights, double stiffness, double damping, 
      double max_tau, const std::string& projector_type = "none") {
    
    // Compute joint error and secondary task
    Eigen::VectorXd q_error = q - q_init;
    Eigen::VectorXd tau_secondary = -stiffness * weights.cwiseProduct(q_error) - 
                                    damping * weights.cwiseProduct(dq);
    
    // Compute nullspace projection
    Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(jacobian.cols(), jacobian.cols());
    Eigen::MatrixXd nullspace_projection;
    
    if (projector_type == "kinematic") {
      // Kinematic nullspace: N = I - J# * J
      Eigen::MatrixXd J_pinv = jacobian.completeOrthogonalDecomposition().pseudoInverse();
      nullspace_projection = Id - J_pinv * jacobian;
    } else {
      // No projection or other types - use identity for testing
      nullspace_projection = Id;
    }
    
    // Apply projection and torque limits
    Eigen::VectorXd tau_nullspace = nullspace_projection * tau_secondary;
    
    // Apply torque limits
    for (int i = 0; i < tau_nullspace.size(); i++) {
      tau_nullspace[i] = std::max(-max_tau, std::min(max_tau, tau_nullspace[i]));
    }
    
    return tau_nullspace;
  }

  int num_joints_;
  Eigen::VectorXd q_, dq_, q_init_, weights_;
  Eigen::MatrixXd jacobian_;
  double stiffness_, damping_, max_tau_;
};

TEST_F(NullspaceControllerTest, NoProjectionZeroError) {
  // Set current position equal to initial position
  q_ = q_init_;
  dq_.setZero();
  
  auto result = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                         stiffness_, damping_, max_tau_, "none");
  
  EXPECT_TRUE(result.isZero(1e-10)) << "Expected zero nullspace torques with zero error";
}

TEST_F(NullspaceControllerTest, ProportionalControlWithError) {
  // Reset velocities to isolate proportional term
  dq_.setZero();
  
  auto result = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                         stiffness_, damping_, max_tau_, "none");
  
  // Expected: -stiffness * weights * (q - q_init)
  Eigen::VectorXd q_error = q_ - q_init_;
  Eigen::VectorXd expected = -stiffness_ * weights_.cwiseProduct(q_error);
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Result: " << result.transpose()
    << "\nExpected: " << expected.transpose();
}

TEST_F(NullspaceControllerTest, DampingControlWithVelocity) {
  // Set positions equal to remove proportional term
  q_ = q_init_;
  
  auto result = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                         stiffness_, damping_, max_tau_, "none");
  
  // Expected: -damping * weights * dq
  Eigen::VectorXd expected = -damping_ * weights_.cwiseProduct(dq_);
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Result: " << result.transpose()
    << "\nExpected: " << expected.transpose();
}

TEST_F(NullspaceControllerTest, TorqueLimiting) {
  // Use very high stiffness to trigger torque limiting
  double high_stiffness = 1000.0;
  q_ = Eigen::VectorXd::Constant(num_joints_, 1.0);  // Large error
  dq_.setZero();
  
  auto result = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                         high_stiffness, damping_, max_tau_, "none");
  
  // All torques should be limited to max_tau
  for (int i = 0; i < result.size(); i++) {
    EXPECT_LE(std::abs(result[i]), max_tau_ + 1e-10) 
      << "Joint " << i << " torque " << result[i] << " exceeds limit " << max_tau_;
  }
}

TEST_F(NullspaceControllerTest, WeightedControl) {
  // Set different weights for different joints
  weights_ << 1.0, 0.5, 2.0, 0.0, 1.5, 0.1, 1.0;
  dq_.setZero();
  
  auto result = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                         stiffness_, damping_, max_tau_, "none");
  
  // Expected: -stiffness * weights * (q - q_init)
  Eigen::VectorXd q_error = q_ - q_init_;
  Eigen::VectorXd expected = -stiffness_ * weights_.cwiseProduct(q_error);
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Weighted control failed\nResult: " << result.transpose()
    << "\nExpected: " << expected.transpose();
  
  // Joint with zero weight should have zero torque
  EXPECT_NEAR(result[3], 0.0, 1e-10) << "Joint with zero weight should have zero nullspace torque";
}

TEST_F(NullspaceControllerTest, KinematicProjection) {
  // Test with kinematic nullspace projection
  dq_.setZero();
  
  auto result = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                         stiffness_, damping_, max_tau_, "kinematic");
  
  // Result should be different from no projection case
  auto no_projection = compute_nullspace_torques(q_, dq_, q_init_, jacobian_, weights_, 
                                                stiffness_, damping_, max_tau_, "none");
  
  // They should be different (unless jacobian is singular in a special way)
  EXPECT_FALSE(result.isApprox(no_projection, 1e-6)) 
    << "Kinematic projection should modify the nullspace torques";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}