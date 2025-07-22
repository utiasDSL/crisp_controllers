#include <gtest/gtest.h>
#include <Eigen/Dense>

// This will test our future FeedbackControlCore component
// For now, we'll test the logic that will be extracted

class FeedbackControlCoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize test data
    num_joints_ = 7;
    external_input_ = Eigen::VectorXd::Zero(num_joints_);
    joint_velocities_ = Eigen::VectorXd::Zero(num_joints_);
    
    // Set some test values
    external_input_ << 0.5, -0.3, 0.1, 0.0, 0.2, -0.1, 0.4;
    joint_velocities_ << 0.1, -0.05, 0.02, 0.0, 0.03, -0.01, 0.08;
    
    k_fb_ = 10.0;
    kd_ = 2.0;
    threshold_ = 0.1;
  }

  // This mimics the core feedback control logic from torque_feedback_controller
  Eigen::VectorXd compute_feedback_torques(
      const Eigen::VectorXd& external_input,
      const Eigen::VectorXd& dq,
      double k_fb, double kd, double threshold) {
    
    // Apply threshold filtering based on input magnitude
    Eigen::VectorXd input_thresholded = Eigen::VectorXd::Zero(external_input.size());
    if (external_input.norm() > threshold) {
      input_thresholded = external_input;
    }
    
    // Compute feedback torques: tau_d = -k_fb * input - kd * dq
    return -k_fb * input_thresholded - kd * dq;
  }
  
  int num_joints_;
  Eigen::VectorXd external_input_;
  Eigen::VectorXd joint_velocities_;
  double k_fb_, kd_, threshold_;
};

TEST_F(FeedbackControlCoreTest, FeedbackTorquesWithInputAboveThreshold) {
  // Input magnitude should be above threshold
  EXPECT_GT(external_input_.norm(), threshold_);
  
  auto result = compute_feedback_torques(external_input_, joint_velocities_, k_fb_, kd_, threshold_);
  
  // Expected: tau_d = -k_fb * external_input - kd * joint_velocities
  Eigen::VectorXd expected = -k_fb_ * external_input_ - kd_ * joint_velocities_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10)) 
    << "Result: " << result.transpose() 
    << "\nExpected: " << expected.transpose();
}

TEST_F(FeedbackControlCoreTest, FeedbackTorquesWithInputBelowThreshold) {
  // Use small external input below threshold
  external_input_ = Eigen::VectorXd::Constant(num_joints_, 0.01); // Small values
  EXPECT_LT(external_input_.norm(), threshold_);
  
  auto result = compute_feedback_torques(external_input_, joint_velocities_, k_fb_, kd_, threshold_);
  
  // Expected: tau_d = -kd * joint_velocities (external input should be filtered out)
  Eigen::VectorXd expected = -kd_ * joint_velocities_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Result: " << result.transpose() 
    << "\nExpected: " << expected.transpose();
}

TEST_F(FeedbackControlCoreTest, ZeroInputAndVelocities) {
  external_input_.setZero();
  joint_velocities_.setZero();
  
  auto result = compute_feedback_torques(external_input_, joint_velocities_, k_fb_, kd_, threshold_);
  
  EXPECT_TRUE(result.isZero()) << "Expected zero torques with zero input and velocities";
}

TEST_F(FeedbackControlCoreTest, ThresholdExactlyAtLimit) {
  // Create input with norm just above threshold to avoid floating point issues
  external_input_.setConstant((threshold_ + 1e-6) / sqrt(num_joints_));
  EXPECT_GT(external_input_.norm(), threshold_);
  
  auto result = compute_feedback_torques(external_input_, joint_velocities_, k_fb_, kd_, threshold_);
  
  // Should use the external input since it's above the threshold
  Eigen::VectorXd expected = -k_fb_ * external_input_ - kd_ * joint_velocities_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-8));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}