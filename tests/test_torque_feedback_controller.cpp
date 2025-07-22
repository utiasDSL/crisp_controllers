#include <gtest/gtest.h>
#include <Eigen/Dense>

// Simplified tests for torque feedback controller logic without ROS dependencies

class TorqueFeedbackControllerLogicTest : public ::testing::Test {
protected:
  void SetUp() override {
    num_joints_ = 7;
    
    // Initialize test data
    q_.resize(num_joints_);
    dq_.resize(num_joints_);
    tau_ext_.resize(num_joints_);
    q_init_.resize(num_joints_);
    
    // Set some test values
    q_ << 0.1, -0.2, 0.15, -0.05, 0.3, 0.1, -0.1;
    dq_ << 0.05, -0.02, 0.01, 0.0, 0.02, -0.01, 0.03;
    tau_ext_ << 0.5, -0.3, 0.1, 0.0, 0.2, -0.1, 0.4;
    q_init_.setZero();
    
    k_fb_ = 10.0;
    kd_ = 2.0;
    threshold_ = 0.1;
  }

  // Core logic from torque feedback controller
  Eigen::VectorXd compute_torque_feedback_control(
      const Eigen::VectorXd& tau_ext,
      const Eigen::VectorXd& dq,
      double k_fb, double kd, double threshold) {
    
    // Apply threshold to external torques based on vector magnitude
    Eigen::VectorXd tau_ext_thresholded = Eigen::VectorXd::Zero(tau_ext.size());
    double tau_ext_magnitude = tau_ext.norm();
    if (tau_ext_magnitude > threshold) {
      tau_ext_thresholded = tau_ext;
    }

    // Compute feedback control: tau_d = -k_fb * tau_ext - kd * dq
    return -k_fb * tau_ext_thresholded - kd * dq;
  }

  int num_joints_;
  Eigen::VectorXd q_, dq_, tau_ext_, q_init_;
  double k_fb_, kd_, threshold_;
};

TEST_F(TorqueFeedbackControllerLogicTest, TorqueAboveThreshold) {
  // tau_ext should be above threshold
  EXPECT_GT(tau_ext_.norm(), threshold_);
  
  auto result = compute_torque_feedback_control(tau_ext_, dq_, k_fb_, kd_, threshold_);
  
  // Expected: tau_d = -k_fb * tau_ext - kd * dq
  Eigen::VectorXd expected = -k_fb_ * tau_ext_ - kd_ * dq_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Result: " << result.transpose()
    << "\nExpected: " << expected.transpose();
}

TEST_F(TorqueFeedbackControllerLogicTest, TorqueBelowThreshold) {
  // Use small torques below threshold
  tau_ext_ = Eigen::VectorXd::Constant(num_joints_, 0.01);
  EXPECT_LT(tau_ext_.norm(), threshold_);
  
  auto result = compute_torque_feedback_control(tau_ext_, dq_, k_fb_, kd_, threshold_);
  
  // Expected: tau_d = -kd * dq (external torques filtered out)
  Eigen::VectorXd expected = -kd_ * dq_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Small torques should be filtered out by threshold";
}

TEST_F(TorqueFeedbackControllerLogicTest, ZeroTorqueZeroVelocity) {
  tau_ext_.setZero();
  dq_.setZero();
  
  auto result = compute_torque_feedback_control(tau_ext_, dq_, k_fb_, kd_, threshold_);
  
  EXPECT_TRUE(result.isZero()) << "Zero input should give zero output";
}

TEST_F(TorqueFeedbackControllerLogicTest, ThresholdExactlyAtLimit) {
  // Create torques with norm just above threshold to avoid floating point issues
  tau_ext_.setConstant((threshold_ + 1e-6) / sqrt(num_joints_));
  EXPECT_GT(tau_ext_.norm(), threshold_);
  
  auto result = compute_torque_feedback_control(tau_ext_, dq_, k_fb_, kd_, threshold_);
  
  // Should use the external torques since they're above the threshold
  Eigen::VectorXd expected = -k_fb_ * tau_ext_ - kd_ * dq_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-8));
}

TEST_F(TorqueFeedbackControllerLogicTest, GainScaling) {
  // Test that gains properly scale the output
  double scale_factor = 2.0;
  
  auto result1 = compute_torque_feedback_control(tau_ext_, dq_, k_fb_, kd_, 0.0); // no threshold
  auto result2 = compute_torque_feedback_control(tau_ext_, dq_, scale_factor * k_fb_, scale_factor * kd_, 0.0);
  
  EXPECT_TRUE(result2.isApprox(scale_factor * result1, 1e-10))
    << "Scaling gains should scale output proportionally";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}