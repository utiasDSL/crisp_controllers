#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <geometry_msgs/msg/wrench_stamped.hpp>

// Test for future WrenchFeedbackController
// This tests the core logic differences from TorqueFeedbackController

class WrenchFeedbackControllerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a test jacobian (6 DOF wrench to 7 DOF joint space)
    jacobian_ = Eigen::MatrixXd::Random(6, 7);
    
    // Test wrench input - initialize properly
    wrench_input_.resize(6);
    wrench_input_ << 1.0, -0.5, 2.0,    // forces
                     0.1, 0.2, -0.3;    // torques
    
    threshold_ = 0.5;
    k_fb_ = 10.0;
    kd_ = 2.0;
    
    joint_velocities_ = Eigen::VectorXd::Random(7);
  }

  // Convert geometry_msgs::Wrench to Eigen vector
  Eigen::VectorXd wrench_to_eigen(const geometry_msgs::msg::Wrench& wrench) {
    Eigen::VectorXd w(6);
    w << wrench.force.x, wrench.force.y, wrench.force.z,
         wrench.torque.x, wrench.torque.y, wrench.torque.z;
    return w;
  }

  // Convert Eigen vector to geometry_msgs::Wrench
  geometry_msgs::msg::Wrench eigen_to_wrench(const Eigen::VectorXd& w) {
    geometry_msgs::msg::Wrench wrench;
    wrench.force.x = w[0]; wrench.force.y = w[1]; wrench.force.z = w[2];
    wrench.torque.x = w[3]; wrench.torque.y = w[4]; wrench.torque.z = w[5];
    return wrench;
  }

  // Core wrench feedback logic
  Eigen::VectorXd compute_wrench_feedback_torques(
      const Eigen::VectorXd& wrench_input,
      const Eigen::MatrixXd& jacobian,
      const Eigen::VectorXd& joint_velocities,
      double k_fb, double kd, double threshold) {
    
    // Apply threshold to wrench magnitude
    Eigen::VectorXd wrench_thresholded = Eigen::VectorXd::Zero(6);
    if (wrench_input.norm() > threshold) {
      wrench_thresholded = wrench_input;
    }
    
    // Convert wrench to joint torques using jacobian transpose
    Eigen::VectorXd tau_ext_from_wrench = jacobian.transpose() * wrench_thresholded;
    
    // Apply feedback control law
    return -k_fb * tau_ext_from_wrench - kd * joint_velocities;
  }

  Eigen::MatrixXd jacobian_;
  Eigen::VectorXd wrench_input_;
  Eigen::VectorXd joint_velocities_;
  double threshold_, k_fb_, kd_;
};

TEST_F(WrenchFeedbackControllerTest, WrenchToTorqueConversion) {
  // Test that wrench is properly converted to joint torques via jacobian transpose
  
  auto result = compute_wrench_feedback_torques(wrench_input_, jacobian_, 
                                               Eigen::VectorXd::Zero(7), // zero velocities
                                               k_fb_, 0.0, 0.0); // no damping, no threshold
  
  // Expected: tau = -k_fb * J^T * wrench
  Eigen::VectorXd expected = -k_fb_ * jacobian_.transpose() * wrench_input_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Wrench to torque conversion failed\nResult: " << result.transpose()
    << "\nExpected: " << expected.transpose();
}

TEST_F(WrenchFeedbackControllerTest, WrenchThresholdFiltering) {
  // Test wrench below threshold
  Eigen::VectorXd small_wrench = Eigen::VectorXd::Constant(6, 0.1);
  EXPECT_LT(small_wrench.norm(), threshold_);
  
  auto result = compute_wrench_feedback_torques(small_wrench, jacobian_, 
                                               joint_velocities_, k_fb_, kd_, threshold_);
  
  // Should only have damping term since wrench is filtered out
  Eigen::VectorXd expected = -kd_ * joint_velocities_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Small wrench should be filtered out by threshold";
}

TEST_F(WrenchFeedbackControllerTest, WrenchAboveThreshold) {
  // Test wrench above threshold
  EXPECT_GT(wrench_input_.norm(), threshold_);
  
  auto result = compute_wrench_feedback_torques(wrench_input_, jacobian_, 
                                               joint_velocities_, k_fb_, kd_, threshold_);
  
  // Expected: tau = -k_fb * J^T * wrench - kd * dq
  Eigen::VectorXd expected = -k_fb_ * jacobian_.transpose() * wrench_input_ - kd_ * joint_velocities_;
  
  EXPECT_TRUE(result.isApprox(expected, 1e-10))
    << "Large wrench should pass through threshold filter";
}

TEST_F(WrenchFeedbackControllerTest, GeometryMsgsConversion) {
  // Test conversion between geometry_msgs::Wrench and Eigen::VectorXd
  
  geometry_msgs::msg::Wrench wrench_msg;
  wrench_msg.force.x = 1.0; wrench_msg.force.y = 2.0; wrench_msg.force.z = 3.0;
  wrench_msg.torque.x = 0.1; wrench_msg.torque.y = 0.2; wrench_msg.torque.z = 0.3;
  
  auto eigen_wrench = wrench_to_eigen(wrench_msg);
  auto back_to_msg = eigen_to_wrench(eigen_wrench);
  
  EXPECT_NEAR(back_to_msg.force.x, wrench_msg.force.x, 1e-10);
  EXPECT_NEAR(back_to_msg.force.y, wrench_msg.force.y, 1e-10);
  EXPECT_NEAR(back_to_msg.force.z, wrench_msg.force.z, 1e-10);
  EXPECT_NEAR(back_to_msg.torque.x, wrench_msg.torque.x, 1e-10);
  EXPECT_NEAR(back_to_msg.torque.y, wrench_msg.torque.y, 1e-10);
  EXPECT_NEAR(back_to_msg.torque.z, wrench_msg.torque.z, 1e-10);
}

TEST_F(WrenchFeedbackControllerTest, ZeroWrenchZeroVelocity) {
  Eigen::VectorXd zero_wrench = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd zero_velocities = Eigen::VectorXd::Zero(7);
  
  auto result = compute_wrench_feedback_torques(zero_wrench, jacobian_, 
                                               zero_velocities, k_fb_, kd_, threshold_);
  
  EXPECT_TRUE(result.isZero()) << "Zero wrench and zero velocities should give zero torques";
}

TEST_F(WrenchFeedbackControllerTest, CompareWithTorqueController) {
  // Compare wrench controller with equivalent torque input
  
  // Convert wrench to equivalent joint torques
  Eigen::VectorXd equivalent_joint_torques = jacobian_.transpose() * wrench_input_;
  
  // Wrench controller result
  auto wrench_result = compute_wrench_feedback_torques(wrench_input_, jacobian_, 
                                                      joint_velocities_, k_fb_, kd_, 0.0); // no threshold
  
  // Equivalent torque controller result: tau = -k_fb * tau_ext - kd * dq
  Eigen::VectorXd torque_result = -k_fb_ * equivalent_joint_torques - kd_ * joint_velocities_;
  
  EXPECT_TRUE(wrench_result.isApprox(torque_result, 1e-10))
    << "Wrench controller should produce same result as torque controller with equivalent joint torques";
}

TEST_F(WrenchFeedbackControllerTest, JacobianDependency) {
  // Test that different jacobians produce different results
  Eigen::MatrixXd different_jacobian = 2.0 * jacobian_;
  
  auto result1 = compute_wrench_feedback_torques(wrench_input_, jacobian_, 
                                                Eigen::VectorXd::Zero(7), k_fb_, 0.0, 0.0);
  auto result2 = compute_wrench_feedback_torques(wrench_input_, different_jacobian, 
                                                Eigen::VectorXd::Zero(7), k_fb_, 0.0, 0.0);
  
  // Results should be different and result2 should be 2x result1
  EXPECT_TRUE(result2.isApprox(2.0 * result1, 1e-10))
    << "Scaling jacobian should scale the output torques proportionally";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}