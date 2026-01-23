// "Copyright [2026] <Analog Devices, Inc.>"
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>  // NOLINT(build/include_order)

/**
  * @brief Compute the exponential moving average of a value
  * @param output Previous output value
  * @param current Current value measured to be filtered
   * @param alpha Smoothing factor (0 < alpha < 1). The closer to 1, the more it
  * smooths the value
  * @return returns the filtered value
  */
template <typename T>
inline T exponential_moving_average(const T output, const T current, const double alpha) {
  return (1.0 - alpha) * current + alpha * output;
}

/**
  * @brief Filter the joint values from a message used by the controller
  * @param msg_names Names of the joints in the message
  * @param msg_values Values of the joints in the message. This could be either position or velocity
  * @param desired_joint_names Names of the joints to be filtered
  * @param output Output vector to store the filtered joint values
  */
template <typename FieldType>
void filterJointValues(
  const std::vector<std::string> & msg_names,
  const FieldType & msg_values,
  const std::vector<std::string> & desired_joint_names,
  Eigen::VectorXd & output) {
  std::unordered_map<std::string, size_t> name_to_index;
  for (size_t i = 0; i < desired_joint_names.size(); ++i) {
    name_to_index[desired_joint_names[i]] = i;
  }

  for (size_t i = 0; i < msg_names.size(); ++i) {
    if (msg_names[i].empty() || msg_names[i].size() > 1000) {
      continue;
    }
    auto it = name_to_index.find(msg_names[i]);
    if (it != name_to_index.end() && i < msg_values.size()) {
      output(it->second) = msg_values[i];
    }
  }
}
