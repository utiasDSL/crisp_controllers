#pragma once

#include <cmath>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace crisp_controllers {

/**
 * @brief Expand a nullspace gain parameter into one gain per degree of freedom.
 *
 * A parameter holding a single value is broadcast to all degrees of freedom.
 * Otherwise, it has to provide one value per degree of freedom already.
 *
 * @param values Gain values as given in the parameters.
 * @param nv Number of degrees of freedom of the model.
 * @return The expanded gains, or std::nullopt if values has neither 1 nor nv elements.
 */
inline std::optional<Eigen::VectorXd>
expand_nullspace_gains(const std::vector<double> & values, Eigen::Index nv) {
  if (values.size() == 1) {
    return Eigen::VectorXd::Constant(nv, values.front());
  }
  if (static_cast<Eigen::Index>(values.size()) == nv) {
    return Eigen::Map<const Eigen::VectorXd>(values.data(), nv);
  }
  return std::nullopt;
}

/**
 * @brief Resolve the damping gains, replacing negative entries by their critical damping.
 *
 * A negative damping gain means "pick a sensible value for me", which is 2 sqrt(k) for the
 * stiffness k of that same degree of freedom. Zero is a valid request for no damping at all.
 *
 * @param damping Damping gains, one per degree of freedom.
 * @param stiffness Stiffness gains, one per degree of freedom.
 * @return The resolved damping gains.
 */
inline Eigen::VectorXd
resolve_nullspace_damping(const Eigen::VectorXd & damping, const Eigen::VectorXd & stiffness) {
  Eigen::VectorXd resolved(damping.size());
  for (Eigen::Index i = 0; i < damping.size(); ++i) {
    resolved[i] = damping[i] < 0.0 ? 2.0 * std::sqrt(stiffness[i]) : damping[i];
  }
  return resolved;
}

}  // namespace crisp_controllers
