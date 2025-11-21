#pragma once

#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace iskhakov_d_trapezoidal_integration {

struct InType {
  double lower_level;
  double top_level;
  std::function<double(double)> function;
  int number_steps;
};

using OutType = double;

using TestType = std::tuple<InType, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_trapezoidal_integration
