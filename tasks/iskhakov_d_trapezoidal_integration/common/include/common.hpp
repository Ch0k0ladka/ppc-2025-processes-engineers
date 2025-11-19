#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace iskhakov_d_trapezoidal_integration {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_trapezoidal_integration
