#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace iskhakov_d_linear_topology {

struct Message {
  int head_process;
  int tail_process;
  std::vector<int> data;
  bool delivered;
};

using InType = Message;
using OutType = Message;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_linear_topology
