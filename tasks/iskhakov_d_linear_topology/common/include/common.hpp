#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace iskhakov_d_linear_topology {

struct Message {
  int head_process = 0;
  int tail_process = 0;
  bool delivered = false;
  std::vector<int> data;

  [[nodiscard]] int DataSize() const {
    return static_cast<int>(data.size());
  }

  void SetData(const std::vector<int> &new_data) {
    data = new_data;
  }

  void SetData(std::vector<int> &&new_data) {
    data = std::move(new_data);
  }
};

using InType = Message;
using OutType = Message;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace iskhakov_d_linear_topology
